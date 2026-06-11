// TNT microphone helper: native AVFoundation capture, isolated in a child
// process so the parent can always reclaim the mic by killing it.
//
// Streams raw little-endian 16-bit mono PCM (default 16 kHz) to stdout.
//
// Protocol with the parent:
//   stderr  "TNT_READY"          capture is running, PCM follows on stdout
//   stderr  "TNT_ERROR: <msg>"   startup failed, process exits non-zero
//   SIGTERM / SIGINT             stop capture and exit 0
//   stdin EOF                    parent died: stop capture and exit 0
//
// Usage:
//   tnt-mic [--rate 16000] [--device <index|name-substring|uid>]
//   tnt-mic --list

import AVFoundation
import CoreAudio
import Foundation

func emitError(_ msg: String) -> Never {
    FileHandle.standardError.write("TNT_ERROR: \(msg)\n".data(using: .utf8)!)
    exit(1)
}

/// Write the whole buffer to a file descriptor; exit quietly if the pipe dies.
func writeAll(_ fd: Int32, _ base: UnsafeRawPointer, _ count: Int) {
    var offset = 0
    while offset < count {
        let n = write(fd, base.advanced(by: offset), count - offset)
        if n > 0 {
            offset += n
        } else if errno != EINTR {
            // Parent closed the pipe or died; stop holding the microphone.
            exit(0)
        }
    }
}

// MARK: - CoreAudio device discovery

struct InputDevice {
    let id: AudioDeviceID
    let uid: String
    let name: String
}

func deviceStringProperty(
    _ id: AudioDeviceID, _ selector: AudioObjectPropertySelector
) -> String? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var value: CFString?
    var size = UInt32(MemoryLayout<CFString?>.size)
    let status = withUnsafeMutablePointer(to: &value) {
        AudioObjectGetPropertyData(id, &address, 0, nil, &size, $0)
    }
    guard status == noErr, let value else { return nil }
    return value as String
}

func inputChannelCount(_ id: AudioDeviceID) -> Int {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: kAudioDevicePropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr,
        size > 0
    else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(
        byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr
    else { return 0 }
    let buffers = UnsafeMutableAudioBufferListPointer(
        raw.assumingMemoryBound(to: AudioBufferList.self))
    return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func listInputDevices() -> [InputDevice] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    let systemObject = AudioObjectID(kAudioObjectSystemObject)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(systemObject, &address, 0, nil, &size) == noErr
    else { return [] }
    var ids = [AudioDeviceID](
        repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(systemObject, &address, 0, nil, &size, &ids) == noErr
    else { return [] }

    return ids.compactMap { id in
        guard inputChannelCount(id) > 0 else { return nil }
        return InputDevice(
            id: id,
            uid: deviceStringProperty(id, kAudioDevicePropertyDeviceUID) ?? "",
            name: deviceStringProperty(id, kAudioObjectPropertyName) ?? "device-\(id)")
    }
}

func matchDevice(_ devices: [InputDevice], _ requested: String) -> InputDevice? {
    if let index = Int(requested), devices.indices.contains(index) {
        return devices[index]
    }
    let needle = requested.lowercased()
    if let exact = devices.first(where: {
        $0.uid.lowercased() == needle || $0.name.lowercased() == needle
    }) {
        return exact
    }
    return devices.first(where: { $0.name.lowercased().contains(needle) })
}

// MARK: - Argument parsing

var requestedDevice: String?
var sampleRate = 16000.0
var listOnly = false

var argIterator = CommandLine.arguments.dropFirst().makeIterator()
while let arg = argIterator.next() {
    switch arg {
    case "--list":
        listOnly = true
    case "--device":
        guard let value = argIterator.next() else { emitError("--device needs a value") }
        requestedDevice = value
    case "--rate":
        guard let value = argIterator.next(), let rate = Double(value), rate > 0
        else { emitError("--rate needs a positive number") }
        sampleRate = rate
    default:
        emitError("unknown argument: \(arg)")
    }
}

if listOnly {
    for (index, device) in listInputDevices().enumerated() {
        print("Input device \(index): \(device.name) (uid=\(device.uid))")
    }
    exit(0)
}

// MARK: - Capture

let engine = AVAudioEngine()
let input = engine.inputNode

if let requested = requestedDevice {
    let devices = listInputDevices()
    guard let device = matchDevice(devices, requested) else {
        let names = devices.map { $0.name }.joined(separator: ", ")
        emitError("no input device matching '\(requested)' (available: \(names))")
    }
    guard let unit = input.audioUnit else {
        emitError("input node has no audio unit; cannot select device")
    }
    var deviceID = device.id
    let status = AudioUnitSetProperty(
        unit,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &deviceID,
        UInt32(MemoryLayout<AudioDeviceID>.size))
    guard status == noErr else {
        emitError("failed to select input device '\(device.name)' (OSStatus \(status))")
    }
}

let hardwareFormat = input.inputFormat(forBus: 0)
guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 else {
    emitError(
        "no usable input device; check that a microphone is connected and that "
            + "this terminal has microphone permission "
            + "(System Settings > Privacy & Security > Microphone)")
}

guard
    let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: sampleRate,
        channels: 1,
        interleaved: true),
    let converter = AVAudioConverter(from: hardwareFormat, to: outputFormat)
else {
    emitError("cannot convert \(hardwareFormat) to \(Int(sampleRate)) Hz mono int16")
}

input.installTap(onBus: 0, bufferSize: 2048, format: hardwareFormat) { buffer, _ in
    let ratio = outputFormat.sampleRate / hardwareFormat.sampleRate
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
    guard
        let converted = AVAudioPCMBuffer(
            pcmFormat: outputFormat, frameCapacity: capacity)
    else { return }

    // Feed exactly this tap buffer; the converter keeps resampler state
    // across calls, so leftovers surface in the next callback.
    var consumed = false
    var conversionError: NSError?
    let status = converter.convert(to: converted, error: &conversionError) {
        _, inputStatus in
        if consumed {
            inputStatus.pointee = .noDataNow
            return nil
        }
        consumed = true
        inputStatus.pointee = .haveData
        return buffer
    }
    guard status != .error, converted.frameLength > 0,
        let samples = converted.int16ChannelData
    else { return }
    writeAll(1, samples[0], Int(converted.frameLength) * 2)
}

engine.prepare()
do {
    try engine.start()
} catch {
    emitError("audio engine failed to start: \(error.localizedDescription)")
}

FileHandle.standardError.write("TNT_READY\n".data(using: .utf8)!)

func shutdown() -> Never {
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    exit(0)
}

// Stop on SIGTERM/SIGINT via dispatch sources (safe to touch the engine there).
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
signal(SIGPIPE, SIG_IGN)
let signalSources = [SIGTERM, SIGINT].map { sig -> DispatchSourceSignal in
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { shutdown() }
    source.resume()
    return source
}
_ = signalSources

// Stop when stdin reaches EOF: the parent holds the write end open for our
// whole lifetime, so EOF means it died and the mic must be released.
DispatchQueue.global().async {
    var scratch = UInt8(0)
    while true {
        let n = read(0, &scratch, 1)
        if n == 0 || (n < 0 && errno != EINTR) {
            DispatchQueue.main.async { shutdown() }
            return
        }
    }
}

dispatchMain()
