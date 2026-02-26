#!/usr/bin/env python3
"""
Test script for the trained Тимошка wake word model.

Listens to the microphone in real-time and prints detection scores.
Requires: pip install openwakeword pyaudio numpy

Usage:
    python test_timoshka.py --model timoshka.tflite
    python test_timoshka.py --model timoshka.tflite --threshold 0.5
    python test_timoshka.py --model timoshka.tflite --test-file test.wav
"""

import argparse
import sys
import time

import numpy as np


def test_from_microphone(model_path: str, threshold: float = 0.5):
    """Real-time microphone test with live score display."""
    import pyaudio
    from openwakeword.model import Model

    CHUNK = 1280  # 80ms at 16kHz — required by openWakeWord
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    model = Model(wakeword_models=[model_path])
    model_name = list(model.models.keys())[0]

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"Model: {model_path}")
    print(f"Wake word: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Listening... (Ctrl+C to stop)\n")

    detections = 0
    start_time = time.time()

    try:
        while True:
            audio = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )

            prediction = model.predict(audio)
            score = prediction[model_name]

            # Visual bar
            bar_len = int(score * 50)
            bar = "#" * bar_len + "." * (50 - bar_len)

            if score >= threshold:
                detections += 1
                elapsed = time.time() - start_time
                print(
                    f"\r  DETECTED! [{bar}] {score:.3f}  "
                    f"(#{detections}, {elapsed:.0f}s elapsed)"
                )
                print()  # New line after detection
            else:
                print(f"\r  [{bar}] {score:.3f}", end="", flush=True)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nStopped after {elapsed:.0f}s")
        print(f"Total detections: {detections}")
        if elapsed > 60:
            rate = detections / (elapsed / 3600)
            print(f"Detection rate: {rate:.1f}/hour")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def test_from_file(model_path: str, wav_path: str, threshold: float = 0.5):
    """Test model against a WAV file."""
    import wave

    from openwakeword.model import Model

    model = Model(wakeword_models=[model_path])
    model_name = list(model.models.keys())[0]

    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getframerate() == 16000, "WAV must be 16kHz"

        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    print(f"Model: {model_path}")
    print(f"File: {wav_path}")
    print(f"Duration: {len(audio)/16000:.1f}s")
    print(f"Threshold: {threshold}")
    print()

    # Process in 80ms chunks
    chunk_size = 1280
    detections = []

    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i : i + chunk_size]
        prediction = model.predict(chunk)
        score = prediction[model_name]

        if score >= threshold:
            time_s = i / 16000
            detections.append((time_s, score))
            print(f"  DETECTED at {time_s:.2f}s — score {score:.3f}")

    if not detections:
        print("  No detections.")
    else:
        print(f"\nTotal: {len(detections)} detection(s)")


def test_adversarial(
    model_path: str, test_dir: str, threshold: float = 0.5
):
    """Test model against a directory of adversarial negative WAV files."""
    import glob
    import wave

    from openwakeword.model import Model

    model = Model(wakeword_models=[model_path])
    model_name = list(model.models.keys())[0]

    wav_files = sorted(glob.glob(f"{test_dir}/*.wav"))
    if not wav_files:
        print(f"No WAV files found in {test_dir}")
        return

    print(f"Testing {len(wav_files)} adversarial samples...")
    print(f"Threshold: {threshold}\n")

    false_positives = 0
    total_tested = 0

    for wav_path in wav_files:
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            # Resample if needed
            if len(audio) == 0:
                continue

            chunk_size = 1280
            max_score = 0.0

            model.reset()
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i : i + chunk_size]
                prediction = model.predict(chunk)
                score = prediction[model_name]
                max_score = max(max_score, score)

            total_tested += 1
            if max_score >= threshold:
                false_positives += 1
                fname = wav_path.split("/")[-1]
                print(f"  FALSE POSITIVE: {fname} (score={max_score:.3f})")

        except Exception as e:
            print(f"  Error processing {wav_path}: {e}")

    fp_rate = false_positives / max(total_tested, 1) * 100
    print(f"\nResults: {false_positives}/{total_tested} false positives ({fp_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test Тимошка wake word model"
    )
    parser.add_argument(
        "--model", required=True, help="Path to .tflite model file"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold 0.0-1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--test-file", help="Test against a WAV file instead of microphone"
    )
    parser.add_argument(
        "--test-adversarial",
        help="Test against a directory of adversarial WAV files"
    )

    args = parser.parse_args()

    if args.test_adversarial:
        test_adversarial(args.model, args.test_adversarial, args.threshold)
    elif args.test_file:
        test_from_file(args.model, args.test_file, args.threshold)
    else:
        test_from_microphone(args.model, args.threshold)


if __name__ == "__main__":
    main()
