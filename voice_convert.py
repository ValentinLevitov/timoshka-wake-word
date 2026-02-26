#!/usr/bin/env python3
"""
Voice conversion script for Тимошка wake word training.

Takes TTS-generated samples and converts them using target voice recordings
from Mozilla Common Voice (or any short WAV clips of real speakers).
Uses Coqui TTS FreeVC24 — a multilingual voice conversion model.

Usage:
    python voice_convert.py \
        --source-dir tts_positive/ \
        --target-dir voice_references/ \
        --output-dir converted_positive/ \
        --device cuda

Expects:
    - source-dir: WAV files from piper-sample-generator (TTS output)
    - target-dir: WAV files of real speakers (5-15 sec each, 16kHz mono)
    - output-dir: will be created, one converted file per (source, target) pair
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio


def resample_if_needed(wav_path: str, target_sr: int = 16000) -> str:
    """Ensure WAV is 16kHz mono. Returns path (may rewrite in-place)."""
    waveform, sr = torchaudio.load(wav_path)
    changed = False

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        changed = True

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        changed = True

    if changed:
        torchaudio.save(wav_path, waveform, target_sr)

    return wav_path


def convert_batch(
    source_dir: str,
    target_dir: str,
    output_dir: str,
    device: str = "cuda",
    max_targets: int = 0,
    resume: bool = True,
):
    """Convert all source samples with all target voices."""
    from TTS.api import TTS

    os.makedirs(output_dir, exist_ok=True)

    source_files = sorted(
        glob.glob(os.path.join(source_dir, "*.wav"))
    )
    target_files = sorted(
        glob.glob(os.path.join(target_dir, "*.wav"))
    )

    if not source_files:
        print(f"ERROR: No WAV files found in {source_dir}")
        sys.exit(1)
    if not target_files:
        print(f"ERROR: No WAV files found in {target_dir}")
        sys.exit(1)

    if max_targets > 0:
        target_files = target_files[:max_targets]

    total = len(source_files) * len(target_files)
    print(f"Source samples: {len(source_files)}")
    print(f"Target voices:  {len(target_files)}")
    print(f"Total conversions: {total}")
    print()

    # Load model once
    print("Loading FreeVC24 model...")
    tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to(device)
    print("Model loaded.\n")

    done = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, src in enumerate(source_files):
        src_name = Path(src).stem
        for j, tgt in enumerate(target_files):
            tgt_name = Path(tgt).stem
            out_path = os.path.join(output_dir, f"{src_name}_vc{tgt_name}.wav")

            if resume and os.path.exists(out_path):
                skipped += 1
                done += 1
                continue

            try:
                tts.voice_conversion_to_file(
                    source_wav=src,
                    target_wav=tgt,
                    file_path=out_path,
                )
                done += 1
            except Exception as e:
                print(f"  ERROR converting {src_name} + {tgt_name}: {e}")
                errors += 1
                done += 1

            # Progress every 100 conversions
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = (done - skipped) / max(elapsed, 1)
                eta = (total - done) / max(rate, 0.01)
                print(
                    f"  [{done}/{total}] "
                    f"{rate:.1f} conv/s, "
                    f"ETA {eta/60:.0f} min, "
                    f"errors={errors}, skipped={skipped}"
                )

    elapsed = time.time() - t0
    print(f"\nDone! {done - skipped - errors} converted, "
          f"{skipped} skipped, {errors} errors in {elapsed/60:.1f} min")


def prepare_common_voice_targets(
    cv_dir: str,
    output_dir: str,
    max_clips: int = 100,
    min_duration_s: float = 3.0,
    max_duration_s: float = 15.0,
):
    """
    Select and prepare target voice clips from a Common Voice dataset export.

    cv_dir: path to extracted Common Voice dataset (contains clips/ and validated.tsv)
    output_dir: where to save prepared clips
    """
    import csv
    import random

    os.makedirs(output_dir, exist_ok=True)

    tsv_path = os.path.join(cv_dir, "validated.tsv")
    clips_dir = os.path.join(cv_dir, "clips")

    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found. Download Russian Common Voice dataset.")
        sys.exit(1)

    # Read metadata, pick one clip per unique speaker
    speakers = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            client_id = row["client_id"]
            if client_id not in speakers:
                speakers[client_id] = row["path"]

    # Shuffle and select
    items = list(speakers.items())
    random.shuffle(items)

    saved = 0
    for client_id, clip_name in items:
        if saved >= max_clips:
            break

        # Common Voice may have .mp3 files
        clip_path = os.path.join(clips_dir, clip_name)
        if not os.path.exists(clip_path):
            # Try with .mp3 extension
            mp3_path = clip_path.replace(".wav", ".mp3")
            if os.path.exists(mp3_path):
                clip_path = mp3_path
            else:
                continue

        try:
            waveform, sr = torchaudio.load(clip_path)
            duration = waveform.shape[1] / sr

            if duration < min_duration_s or duration > max_duration_s:
                continue

            # Convert to 16kHz mono WAV
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            out_path = os.path.join(output_dir, f"speaker_{saved:04d}.wav")
            torchaudio.save(out_path, waveform, 16000)
            saved += 1

        except Exception as e:
            print(f"  Skipping {clip_name}: {e}")

    print(f"Prepared {saved} target voice clips in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Voice conversion for wake word training"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- convert command ---
    p_convert = subparsers.add_parser(
        "convert", help="Run voice conversion on TTS samples"
    )
    p_convert.add_argument(
        "--source-dir", required=True, help="Directory with TTS WAV samples"
    )
    p_convert.add_argument(
        "--target-dir", required=True, help="Directory with target voice WAV files"
    )
    p_convert.add_argument(
        "--output-dir", required=True, help="Output directory for converted samples"
    )
    p_convert.add_argument(
        "--device", default="cuda", help="Device: cuda or cpu (default: cuda)"
    )
    p_convert.add_argument(
        "--max-targets", type=int, default=0,
        help="Limit number of target voices (0 = all)"
    )
    p_convert.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip already converted files"
    )

    # --- prepare-targets command ---
    p_prep = subparsers.add_parser(
        "prepare-targets",
        help="Prepare target voices from Common Voice dataset"
    )
    p_prep.add_argument(
        "--cv-dir", required=True,
        help="Path to extracted Common Voice dataset"
    )
    p_prep.add_argument(
        "--output-dir", required=True,
        help="Output directory for prepared clips"
    )
    p_prep.add_argument(
        "--max-clips", type=int, default=100,
        help="Maximum number of target clips (default: 100)"
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_batch(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            device=args.device,
            max_targets=args.max_targets,
            resume=not args.no_resume,
        )
    elif args.command == "prepare-targets":
        prepare_common_voice_targets(
            cv_dir=args.cv_dir,
            output_dir=args.output_dir,
            max_clips=args.max_clips,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
