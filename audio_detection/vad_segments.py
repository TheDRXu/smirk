#!/usr/bin/env python3
"""
vad_segments.py

Produce speech segments (start,end) from an input WAV using webrtcvad.

Usage examples:
  # default: writes CSV into outputs/segments_vad.csv
  python3 vad_segments.py --wav path/to/audio.wav --out_dir outputs

  # specify exact CSV output file
  python3 vad_segments.py --wav path/to/audio.wav --out_csv /tmp/my_segments.csv

Requirements:
  pip install webrtcvad librosa soundfile numpy pandas
"""
import argparse
from pathlib import Path

import numpy as np
import librosa
import webrtcvad
import pandas as pd

def read_audio_mono_16k(wav_path, target_sr=16000):
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    # ensure float32 in range -1..1
    y = y.astype('float32')
    return y, target_sr

def frame_generator(frame_ms, audio, sample_rate):
    """
    Yield successive frames of audio as raw 16-bit PCM bytes.
    frame_ms: frame duration in milliseconds (10,20,30 recommended)
    """
    n_samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    offset = 0
    total = len(audio)
    while offset + n_samples_per_frame <= total:
        chunk = audio[offset:offset + n_samples_per_frame]
        # convert float32 [-1,1] to int16 pcm
        pcm = (chunk * 32767).astype(np.int16).tobytes()
        start_s = offset / sample_rate
        yield start_s, pcm
        offset += n_samples_per_frame

def vad_collector(sample_rate, frame_ms, aggressiveness, audio):
    """
    Simple VAD collector: returns list of tuples (frame_start_s, is_speech_bool).
    """
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = list(frame_generator(frame_ms, audio, sample_rate))
    speech_flags = []
    for start_s, frame_bytes in frames:
        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
        except Exception:
            # in case of malformed frame, assume non-speech
            is_speech = False
        speech_flags.append((start_s, is_speech))
    return speech_flags

def merge_speech_frames_to_segments(speech_flags, frame_ms, min_silence_ms=200, pad_ms=200, audio_duration=None):
    """
    speech_flags: list of (start_s, bool)
    Merge consecutive speech frames into segments, then merge segments separated by gap < min_silence_ms.
    Add pad_ms padding to each segment (clamped to audio bounds).
    Returns list of (start,end).
    """
    frame_dur = frame_ms / 1000.0
    segments = []
    in_seg = False
    seg_start = None
    for i, (start_s, is_speech) in enumerate(speech_flags):
        if is_speech and not in_seg:
            in_seg = True
            seg_start = start_s
        elif not is_speech and in_seg:
            in_seg = False
            seg_end = start_s  # end at start of this (non-speech) frame
            segments.append((seg_start, seg_end))
            seg_start = None
    # if ended in speech
    if in_seg and seg_start is not None:
        last_start, _ = speech_flags[-1]
        seg_end = last_start + frame_dur
        segments.append((seg_start, seg_end))

    if not segments:
        return []

    # merge small gaps
    merged = []
    prev_s, prev_e = segments[0]
    for s, e in segments[1:]:
        gap = s - prev_e
        if gap * 1000.0 <= min_silence_ms:
            # merge
            prev_e = e
        else:
            merged.append((prev_s, prev_e))
            prev_s, prev_e = s, e
    merged.append((prev_s, prev_e))

    # pad
    padded = []
    dur = audio_duration if audio_duration is not None else float('inf')
    pad = pad_ms / 1000.0
    for s, e in merged:
        s2 = max(0.0, s - pad)
        e2 = min(dur, e + pad)
        # ensure e2 > s2
        if e2 - s2 >= 0.01:
            padded.append((s2, e2))
    return padded

def write_segments_csv(segments, out_dir=None, out_csv=None):
    """
    Write segments to CSV.
    - If out_csv provided => write CSV there.
    - Otherwise write into out_dir with default filename segments_vad.csv.
    Returns Path to CSV file.
    """
    entries = [{"start": round(s, 6), "end": round(e, 6), "text": ""} for s, e in segments]

    if out_csv:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(out_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv_path = Path(out_dir) / "segments_vad.csv"

    pd.DataFrame(entries).to_csv(out_csv_path, index=False)
    return out_csv_path

def main():
    p = argparse.ArgumentParser(description="VAD -> speech segments")
    p.add_argument("--wav", required=True, help="Path to WAV file")
    p.add_argument("--out_dir", default="outputs", help="Output folder (used when --out_csv not provided)")
    p.add_argument("--out_csv", default=None, help="Path to output CSV file (overrides out_dir)")
    p.add_argument("--aggressiveness", type=int, default=3, choices=[0,1,2,3], help="webrtcvad aggressiveness (0-3), higher = more aggressive")
    p.add_argument("--frame_ms", type=int, default=30, choices=[10,20,30], help="frame size in ms")
    p.add_argument("--min_silence_ms", type=int, default=200, help="merge segments separated by gap < this (ms)")
    p.add_argument("--pad_ms", type=int, default=200, help="pad each segment by this many ms")
    args = p.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    print("[INFO] Loading audio and resampling to 16 kHz mono (librosa)...")
    audio, sr = read_audio_mono_16k(wav_path, target_sr=16000)
    duration = len(audio) / float(sr)
    print(f"[INFO] audio duration: {duration:.2f} s, sample_rate={sr}")

    print("[INFO] Running VAD frames...")
    speech_flags = vad_collector(sr, args.frame_ms, args.aggressiveness, audio)
    n_frames = len(speech_flags)
    n_speech_frames = sum(1 for _, f in speech_flags if f)
    print(f"[INFO] frames: {n_frames}, speech frames: {n_speech_frames}")

    print("[INFO] Merging frames into segments...")
    segments = merge_speech_frames_to_segments(speech_flags, args.frame_ms, args.min_silence_ms, args.pad_ms, audio_duration=duration)
    print(f"[INFO] detected {len(segments)} speech segments")

    csv_path = write_segments_csv(segments, out_dir=args.out_dir, out_csv=args.out_csv)
    print(f"[DONE] Wrote segments CSV to: {csv_path}")

    if segments:
        print("Example segments:")
        for s,e in segments[:10]:
            print(f"  {s:.3f} - {e:.3f}  ({e-s:.3f}s)")

if __name__ == "__main__":
    main()
