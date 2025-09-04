#!/usr/bin/env python3
"""
transcribe_wav.py
Usage:
  python3 transcribe_wav.py --wav /path/to/audio.wav --model base --out_dir outputs
"""
import argparse, json
from pathlib import Path
import whisper
import pandas as pd

def transcribe(wav, model_name='base'):
    print(f"[INFO] Loading Whisper model '{model_name}' ...")
    model = whisper.load_model(model_name)
    print("[INFO] Transcribing ...")
    result = model.transcribe(str(wav), verbose=False)
    segments = []
    if 'segments' in result:
        for s in result['segments']:
            segments.append({'start': float(s['start']), 'end': float(s['end']), 'text': s['text'].strip()})
    else:
        segments.append({'start': 0.0, 'end': result.get('duration', 0.0), 'text': result.get('text','').strip()})
    return segments

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav', required=True, help='Path to WAV file')
    p.add_argument('--model', default='base', help='Whisper model (tiny, small, base, large)')
    p.add_argument('--out_dir', default='outputs', help='Output folder')
    args = p.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    segments = transcribe(args.wav, model_name=args.model)
    json_path = outdir / 'segments.json'
    csv_path  = outdir / 'segments.csv'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)
    pd.DataFrame(segments).to_csv(csv_path, index=False)
    print(f"[DONE] Wrote {len(segments)} segments to {json_path} and {csv_path}")

if __name__ == '__main__':
    main()