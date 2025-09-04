#!/usr/bin/env python3
"""
train_and_predict_ecapa.py  (fixed: simple 0.5 decisioning)

Behavior:
 - Training & model saving unchanged from before.
 - Prediction stage now applies a simple threshold: prob_doctor >= 0.5 -> Doctor, else Patient.
 - Embedding extraction, LOSO, prototype code left intact (no functional change),
   but prototypes/hysteresis/smoothing are NOT applied to final decision by default.
"""
import argparse
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from scipy.signal import medfilt

# ----------------- EDIT THIS PAIRS LIST (or supply --pairs_json) -----------------
PAIRS = [
    {'wav': 'dataset/audio/2.wav', 'labels': 'dataset/output/vad_labeled_2.csv'},
    {'wav': 'dataset/audio/3.wav', 'labels': 'dataset/output/vad_labeled_3.csv'},
    {'wav': 'dataset/audio/4.wav', 'labels': 'dataset/output/vad_labeled_4.csv'}
]
# --------------------------------------------------------------------------------

CACHE_DIR = Path("embeddings_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# PARAMETERS you can tune quickly
MIN_SEG_SEC = 0.5        # skip segments shorter than this (or extend)
MIN_RMS = 1e-4          # skip very quiet segments
PROTO_FUSION_WEIGHT = 0.25  # how much to trust session prototypes (unused for final simple decision)
SMOOTH_KERNEL = 3       # median filter kernel (odd) (unused for final simple decision)
HYSTERESIS_DELTA = 0.12 # hysteresis distance (unused for final simple decision)


def make_ecapa():
    try:
        from speechbrain.pretrained import SpeakerRecognition
    except Exception as e:
        raise RuntimeError("speechbrain not installed or torch missing. Install with pip and a compatible torch.") from e
    model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                           savedir="pretrained_models/spkrec-ecapa-voxceleb")
    return model


def extract_ecapa_embedding(model, wav_path, start_s, end_s, sr=16000, cache_key=None, min_sec=MIN_SEG_SEC):
    """
    Extract ecapa embedding for segment. Skips segments that are too short or too quiet by returning None.
    """
    import librosa, soundfile as sf, tempfile, os, numpy as np
    duration = max(0.01, float(end_s) - float(start_s))
    read_duration = max(duration, min_sec)
    y, _ = librosa.load(wav_path, sr=sr, offset=max(0, float(start_s)), duration=read_duration, mono=True)
    if y.shape[0] < int(0.2 * sr):
        pad_len = int(max(0, 0.5 * sr - y.shape[0]))
        if pad_len > 0:
            y = np.concatenate([y, np.zeros(pad_len)], axis=0)

    rms = float(np.sqrt(np.mean(y ** 2) + 1e-16))
    if duration < 0.15 or rms < MIN_RMS:
        # mark as invalid / too short / too quiet
        return None

    fd, tmppath = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmppath, y, sr, format="WAV", subtype="PCM_16")

    emb = None
    try:
        if hasattr(model, "encode_file"):
            emb_tensor = model.encode_file(tmppath)
            emb = emb_tensor.squeeze().detach().cpu().numpy()
        elif hasattr(model, "encode_batch"):
            import torch
            wav_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            try:
                emb_tensor = model.encode_batch(wav_t)
            except Exception:
                wav_t2 = wav_t.unsqueeze(1) if wav_t.ndim == 2 else wav_t
                emb_tensor = model.encode_batch(wav_t2)
            emb = emb_tensor.squeeze().detach().cpu().numpy()
        elif hasattr(model, "get_embedding"):
            try:
                emb_tensor = model.get_embedding(tmppath)
                emb = emb_tensor.squeeze().detach().cpu().numpy()
            except Exception:
                import torch
                wav_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                emb_tensor = model.get_embedding(wav_t)
                emb = emb_tensor.squeeze().detach().cpu().numpy()
        elif hasattr(model, "encode_utterance"):
            import torch
            wav_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            emb_tensor = model.encode_utterance(wav_t)
            emb = emb_tensor.squeeze().detach().cpu().numpy()
        else:
            raise RuntimeError("SpeechBrain model instance lacks recognized embedding methods.")
    finally:
        try:
            os.remove(tmppath)
        except Exception:
            pass

    if emb is None:
        return None

    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def build_dataset(pairs, verbose=True):
    model = make_ecapa()
    rows = []
    for p in pairs:
        wav = p['wav']; csvp = p['labels']
        if not Path(wav).exists():
            raise FileNotFoundError(f"WAV not found: {wav}")
        if not Path(csvp).exists():
            raise FileNotFoundError(f"labels CSV not found: {csvp}")
        session = Path(wav).stem
        df = pd.read_csv(csvp)
        good = 0
        for idx, r in df.iterrows():
            lbl = str(r.get('label', '')).strip()
            if lbl == '' or lbl.lower() in ['nan', 'none']:
                continue
            start = float(r['start']); end = float(r['end'])
            y = 1 if lbl.lower().startswith('d') else 0
            cache_key = f"{session}_{idx}"
            cache_file = CACHE_DIR / (cache_key + ".npy")
            if cache_file.exists():
                try:
                    emb = np.load(cache_file)
                except Exception:
                    emb = None
            else:
                emb = extract_ecapa_embedding(model, wav, start, end, cache_key=cache_key)
                if emb is not None:
                    np.save(cache_file, emb)
            if emb is None:
                # skip low-quality segment
                continue
            rows.append({'session': session, 'start': start, 'end': end, 'label': lbl, 'y': y, 'embedding': emb})
            good += 1
        if verbose:
            print(f"Processed session {session}, loaded {len(df)} rows, kept {good} usable segments.")
    df_out = pd.DataFrame(rows)
    if verbose:
        print("Built dataset with", len(df_out), "examples across", df_out['session'].nunique(), "sessions.")
    return df_out


def make_clf_gridsearch():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'))])
    grid = {'clf__C': [0.05, 0.2, 0.5, 1.0, 2.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    return gs


def choose_threshold_patient_f1(probs, y):
    """
    Choose threshold for doctor-probabilities `probs` that maximizes patient F1 on dev.
    y: 1=Doctor, 0=Patient
    (This is still used during LOSO tuning, but final prediction uses 0.5 rule.)
    """
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        patient_pred = (probs < t).astype(int)  # 1 means predicted patient
        patient_true = (y == 0).astype(int)
        f1 = f1_score(patient_true, patient_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def loso_eval_and_train(df, save_model_path="role_model_ecapa.joblib"):
    sessions = df['session'].unique().tolist()
    all_y_true = []; all_y_pred = []
    thr_folds = []
    for test_s in sessions:
        train_df = df[df['session'] != test_s]
        test_df = df[df['session'] == test_s]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_all = np.vstack(train_df['embedding'].values)
        y_all = train_df['y'].values
        X_train_core, X_dev, y_train_core, y_dev = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)

        gs = make_clf_gridsearch()
        gs.fit(X_train_core, y_train_core)
        clf = gs.best_estimator_
        print(f"[LOSO][{test_s}] best C = {gs.best_params_['clf__C']}")

        probs_dev = clf.predict_proba(X_dev)[:, 1]
        thr_fold = choose_threshold_patient_f1(probs_dev, y_dev)
        thr_folds.append(thr_fold)

        X_test = np.vstack(test_df['embedding'].values)
        y_test = test_df['y'].values
        probs_test = clf.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= thr_fold).astype(int)
        acc = accuracy_score(y_test, preds_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, preds_test, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs_test)
        except Exception:
            auc = float('nan')
        print(f"LOSO test session={test_s} | n_test={len(y_test)} | acc={acc:.3f} | f1={f:.3f} | auc={auc:.3f} | thr={thr_fold:.3f}")
        all_y_true.extend(y_test.tolist()); all_y_pred.extend(preds_test.tolist())

    if all_y_true:
        acc_overall = accuracy_score(all_y_true, all_y_pred)
        p, r, f, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
        print(f"\nOverall LOSO acc={acc_overall:.3f} f1={f:.3f}")

    X = np.vstack(df['embedding'].values)
    y = df['y'].values
    gs_final = make_clf_gridsearch()
    gs_final.fit(X, y)
    final_pipe = gs_final.best_estimator_
    if len(thr_folds) > 0:
        chosen_threshold = float(np.median(thr_folds))
    else:
        chosen_threshold = 0.5
    save_pack = {'model': final_pipe, 'threshold': float(chosen_threshold)}
    joblib.dump(save_pack, save_model_path)
    print("Saved final model + threshold to", save_model_path, "| threshold =", chosen_threshold)
    return final_pipe, chosen_threshold


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_session_prototypes(embs, probs, thr):
    """
    Build doctor/patient prototypes from embeddings with confident LR probs.
    (Kept for inspection/experimentation but not used for final 0.5 decision.)
    """
    doc_embs = [e for e, p in zip(embs, probs) if p >= thr + 0.10]
    pat_embs = [e for e, p in zip(embs, probs) if p <= thr - 0.10]
    if len(doc_embs) < 2 or len(pat_embs) < 2:
        return None, None
    proto_d = np.mean(np.vstack(doc_embs), axis=0)
    proto_p = np.mean(np.vstack(pat_embs), axis=0)
    return proto_d, proto_p


def predict_on_segments(model_pack_or_model, predict_wav, predict_segments_csv, out_csv):
    # handle model pack format or old-style estimator
    if isinstance(model_pack_or_model, dict):
        model = model_pack_or_model['model']
        # we keep the saved threshold in the file for analysis/compatibility,
        # but the prediction decision below uses the fixed 0.5 rule as requested.
        saved_thr = float(model_pack_or_model.get('threshold', 0.5))
    else:
        model = model_pack_or_model
        saved_thr = 0.5

    model_ecapa = make_ecapa()
    seg_df = pd.read_csv(predict_segments_csv)
    emb_list = []
    rows = []
    for idx, r in seg_df.iterrows():
        s = float(r['start']); e = float(r['end'])
        cache_key = f"predict_{Path(predict_wav).stem}_{idx}"
        cache_file = CACHE_DIR / (cache_key + ".npy")
        emb = None
        if cache_file.exists():
            try:
                emb = np.load(cache_file)
            except Exception:
                emb = None
        if emb is None:
            emb = extract_ecapa_embedding(model_ecapa, predict_wav, s, e, cache_key=cache_key)
            if emb is not None:
                np.save(cache_file, emb)
        if emb is None:
            rows.append({'start': s, 'end': e, 'pred': 'Unknown', 'prob_doctor': float('nan')})
            emb_list.append(None)
        else:
            emb_list.append(emb)
            rows.append(None)

    valid_embs = [e for e in emb_list if e is not None]
    if len(valid_embs) == 0:
        print("[WARN] No valid embeddings extracted for prediction.")
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_csv, index=False)
        print("Wrote predictions to", out_csv)
        return out_df

    X = np.vstack(valid_embs)
    probs = model.predict_proba(X)[:, 1]

    # NOTE: we compute prototypes for diagnostics but DO NOT use them to change final labels.
    try:
        proto_d, proto_p = build_session_prototypes(X, probs, saved_thr)
    except Exception:
        proto_d = proto_p = None

    # In the fixed behavior below we use the **simple 0.5 threshold rule**
    # (user requested: if prob_doctor < 0.5 => Patient).
    fused_probs = np.array([float(p) for p in probs], dtype=float)

    # map fused_probs back to full-length array aligned with seg_df indices
    all_probs = np.full(len(emb_list), np.nan, dtype=float)
    vi = 0
    for i, emb in enumerate(emb_list):
        if emb is not None:
            all_probs[i] = fused_probs[vi]; vi += 1

    # interpolate NaNs from neighbors (simple linear/neighbour hold)
    n = len(all_probs)
    for i in range(n):
        if np.isnan(all_probs[i]):
            left = i - 1
            while left >= 0 and np.isnan(all_probs[left]):
                left -= 1
            right = i + 1
            while right < n and np.isnan(all_probs[right]):
                right += 1
            if left >= 0 and right < n:
                all_probs[i] = 0.5 * (all_probs[left] + all_probs[right])
            elif left >= 0:
                all_probs[i] = all_probs[left]
            elif right < n:
                all_probs[i] = all_probs[right]
            else:
                all_probs[i] = 0.5  # fallback

    # --- FIXED DECISION RULE: SIMPLE 0.5 THRESHOLD ---
    decision_threshold = 0.5
    binary = (all_probs >= decision_threshold).astype(int)

    # write rows
    out_rows = []
    for i in range(len(seg_df)):
        s = float(seg_df.loc[i, 'start']); e = float(seg_df.loc[i, 'end'])
        prob = float(all_probs[i]) if not np.isnan(all_probs[i]) else float('nan')
        pred_label = 'Doctor' if int(binary[i]) == 1 else 'Patient'
        out_rows.append({'start': s, 'end': e, 'pred': pred_label, 'prob_doctor': prob})
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print("Wrote predictions to", out_csv)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_json", default=None, help="Optional JSON file with list of pairs (wav,labels). If given, overrides PAIRS in-script.")
    ap.add_argument("--predict_wav", default=None, help="Path to WAV to predict (e.g. dataset/audio/5.wav)")
    ap.add_argument("--predict_segments", default=None, help="CSV of segments for predict_wav (start,end). You can generate with vad_segments.py")
    ap.add_argument("--out_preds", default="predictions.csv", help="Output CSV for predictions")
    ap.add_argument("--save_model", default="role_model_ecapa.joblib", help="Where to save or load trained model")
    ap.add_argument("--force_train", action="store_true", help="Force retraining even if saved model exists")
    ap.add_argument("--train_only", action="store_true", help="Train (or retrain) and exit without predicting")
    ap.add_argument("--predict_only", action="store_true", help="Only predict; require an existing saved model (error if missing)")
    args = ap.parse_args()

    pairs = PAIRS
    if args.pairs_json:
        pairs = pd.read_json(args.pairs_json).to_dict(orient='records')

    save_model_path = Path(args.save_model)

    model_pipe = None
    # Decide: load existing model or train
    if save_model_path.exists() and not args.force_train:
        if args.predict_only or args.predict_wav:
            print(f"[INFO] Loading existing model from {save_model_path}")
            pack = joblib.load(save_model_path)
            if isinstance(pack, dict) and 'model' in pack:
                model_pipe = pack
            else:
                # backward compatibility: old model-only files
                model_pipe = {'model': pack, 'threshold': 0.5}
        else:
            print(f"[INFO] Found existing model at {save_model_path}; loading it (use --force_train to retrain).")
            pack = joblib.load(save_model_path)
            if isinstance(pack, dict) and 'model' in pack:
                model_pipe = pack
            else:
                model_pipe = {'model': pack, 'threshold': 0.5}

    if model_pipe is None:
        # No saved model or force_train -> build dataset & train
        print("[INFO] No existing model loaded (missing or --force_train). Building dataset and training now...")
        df = build_dataset(pairs)
        if df.shape[0] == 0:
            raise RuntimeError("No labeled examples found in PAIRS. Cannot train.")
        final_pipe, chosen_thr = loso_eval_and_train(df, save_model_path=str(save_model_path))
        model_pipe = {'model': final_pipe, 'threshold': chosen_thr}

    # If user only wanted to train, exit now
    if args.train_only:
        print("[INFO] train_only set — exiting after training/saving model.")
        return

    # Prediction stage
    if args.predict_only and not save_model_path.exists() and model_pipe is None:
        raise RuntimeError(f"--predict_only was specified but no saved model found at {save_model_path}")

    if args.predict_wav and args.predict_segments:
        if model_pipe is None:
            if save_model_path.exists():
                pack = joblib.load(save_model_path)
                if isinstance(pack, dict) and 'model' in pack:
                    model_pipe = pack
                else:
                    model_pipe = {'model': pack, 'threshold': 0.5}
            else:
                raise RuntimeError("No model available to run prediction. Provide --save_model path or train first.")
        predict_on_segments(model_pipe, args.predict_wav, args.predict_segments, args.out_preds)
    else:
        print("No prediction requested (provide --predict_wav and --predict_segments).")


if __name__ == "__main__":
    main()
