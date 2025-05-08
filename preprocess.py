#!/usr/bin/env python3
import os
import pickle
import numpy as np
import mne
from tqdm import tqdm
import argparse


def parse_tse_bi(path):
    """
    Parse a .tse_bi file and return list of (start, end, label) tuples.
    """
    lines = open(path).read().splitlines()[2:]
    return [(float(s), float(e), lab)
            for s,e,lab,_ in (ln.split() for ln in lines)]


def process_one(edf_path, sfreq=200, win_s=4, step_s=1):
    """
    Load an EDF, filter 0.5–99Hz via IIR, resample, and slide windows.
    Returns: segments, binary labels, onset/offset tags, subject IDs.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # Band-pass 0.5–99 Hz using 4th-order Butterworth IIR
    raw.filter(
        l_freq=0.5,
        h_freq=99.0,
        method='iir',
        iir_params={'order': 4, 'ftype': 'butter'},
        verbose=False
    )
    raw.resample(sfreq, npad='auto')
    data = raw.get_data()

    labels = parse_tse_bi(edf_path.replace('.edf', '.tse_bi'))
    win_samples  = int(win_s  * sfreq)
    step_samples = int(step_s * sfreq)

    subj_dir = os.path.basename(os.path.dirname(edf_path))
    subj = int(subj_dir.split('_')[0])

    segs, labs, tags, subjs = [], [], [], []
    for (s,e,lab) in labels:
        start_i = int(s * sfreq)
        end_i   = int(e * sfreq)
        length  = end_i - start_i
        n_steps = max(1, (length - win_samples)//step_samples + 1)

        for i in range(n_steps):
            st = start_i + i * step_samples
            en = st + win_samples
            w  = data[:, st:en]
            # per-channel z-score
            w  = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)

            # binary label
            labs.append(1 if lab!='bckg' else 0)
            # onset/offset tagging
            rel_start = st/sfreq - s
            rel_end   = e - (en/sfreq)
            if abs(rel_start) < step_s:
                tag = 1  # ictal_onset
            elif abs(rel_end) < step_s:
                tag = 2  # ictal_offset
            elif lab!='bckg':
                tag = 3  # ictal
            else:
                tag = 0  # non-ictal
            tags.append(tag)
            subjs.append(subj)
            segs.append(w)

    return segs, labs, tags, subjs


def main(input_dir, output_file):
    all_X, all_y, all_t, all_s = [], [], [], []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.endswith('.edf'):
                segs, labs, tags, subs = process_one(os.path.join(root, fn))
                all_X.extend(segs)
                all_y.extend(labs)
                all_t.extend(tags)
                all_s.extend(subs)

    X    = np.stack(all_X, axis=0)
    y    = np.array(all_y, dtype=np.int64)
    t    = np.array(all_t, dtype=np.int64)
    subj = np.array(all_s, dtype=np.int64)

    # Create output dir if needed
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump({'eeg': X, 'label': y, 'tag': t, 'subj': subj}, f)

    print(f"Saved {X.shape[0]} windows -> {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess synthetic EEG data for seizure detection"
    )
    parser.add_argument('--input_dir', '-i', default='synthetic_dataset/train',
                        help='Root folder of EDF/.tse_bi files')
    parser.add_argument('--output', '-o', default='preproc.pkl',
                        help='Output pickle filename')
    args = parser.parse_args()
    main(args.input_dir, args.output)
