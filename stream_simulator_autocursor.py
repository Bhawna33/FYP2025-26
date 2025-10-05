#!/usr/bin/env python3
"""
stream_simulator_autocursor.py

Self-contained demo for a BCI "streaming" simulator that:
 - Generates a small synthetic EEG-like dataset (left vs right motor imagery)
 - Preprocesses (bandpass), extracts PSD bandpower features (alpha/beta)
 - Trains a small SVM classifier (with scaling)
 - Simulates streaming predictions epoch-by-epoch
 - Optionally moves the system cursor left/right based on predictions
 
USAGE example (to actually move your cursor):
  pip install numpy scipy scikit-learn joblib pyautogui
  python stream_simulator_autocursor.py --do-move --step 0.25

By default --do-move is OFF for safety; pass --do-move to enable real cursor moves.
"""

import argparse
import time
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# ---------------------------
# Preprocessing
# ---------------------------
def bandpass(data, fs, lo=7, hi=30, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def preprocess_epochs(X, fs):
    return bandpass(X, fs)

# ---------------------------
# Feature extraction (bandpower features)
# ---------------------------
def bandpower_feature(epoch, fs, band):
    """Compute average band power for an epoch for each channel, then average channels."""
    f, Pxx = welch(epoch, fs=fs, nperseg=min(256, epoch.size))
    # select band
    idx = np.logical_and(f >= band[0], f <= band[1])
    # integrate PSD over band (area) -> approximate by mean
    return np.mean(np.trapz(Pxx[idx], f[idx]))

def psd_features(X, fs):
    """
    For each epoch compute two features: alpha (8-12 Hz) and beta (13-30 Hz) average bandpower.
    X shape expected: (n_epochs, n_channels, n_samples)
    Returns (n_epochs, 2)
    """
    bands = {'alpha': (8, 12), 'beta': (13, 30)}
    feats = []
    for epoch in X:
        # epoch: (n_channels, n_samples)
        ch_feats = []
        for ch in epoch:
            # compute band power per channel for each band
            ap = bandpower_feature(ch, fs, bands['alpha'])
            bp = bandpower_feature(ch, fs, bands['beta'])
            ch_feats.append((ap, bp))
        # average across channels
        ch_feats = np.array(ch_feats)
        feats.append(np.mean(ch_feats, axis=0))
    return np.array(feats)  # shape (n_epochs, 2)

# ---------------------------
# Synthetic dataset generator
# ---------------------------
def generate_synthetic_eeg(n_epochs=200, n_channels=4, epoch_sec=1.0, fs=128, random_state=42):
    """
    Create a synthetic EEG-like dataset with two classes (0:left, 1:right).
    Class 0: stronger alpha (10 Hz) on channel 0
    Class 1: stronger beta (20 Hz) on channel 1
    """
    rng = np.random.RandomState(random_state)
    n_samples = int(epoch_sec * fs)
    X = np.zeros((n_epochs, n_channels, n_samples), dtype=float)
    y = np.zeros(n_epochs, dtype=int)
    for i in range(n_epochs):
        lab = int(i < n_epochs // 2)  # first half class 1? we'll alternate for balance
        if rng.rand() > 0.5:
            lab = 1 - lab
        y[i] = lab
        t = np.arange(n_samples) / fs
        # baseline noise for all channels
        for ch in range(n_channels):
            X[i, ch] = 0.5 * rng.randn(n_samples)
        # add class-specific oscillation
        if lab == 0:
            # alpha 10 Hz in channel 0
            X[i, 0] += 1.0 * np.sin(2 * np.pi * 10 * t) + 0.2 * rng.randn(n_samples)
        else:
            # beta 20 Hz in channel 1
            X[i, 1] += 1.0 * np.sin(2 * np.pi * 20 * t) + 0.2 * rng.randn(n_samples)
    return X, y, fs

# ---------------------------
# Action mapping and execution
# ---------------------------
def action_from_label(label):
    mapping = {0: "left", 1: "right"}
    return mapping.get(label, f"unknown({label})")

def execute_action(action, do_move=False, step_px=30):
    """If do_move True, move the mouse cursor left/right by step_px using pyautogui."""
    if not do_move:
        print(f"[SIM] would execute action: {action} (dry run)")
        return
    try:
        import pyautogui
    except Exception as e:
        print("pyautogui not available. Install with: pip install pyautogui")
        print("Exception:", e)
        return
    # get current position
    x, y = pyautogui.position()
    if action == "left":
        pyautogui.moveTo(max(0, x - step_px), y)
    elif action == "right":
        screenW, screenH = pyautogui.size()
        pyautogui.moveTo(min(screenW - 1, x + step_px), y)
    else:
        pass  # no-op for unknown actions

# ---------------------------
# Main streaming simulator
# ---------------------------
def main(args):
    # 1) Prepare (or load) dataset
    print("Generating synthetic dataset...")
    X, y, fs = generate_synthetic_eeg(n_epochs=args.n_epochs, n_channels=args.n_channels,
                                      epoch_sec=args.epoch_sec, fs=args.fs, random_state=args.seed)

    # 2) Preprocess and extract features (for training)
    print("Preprocessing and extracting features...")
    Xf = preprocess_epochs(X, fs)
    F = psd_features(Xf, fs)  # shape (n_epochs, 2)

    # 3) Train-test split and train classifier
    print("Training classifier...")
    scaler = StandardScaler()
    Fs = scaler.fit_transform(F)
    Xtr, Xte, ytr, yte = train_test_split(Fs, y, test_size=0.25, random_state=args.seed, stratify=y)
    clf = SVC(kernel='rbf', probability=True, random_state=args.seed)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Validation accuracy (synthetic data): {acc:.3f}")
    print(classification_report(yte, ypred))

    # Optionally save model to file
    if args.save_model:
        dump({'model': clf, 'scaler': scaler}, args.save_model)
        print("Saved model to", args.save_model)

    # 4) Streaming loop: use trained classifier to predict on each epoch sequentially
    print("Starting streaming simulation... (press Ctrl+C to stop)")
    # We'll iterate through the *same* epochs in F (simulating streaming). In real use replace with live epochs.
    for i in range(F.shape[0]):
        feat = scaler.transform(F[i:i+1])
        probs = clf.predict_proba(feat)[0]
        pred = int(np.argmax(probs))
        action = action_from_label(pred)
        print(f"{i:04d} -> pred={pred} action={action} proba={probs}")
        execute_action(action, do_move=args.do_move, step_px=args.step_px)
        time.sleep(args.step)  # pacing between epochs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BCI stream simulator that can move the cursor based on predictions.")
    p.add_argument('--n-epochs', type=int, default=200, help='Number of synthetic epochs to generate')
    p.add_argument('--n-channels', type=int, default=4, help='Channels per epoch')
    p.add_argument('--epoch-sec', type=float, default=1.0, help='Epoch length in seconds')
    p.add_argument('--fs', type=int, default=128, help='Sampling frequency (Hz)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed')
    p.add_argument('--step', type=float, default=0.25, help='Seconds between simulated epochs')
    p.add_argument('--step-px', dest='step_px', type=int, default=40, help='Pixels to move the cursor per action')
    p.add_argument('--do-move', dest='do_move', action='store_true', help='If set, will move the real cursor using pyautogui')
    p.add_argument('--save-model', dest='save_model', default=None, help='Optionally save trained model to file (joblib)')
    args = p.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("Streaming stopped by user. Exiting.")
