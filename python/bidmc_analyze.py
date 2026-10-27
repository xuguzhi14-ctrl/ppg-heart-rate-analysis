"""
bidmc_analyze.py - evaluate my PPG heart-rate algorithm on the BIDMC clinical dataset.

For every recording:
    1. Load the PPG ("PLETH", 125 Hz) and the reference heart rate measured from the
       ECG by the hospital monitor ("HR", 1 Hz)
    2. Split the PPG into windows (default 10 s, sliding every 2 s)
    3. Estimate HR in each window with my algorithm (band-pass filter + peak detection,
       and FFT as an alternative method)
    4. Compare with the mean ECG reference HR over the same window

Outputs (in ../results/):
    per_recording.csv        mean absolute error (MAE) for each patient
    summary.txt              overall accuracy figures
    bland_altman.png         agreement plot (my estimate vs ECG reference)
    error_by_recording.png   MAE for each patient
    example_bidmc_XX.png     raw/filtered signal with detected beats for one recording

Usage:
    python bidmc_analyze.py
    python bidmc_analyze.py --window 8 --low 0.7 --high 3.5      # try other settings
    python bidmc_analyze.py --example 5                          # plot recording 05 in detail
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg import bandpass, detect_peaks, fft_hr, peaks_hr

HERE = os.path.dirname(os.path.abspath(__file__))


def load_recording(sig_path):
    num_path = sig_path.replace("_Signals.csv", "_Numerics.csv")
    sig = pd.read_csv(sig_path, skipinitialspace=True)
    num = pd.read_csv(num_path, skipinitialspace=True)
    sig.columns = [c.strip() for c in sig.columns]
    num.columns = [c.strip() for c in num.columns]
    t = sig["Time [s]"].to_numpy(dtype=float)
    ppg = sig["PLETH"].to_numpy(dtype=float)
    t_ref = num["Time [s]"].to_numpy(dtype=float)
    hr_ref = pd.to_numeric(num["HR"], errors="coerce").to_numpy(dtype=float)
    # Average sampling rate over the whole recording. (The CSV timestamps are rounded,
    # so the median of neighbouring differences is biased: it gave 100 Hz instead of 125 Hz.)
    fs = (len(t) - 1) / (t[-1] - t[0])
    return t, ppg, fs, t_ref, hr_ref


def evaluate_recording(t, ppg, fs, t_ref, hr_ref, window, step, low, high):
    # Filter the whole recording once (avoids edge effects in every window)
    x_filt = bandpass(ppg, fs, low, high)
    peaks = detect_peaks(x_filt, fs)
    t_beats, hr_beats = peaks_hr(t, peaks)

    rows = []
    start = t[0]
    while start + window <= t[-1]:
        end = start + window
        ref_mask = (t_ref >= start) & (t_ref < end)
        ref = np.nanmean(hr_ref[ref_mask]) if ref_mask.any() else np.nan

        beat_mask = (t_beats >= start) & (t_beats < end)
        est_peaks = np.median(hr_beats[beat_mask]) if beat_mask.sum() >= 2 else np.nan

        sig_mask = (t >= start) & (t < end)
        est_fft = fft_hr(x_filt[sig_mask], fs, low, high)

        if not np.isnan(ref) and ref > 0:
            rows.append({"t_start": start, "ref": ref, "est_peaks": est_peaks, "est_fft": est_fft})
        start += step
    return pd.DataFrame(rows), x_filt, peaks


def plot_example(name, t, ppg, x_filt, peaks, figpath, seconds=20):
    m = t <= t[0] + seconds
    pk = peaks[t[peaks] <= t[0] + seconds]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t[m], ppg[m], color="tab:gray", lw=0.9)
    ax[0].set_title(f"{name}: raw PPG (first {seconds} s)")
    ax[1].plot(t[m], x_filt[m], color="tab:blue", lw=0.9, label="Band-pass filtered")
    ax[1].plot(t[pk], x_filt[pk], "rx", ms=7, label="Detected beats")
    ax[1].set_title("Filtered PPG with detected heartbeats")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figpath, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate PPG HR algorithm on BIDMC dataset")
    p.add_argument("--data", default=os.path.join(HERE, "..", "data", "bidmc_csv"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    p.add_argument("--window", type=float, default=10.0, help="Window length (s)")
    p.add_argument("--step", type=float, default=2.0, help="Window step (s)")
    p.add_argument("--low", type=float, default=0.5, help="Band-pass low cut-off (Hz)")
    p.add_argument("--high", type=float, default=4.0, help="Band-pass high cut-off (Hz)")
    p.add_argument("--example", type=int, default=1, help="Recording number to plot in detail")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "bidmc_*_Signals.csv")))
    if not files:
        print(f"No data found in {args.data}. Run download_bidmc.py first.")
        return
    os.makedirs(args.out, exist_ok=True)

    per_rec, all_windows = [], []
    for f in files:
        name = os.path.basename(f).replace("_Signals.csv", "")
        t, ppg, fs, t_ref, hr_ref = load_recording(f)
        df, x_filt, peaks = evaluate_recording(t, ppg, fs, t_ref, hr_ref,
                                               args.window, args.step, args.low, args.high)
        if df.empty:
            print(f"{name}: no usable windows")
            continue
        df["recording"] = name
        all_windows.append(df)

        err_p = (df["est_peaks"] - df["ref"]).abs()
        err_f = (df["est_fft"] - df["ref"]).abs()
        per_rec.append({
            "recording": name,
            "windows": len(df),
            "ref_mean_bpm": df["ref"].mean(),
            "mae_peaks_bpm": err_p.mean(),
            "mae_fft_bpm": err_f.mean(),
            "pct_within_5bpm_peaks": 100 * (err_p <= 5).mean(),
        })
        print(f"{name}: ref {df['ref'].mean():5.1f} bpm | MAE peaks {err_p.mean():5.2f} | "
              f"MAE FFT {err_f.mean():5.2f} bpm")

        if int(name.split("_")[1]) == args.example:
            plot_example(name, t, ppg, x_filt, peaks, os.path.join(args.out, f"example_{name}.png"))

    rec = pd.DataFrame(per_rec)
    win = pd.concat(all_windows, ignore_index=True)
    rec.to_csv(os.path.join(args.out, "per_recording.csv"), index=False, float_format="%.2f")

    # ---------- overall summary ----------
    diff = (win["est_peaks"] - win["ref"]).dropna()
    abs_p = diff.abs()
    abs_f = (win["est_fft"] - win["ref"]).abs().dropna()
    bias, sd = diff.mean(), diff.std()
    lines = [
        f"Settings: window {args.window}s, step {args.step}s, band-pass {args.low}-{args.high} Hz",
        f"Recordings analysed: {len(rec)}   Windows: {len(win)}",
        "",
        "Peak-detection method:",
        f"  MAE                 {abs_p.mean():.2f} bpm",
        f"  Median abs error    {abs_p.median():.2f} bpm",
        f"  Windows within 5bpm {100 * (abs_p <= 5).mean():.1f}%",
        f"  Bias (est - ref)    {bias:+.2f} bpm, 95% limits of agreement "
        f"{bias - 1.96 * sd:+.1f} to {bias + 1.96 * sd:+.1f} bpm",
        "",
        "FFT method:",
        f"  MAE                 {abs_f.mean():.2f} bpm",
        f"  Windows within 5bpm {100 * (abs_f <= 5).mean():.1f}%",
        "",
        f"Worst recordings (peaks MAE): "
        + ", ".join(f"{r.recording} ({r.mae_peaks_bpm:.1f})"
                    for r in rec.nlargest(3, "mae_peaks_bpm").itertuples()),
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    # ---------- Bland-Altman plot ----------
    w = win.dropna(subset=["est_peaks"])
    mean_ = (w["est_peaks"] + w["ref"]) / 2
    d = w["est_peaks"] - w["ref"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(mean_, d, s=6, alpha=0.4, color="tab:blue")
    ax.axhline(bias, color="k", lw=1, label=f"Bias {bias:+.2f} bpm")
    for k in (-1.96, 1.96):
        ax.axhline(bias + k * sd, color="tab:red", ls="--", lw=1)
    ax.set_title("Bland-Altman: PPG algorithm vs ECG reference heart rate")
    ax.set_xlabel("Mean of the two methods (bpm)")
    ax.set_ylabel("Difference, algorithm - reference (bpm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bland_altman.png"), dpi=150)
    plt.close(fig)

    # ---------- per-recording error ----------
    fig, ax = plt.subplots(figsize=(12, 4.5))
    idx = np.arange(len(rec))
    ax.bar(idx - 0.2, rec["mae_peaks_bpm"], width=0.4, label="Peak detection")
    ax.bar(idx + 0.2, rec["mae_fft_bpm"], width=0.4, label="FFT")
    ax.set_xticks(idx)
    ax.set_xticklabels([r.split("_")[1] for r in rec["recording"]], fontsize=7)
    ax.set_xlabel("Recording")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("Heart-rate error per patient recording")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_by_recording.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
