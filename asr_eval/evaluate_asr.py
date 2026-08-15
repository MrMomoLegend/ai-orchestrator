"""
evaluate_asr.py — Whisper vs Vosk word error rate on the project's own audio.

Produces every number and figure needed for Section 5.6 of the report.
Rerun it any time the audio, the Whisper model size, or the beam width
changes; nothing here is done by hand.

Usage
-----
    python evaluate_asr.py                      # both engines
    python evaluate_asr.py --engines whisper    # skip Vosk
    python evaluate_asr.py --whisper-size small.en

Expects
-------
    references.csv          clip_id, filename, group, speaker, condition, reference
    audio/<filename>        16 kHz mono 16-bit PCM WAV (see the Day 2 guide)
    models/vosk-model-small-en-us-0.15/   (only if running Vosk)

Writes
------
    results/asr_transcripts.csv     every clip, every engine, side by side
    results/asr_wer_by_group.csv    the table that goes in the report
    results/asr_contrasts.csv       the three controlled comparisons
    results/fig_wer_by_group.png    300 dpi, ready to paste into Chapter 5
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"
RESULTS_DIR = HERE / "results"
VOSK_MODEL_DIR = HERE / "models" / "vosk-model-small-en-us-0.15"

# Must match WHISPER_BEAM in main.py. If these drift apart, the reported word
# error rate stops describing the system that actually ships.
WHISPER_BEAM = 5

GROUP_ORDER = ["own_accent", "non_native", "noisy", "technical"]
GROUP_LABELS = {
    "own_accent": "Own accent\n(quiet)",
    "non_native": "Non-native\nspeaker",
    "noisy": "Background\nnoise",
    "technical": "Technical\nvocabulary",
}


# --------------------------------------------------------------------------
# Text normalisation
#
# Applied identically to references and to both engines' output. Without it
# you are measuring punctuation and capitalisation conventions rather than
# recognition: Whisper emits "What is RAG?" and Vosk emits "what is rag",
# which is a formatting difference, not an error.
# --------------------------------------------------------------------------
def normalise(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Audio checks
# --------------------------------------------------------------------------
def check_wav(path: Path) -> str:
    """Return '' if fine, else a human-readable complaint."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.getnframes()
    except Exception as exc:
        return f"not a readable WAV ({exc})"

    problems = []
    if ch != 1:
        problems.append(f"{ch} channels (need mono)")
    if rate != 16000:
        problems.append(f"{rate} Hz (Vosk needs 16000)")
    if width != 2:
        problems.append(f"{width * 8}-bit (need 16-bit PCM)")
    if frames / max(rate, 1) < 0.5:
        problems.append("shorter than 0.5s")
    return "; ".join(problems)


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------
def run_whisper(paths, size):
    from faster_whisper import WhisperModel

    print(f"[whisper] loading {size} ...")
    t0 = time.time()
    model = WhisperModel(size, device="cpu", compute_type="int8")
    print(f"[whisper] loaded in {time.time() - t0:.1f}s")

    out = {}
    for i, (clip_id, path) in enumerate(paths, 1):
        t = time.time()
        segments, _ = model.transcribe(str(path), beam_size=WHISPER_BEAM, language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
        out[clip_id] = {"text": text, "seconds": round(time.time() - t, 2)}
        print(f"  [{i:>2}/{len(paths)}] {clip_id}: {text[:60]}")
    return out


def run_vosk(paths, model_dir):
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    print(f"[vosk] loading {model_dir.name} ...")
    t0 = time.time()
    model = Model(str(model_dir))
    print(f"[vosk] loaded in {time.time() - t0:.1f}s")

    out = {}
    for i, (clip_id, path) in enumerate(paths, 1):
        t = time.time()
        with wave.open(str(path), "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(False)
            pieces = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    pieces.append(json.loads(rec.Result()).get("text", ""))
            pieces.append(json.loads(rec.FinalResult()).get("text", ""))
        text = " ".join(p for p in pieces if p).strip()
        out[clip_id] = {"text": text, "seconds": round(time.time() - t, 2)}
        print(f"  [{i:>2}/{len(paths)}] {clip_id}: {text[:60]}")
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(refs, hyps):
    """Corpus-level WER plus the error breakdown, over aligned lists."""
    import jiwer

    if not refs:
        return {"wer": float("nan"), "S": 0, "D": 0, "I": 0, "H": 0, "N": 0}
    o = jiwer.process_words(refs, hyps)
    return {
        "wer": round(o.wer, 4),
        "S": o.substitutions,
        "D": o.deletions,
        "I": o.insertions,
        "H": o.hits,
        "N": o.substitutions + o.deletions + o.hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="whisper,vosk")
    ap.add_argument("--whisper-size", default="base.en")
    ap.add_argument("--references", default="references.csv")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    RESULTS_DIR.mkdir(exist_ok=True)

    ref_path = HERE / args.references
    if not ref_path.exists():
        sys.exit(f"ERROR: {ref_path} not found.")
    df = pd.read_csv(ref_path)

    # ---- check every clip exists and is in the right format -------------
    missing, malformed, paths = [], [], []
    for _, row in df.iterrows():
        path = AUDIO_DIR / row["filename"]
        if not path.exists():
            missing.append(row["filename"])
            continue
        complaint = check_wav(path)
        if complaint:
            malformed.append(f"{row['filename']}: {complaint}")
        paths.append((row["clip_id"], path))

    if missing:
        print(f"\nMISSING {len(missing)} clip(s) in {AUDIO_DIR}:")
        for m in missing:
            print("   -", m)
    if malformed:
        print(f"\nWRONG FORMAT ({len(malformed)}):")
        for m in malformed:
            print("   -", m)
        print("\nRe-export from Audacity as WAV, 16000 Hz, mono, 16-bit PCM.")
        if "vosk" in engines:
            sys.exit("Vosk cannot read these. Fix the audio and rerun.")
    if not paths:
        sys.exit("No usable audio found. Nothing to do.")

    print(f"\n{len(paths)} clip(s) ready.\n")

    # ---- transcribe -----------------------------------------------------
    results = {}
    if "whisper" in engines:
        results["whisper"] = run_whisper(paths, args.whisper_size)
    if "vosk" in engines:
        if not VOSK_MODEL_DIR.exists():
            print(f"\nSKIPPING Vosk: model not found at {VOSK_MODEL_DIR}")
            print("Download vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models")
            print("and unzip it into models/.\n")
        else:
            results["vosk"] = run_vosk(paths, VOSK_MODEL_DIR)

    if not results:
        sys.exit("No engine produced output.")

    # ---- per-clip table -------------------------------------------------
    df["reference_norm"] = df["reference"].map(normalise)
    for eng, out in results.items():
        df[f"{eng}_raw"] = df["clip_id"].map(lambda c: out.get(c, {}).get("text", ""))
        df[f"{eng}_norm"] = df[f"{eng}_raw"].map(normalise)
        df[f"{eng}_seconds"] = df["clip_id"].map(
            lambda c: out.get(c, {}).get("seconds", float("nan"))
        )

    import jiwer

    scored = df[df["clip_id"].isin({c for c, _ in paths})].copy()
    for eng in results:
        scored[f"{eng}_wer"] = [
            round(jiwer.wer([r], [h]), 4) if r else float("nan")
            for r, h in zip(scored["reference_norm"], scored[f"{eng}_norm"])
        ]

    scored.to_csv(RESULTS_DIR / "asr_transcripts.csv", index=False)

    # ---- WER by group ---------------------------------------------------
    rows = []
    for group in GROUP_ORDER:
        sub = scored[scored["group"] == group]
        if sub.empty:
            continue
        row = {"group": group, "n_clips": len(sub)}
        for eng in results:
            s = score(list(sub["reference_norm"]), list(sub[f"{eng}_norm"]))
            row[f"{eng}_wer"] = s["wer"]
            row[f"{eng}_sub"] = s["S"]
            row[f"{eng}_del"] = s["D"]
            row[f"{eng}_ins"] = s["I"]
            row[f"{eng}_mean_s"] = round(sub[f"{eng}_seconds"].mean(), 2)
        rows.append(row)

    overall = {"group": "ALL", "n_clips": len(scored)}
    for eng in results:
        s = score(list(scored["reference_norm"]), list(scored[f"{eng}_norm"]))
        overall[f"{eng}_wer"] = s["wer"]
        overall[f"{eng}_sub"] = s["S"]
        overall[f"{eng}_del"] = s["D"]
        overall[f"{eng}_ins"] = s["I"]
        overall[f"{eng}_mean_s"] = round(scored[f"{eng}_seconds"].mean(), 2)
    rows.append(overall)

    by_group = pd.DataFrame(rows)
    by_group.to_csv(RESULTS_DIR / "asr_wer_by_group.csv", index=False)

    # ---- the three controlled contrasts ---------------------------------
    # Every pair below reads the *same sentences*, so exactly one variable
    # changes. That is what makes these differences interpretable rather
    # than merely descriptive.
    contrasts = [
        ("Accent", "own_accent", "non_native", "same 8 sentences, different speaker"),
        ("Noise", "own_accent", "noisy", "same speaker, 5 sentences re-recorded with noise"),
        ("Vocabulary", "own_accent", "technical", "same speaker and conditions, domain terms"),
    ]
    crows = []
    for name, base, comp, note in contrasts:
        b = scored[scored["group"] == base]
        c = scored[scored["group"] == comp]
        if b.empty or c.empty:
            continue
        # For noise, restrict the baseline to the same five sentences.
        if comp == "noisy":
            ids = set(c["clip_id"].str.replace("noise_", "own_", regex=False))
            b = b[b["clip_id"].isin(ids)]
        row = {"contrast": name, "baseline": base, "comparison": comp, "note": note}
        for eng in results:
            wb = score(list(b["reference_norm"]), list(b[f"{eng}_norm"]))["wer"]
            wc = score(list(c["reference_norm"]), list(c[f"{eng}_norm"]))["wer"]
            row[f"{eng}_baseline_wer"] = wb
            row[f"{eng}_comparison_wer"] = wc
            row[f"{eng}_delta"] = round(wc - wb, 4)
        crows.append(row)
    pd.DataFrame(crows).to_csv(RESULTS_DIR / "asr_contrasts.csv", index=False)

    # ---- figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_df = by_group[by_group["group"] != "ALL"]
        labels = [GROUP_LABELS.get(g, g) for g in plot_df["group"]]
        x = np.arange(len(labels))
        engs = list(results)
        width = 0.8 / max(len(engs), 1)

        fig, ax = plt.subplots(figsize=(8, 4.6))
        colours = {"whisper": "#1f3864", "vosk": "#c0762a"}
        for i, eng in enumerate(engs):
            vals = plot_df[f"{eng}_wer"] * 100
            bars = ax.bar(
                x + i * width - 0.4 + width / 2, vals, width * 0.9,
                label=eng.capitalize(), color=colours.get(eng, None),
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Word error rate (%)")
        ax.set_title("Word error rate by speech condition", fontsize=11)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_wer_by_group.png", dpi=300)
        print(f"\nFigure -> {RESULTS_DIR / 'fig_wer_by_group.png'}")
    except ImportError:
        print("\nmatplotlib not installed; skipped the figure.")

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 62)
    print("WORD ERROR RATE BY GROUP")
    print("=" * 62)
    cols = ["group", "n_clips"] + [f"{e}_wer" for e in results]
    print(by_group[cols].to_string(index=False))
    if crows:
        print("\nCONTROLLED CONTRASTS (same sentences, one variable changed)")
        print(pd.DataFrame(crows).drop(columns=["note"]).to_string(index=False))
    print(f"\nWrote 3 CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()