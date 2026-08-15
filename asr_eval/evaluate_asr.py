"""
evaluate_asr.py — Whisper vs Vosk word error rate on the project's own audio.

Produces every number and figure needed for Section 5.6 of the report.
Rerun it any time the audio, the Whisper model size, or the beam width
changes; nothing here is done by hand.

Audio format
------------
Record however is convenient — .m4a from a phone or Windows Voice Recorder,
.wav, .mp3, .ogg, .flac, .webm. The script converts every clip once to
16 kHz mono 16-bit PCM WAV in audio_16k/ and feeds *both* engines that same
converted file.

Converting once and sharing the result matters methodologically: Vosk
requires 16 kHz mono and Whisper resamples to 16 kHz mono internally anyway,
so giving both engines byte-identical input removes any question about
whether a format difference, rather than the models, produced the gap.
Use --whisper-source original to confirm the conversion changed nothing.

Usage
-----
    python evaluate_asr.py                          # both engines
    python evaluate_asr.py --engines whisper        # skip Vosk
    python evaluate_asr.py --whisper-size small.en
    python evaluate_asr.py --whisper-source original
    python evaluate_asr.py --reconvert              # rebuild the WAV cache

Expects
-------
    references.csv          clip_id, filename, group, speaker, condition, reference
    audio/<clip_id>.<ext>   any common audio format
    models/vosk-model-small-en-us-0.15/   (only if running Vosk)

Writes
------
    audio_16k/<clip_id>.wav         converted audio (cached between runs)
    results/asr_transcripts.csv     every clip, every engine, side by side
    results/asr_wer_by_group.csv    the table that goes in the report
    results/asr_contrasts.csv       the three controlled comparisons
    results/fig_wer_by_group.png    300 dpi, ready to paste into Chapter 5
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"
WAV_DIR = HERE / "audio_16k"
RESULTS_DIR = HERE / "results"
VOSK_MODEL_DIR = HERE / "models" / "vosk-model-small-en-us-0.15"

TARGET_RATE = 16000
AUDIO_EXTS = [".wav", ".m4a", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm", ".wma"]

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
# Finding the audio
# --------------------------------------------------------------------------
def resolve_clip(clip_id: str, filename: str):
    """
    Find the recording for a clip.

    Tries the exact filename from references.csv first, then any file in
    audio/ whose stem matches the clip_id. That means you can record .m4a
    and drop the files in without editing the CSV — the naming is what
    matters, not the extension.
    """
    exact = AUDIO_DIR / str(filename)
    if exact.exists():
        return exact
    for ext in AUDIO_EXTS:
        cand = AUDIO_DIR / f"{clip_id}{ext}"
        if cand.exists():
            return cand
    matches = sorted(
        p for p in AUDIO_DIR.glob(f"{clip_id}.*")
        if p.suffix.lower() in AUDIO_EXTS
    )
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# Conversion to 16 kHz mono 16-bit PCM WAV
# --------------------------------------------------------------------------
def is_conforming_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getnchannels() == 1
                and w.getframerate() == TARGET_RATE
                and w.getsampwidth() == 2
            )
    except Exception:
        return False


def convert_with_pyav(src: Path, dst: Path) -> None:
    """
    Decode and resample with PyAV.

    PyAV ships as a dependency of faster-whisper, so if Whisper runs on this
    machine, this works — no ffmpeg install, no PATH surgery on Windows.
    """
    import av
    import numpy as np

    with av.open(str(src)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_RATE)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))

    pcm = np.concatenate(chunks).astype("<i2") if chunks else np.zeros(0, "<i2")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())


def convert_with_ffmpeg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(TARGET_RATE),
         "-sample_fmt", "s16", "-loglevel", "error", str(dst)],
        check=True,
    )


def prepare_audio(src: Path, clip_id: str, force: bool = False) -> Path:
    """Return a path to 16 kHz mono WAV for this clip, converting if needed."""
    if is_conforming_wav(src):
        return src

    dst = WAV_DIR / f"{clip_id}.wav"
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst  # cached and still current

    try:
        convert_with_pyav(src, dst)
        return dst
    except ImportError:
        pass
    except Exception as exc:
        print(f"   PyAV could not read {src.name}: {exc}")

    if shutil.which("ffmpeg"):
        convert_with_ffmpeg(src, dst)
        return dst

    raise RuntimeError(
        f"Cannot convert {src.name}. Install PyAV with:  pip install av\n"
        "(it also arrives with faster-whisper), or install ffmpeg."
    )


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / max(w.getframerate(), 1)


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
    ap.add_argument("--whisper-source", default="converted",
                    choices=["converted", "original"],
                    help="which audio Whisper reads; 'converted' keeps both engines identical")
    ap.add_argument("--references", default="references.csv")
    ap.add_argument("--reconvert", action="store_true", help="rebuild the audio_16k cache")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    RESULTS_DIR.mkdir(exist_ok=True)

    ref_path = HERE / args.references
    if not ref_path.exists():
        sys.exit(f"ERROR: {ref_path} not found.")
    df = pd.read_csv(ref_path)

    # ---- locate and convert ---------------------------------------------
    print("Preparing audio ...")
    missing, failed = [], []
    originals, prepared = {}, {}

    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        src = resolve_clip(clip_id, row["filename"])
        if src is None:
            missing.append(clip_id)
            continue
        originals[clip_id] = src
        try:
            wav = prepare_audio(src, clip_id, force=args.reconvert)
        except Exception as exc:
            failed.append(f"{src.name}: {exc}")
            continue
        secs = wav_seconds(wav)
        if secs < 0.5:
            failed.append(f"{src.name}: only {secs:.2f}s of audio — silent or clipped?")
            continue
        prepared[clip_id] = wav
        tag = "" if src is wav else f"  <- {src.suffix}"
        print(f"  {clip_id:<14} {secs:5.1f}s{tag}")

    if missing:
        print(f"\nMISSING {len(missing)} clip(s) in {AUDIO_DIR}:")
        for m in missing:
            print(f"   - {m}.*  (any of {', '.join(AUDIO_EXTS[:5])} ...)")
    if failed:
        print(f"\nCOULD NOT USE ({len(failed)}):")
        for f in failed:
            print("   -", f)
    if not prepared:
        sys.exit("\nNo usable audio found. Nothing to do.")

    paths_conv = [(cid, prepared[cid]) for cid in df["clip_id"] if cid in prepared]
    paths_orig = [(cid, originals[cid]) for cid in df["clip_id"] if cid in prepared]
    print(f"\n{len(paths_conv)} clip(s) ready.\n")

    # ---- transcribe ------------------------------------------------------
    results = {}
    if "whisper" in engines:
        src_paths = paths_orig if args.whisper_source == "original" else paths_conv
        print(f"[whisper] reading the {args.whisper_source} audio")
        results["whisper"] = run_whisper(src_paths, args.whisper_size)
    if "vosk" in engines:
        if not VOSK_MODEL_DIR.exists():
            print(f"\nSKIPPING Vosk: model not found at {VOSK_MODEL_DIR}")
            print("Download vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models")
            print("and unzip it into models/.\n")
        else:
            results["vosk"] = run_vosk(paths_conv, VOSK_MODEL_DIR)

    if not results:
        sys.exit("No engine produced output.")

    # ---- per-clip table --------------------------------------------------
    df["source_file"] = df["clip_id"].map(lambda c: originals[c].name if c in originals else "")
    df["reference_norm"] = df["reference"].map(normalise)
    for eng, out in results.items():
        df[f"{eng}_raw"] = df["clip_id"].map(lambda c: out.get(c, {}).get("text", ""))
        df[f"{eng}_norm"] = df[f"{eng}_raw"].map(normalise)
        df[f"{eng}_seconds"] = df["clip_id"].map(
            lambda c: out.get(c, {}).get("seconds", float("nan"))
        )

    import jiwer

    scored = df[df["clip_id"].isin(prepared)].copy()
    for eng in results:
        scored[f"{eng}_wer"] = [
            round(jiwer.wer([r], [h]), 4) if r else float("nan")
            for r, h in zip(scored["reference_norm"], scored[f"{eng}_norm"])
        ]

    scored.to_csv(RESULTS_DIR / "asr_transcripts.csv", index=False)

    # ---- WER by group ----------------------------------------------------
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

    # ---- the three controlled contrasts ----------------------------------
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

    # ---- figure ----------------------------------------------------------
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

    # ---- summary ---------------------------------------------------------
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