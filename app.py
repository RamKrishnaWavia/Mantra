import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import io
import zipfile
from dataclasses import dataclass

st.set_page_config(page_title="🕉️ Devotional Mantra Generator", page_icon="🕉️", layout="centered")
st.title("🕉️ Devotional Mantra Generator")
st.write(
    "Create mantra audio with a soothing background (tanpura drone + ambient pad + soft flute). "
    "Works fully on Streamlit Cloud without ffmpeg/pydub."
)

# ---------------------------
# Utility audio synthesis funcs
# ---------------------------
SR = 44100  # sample rate


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    peak = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
    return (x / peak).astype(np.float32)


def fade_in(x: np.ndarray, ms: float) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    n = min(n, len(x))
    env = np.linspace(0.0, 1.0, n)
    y = x.copy()
    y[:n] *= env
    return y


def fade_out(x: np.ndarray, ms: float) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    n = min(n, len(x))
    env = np.linspace(1.0, 0.0, n)
    y = x.copy()
    y[len(x)-n:] *= env
    return y


def sine(freq: float, dur_s: float, vol: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * vol).astype(np.float32)


def soft_pad(dur_s: float, root_hz: float = 130.81) -> np.ndarray:
    """Create an airy pad from multiple close harmonics (C major-ish)."""
    freqs = [root_hz, root_hz * 5/4, root_hz * 3/2, root_hz * 2]
    waves = [sine(f, dur_s, 0.18) for f in freqs]
    pad = sum(waves)
    # gentle movement via very slow LFO
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    lfo = 0.95 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    pad *= lfo.astype(np.float32)
    return normalize(pad)


def tanpura_drone(dur_s: float, root_hz: float = 130.81) -> np.ndarray:
    """Simple tanpura-like base using low sine + soft fifth."""
    base = sine(root_hz, dur_s, 0.12)
    fifth = sine(root_hz * 3/2, dur_s, 0.06)
    octave = sine(root_hz * 2, dur_s, 0.04)
    d = base + fifth + octave
    return normalize(d)


def flute_phrase(dur_s: float, start_hz: float = 523.25) -> np.ndarray:
    """Soft flute-like envelope with slight vibrato and decay."""
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    vibrato = 0.004 * np.sin(2 * np.pi * 5.5 * t)
    freq = start_hz * (1 + vibrato)
    env = np.exp(-t * 1.6)
    tone = np.sin(2 * np.pi * freq * t) * env * 0.35
    return tone.astype(np.float32)


def bell_ping() -> np.ndarray:
    """Gentle temple bell ping (~1.2s)."""
    dur_s = 1.2
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    comps = 0.45 * np.sin(2*np.pi*880*t) * np.exp(-t*3.2) \
          + 0.28 * np.sin(2*np.pi*1320*t) * np.exp(-t*4.1) \
          + 0.18 * np.sin(2*np.pi*1760*t) * np.exp(-t*5.0)
    x = comps.astype(np.float32)
    x = fade_in(x, 15)
    x = fade_out(x, 250)
    return x


def chant_tone(dur_s: float, base_hz: float = 180.0) -> np.ndarray:
    """A vowel-like hum for one name; shaped with ADSR so it feels chanted."""
    t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
    # simple ADSR: 80ms attack, 200ms decay to 80%, sustain, 120ms release
    a = int(0.08 * SR); d = int(0.20 * SR); r = int(0.12 * SR)
    s = len(t) - (a + d + r)
    s = max(s, 0)
    attack = np.linspace(0, 1, a)
    decay = np.linspace(1, 0.8, d)
    sustain = np.full(s, 0.8)
    release = np.linspace(0.8, 0, r)
    env = np.concatenate([attack, decay, sustain, release])
    env = env[:len(t)]
    # slight pitch glide + tiny noise for breathiness
    glide = np.linspace(-0.02, 0.02, len(t))
    phase = 2 * np.pi * (base_hz * (1 + glide))
    tone = np.sin(np.cumsum(phase / SR)) * env * 0.38
    breath = (np.random.randn(len(t)) * 0.003).astype(np.float32)
    y = (tone + breath).astype(np.float32)
    return y


@dataclass
class RenderConfig:
    root_note_hz: float = 130.81  # C3 ~ 130.81
    name_duration_s: float = 6.0
    gap_between_names_s: float = 1.0
    include_bell_every: int = 8
    include_flute_every: int = 4


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")
root_note = st.sidebar.selectbox("Root (Shruti)", ["C (130.81 Hz)", "D (146.83 Hz)", "E (164.81 Hz)", "A (110.00 Hz)"], index=0)
root_map = {
    "A (110.00 Hz)": 110.00,
    "C (130.81 Hz)": 130.81,
    "D (146.83 Hz)": 146.83,
    "E (164.81 Hz)": 164.81,
}
root_hz = root_map[root_note]
name_duration = st.sidebar.slider("Per-name duration (seconds)", 4.0, 10.0, 6.0, 0.5)
gap_duration = st.sidebar.slider("Gap between names (seconds)", 0.0, 3.0, 1.0, 0.1)
flute_every = st.sidebar.number_input("Flute phrase every N names", min_value=0, max_value=20, value=4)
bell_every = st.sidebar.number_input("Bell ping every N names", min_value=0, max_value=20, value=8)

st.markdown("---")
st.subheader("Paste Mantra Lines (one per line)")
default_text = "\n".join([
    "Om Gajananaya Namaha",
    "Om Ganadhyakshaya Namaha",
    "Om Vighnarajaya Namaha",
])
text = st.text_area("Enter lines (e.g., 108 names)", value=default_text, height=220)
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

st.caption("Tip: You can paste the full 108 names here. The audio is chant-like (no external TTS), so it runs on Streamlit Cloud reliably.")

if st.button("Generate Audio + Subtitles"):
    if not lines:
        st.error("Please enter at least one line.")
        st.stop()

    cfg = RenderConfig(
        root_note_hz=root_hz,
        name_duration_s=float(name_duration),
        gap_between_names_s=float(gap_duration),
        include_bell_every=int(bell_every),
        include_flute_every=int(flute_every),
    )

    total_dur = (cfg.name_duration_s + cfg.gap_between_names_s) * len(lines) + 2.0
    total_samples = int(total_dur * SR) + 1

    # Pre-render long background layers
    pad_bg = soft_pad(total_dur, root_hz)
    drone_bg = tanpura_drone(total_dur, root_hz)
    bg = normalize(pad_bg * 0.7 + drone_bg * 0.5)

    # Initialize final track and subtitles
    final = np.zeros(total_samples, dtype=np.float32)
    cursor = 0
    srt_entries = []
    idx = 1

    # Precompute bell and flute samples
    bell = bell_ping()
    flute = flute_phrase(2.4, start_hz=523.25)  # C5-ish quick motif

    name_len = int(cfg.name_duration_s * SR)
    gap_len = int(cfg.gap_between_names_s * SR)

    for i, line in enumerate(lines):
        # Chant chunk
        chant = chant_tone(cfg.name_duration_s, base_hz=180.0)
        chant = fade_in(chant, 60)
        chant = fade_out(chant, 140)

        start = cursor
        end = start + len(chant)
        final[start:end] += chant

        # Optional flute
        if cfg.include_flute_every and ((i+1) % cfg.include_flute_every == 0):
            fstart = start
            fend = min(start + len(flute), total_samples)
            final[fstart:fend] += flute[:(fend - fstart)] * 0.7

        # Optional bell
        if cfg.include_bell_every and ((i+1) % cfg.include_bell_every == 0):
            bstart = start
            bend = min(start + len(bell), total_samples)
            final[bstart:bend] += bell[:(bend - bstart)] * 0.9

        # SRT timing (hh:mm:ss,ms)
        def fmt(ms_total: int) -> str:
            h = ms_total // (1000*60*60)
            ms_total %= (1000*60*60)
            m = ms_total // (1000*60)
            ms_total %= (1000*60)
            s = ms_total // 1000
            ms = ms_total % 1000
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        start_ms = int((start / SR) * 1000)
        end_ms = int(((end + gap_len) / SR) * 1000)
        srt_entries.append(f"{idx}\n{fmt(start_ms)} --> {fmt(end_ms)}\n{line}\n\n")
        idx += 1

        cursor = end + gap_len

    # Mix background with foreground chant & extras
    final_mix = normalize(final * 0.9 + bg[:len(final)] * 0.6)
    final_mix = fade_in(final_mix, 800)
    final_mix = fade_out(final_mix, 1500)

    # Convert to 16-bit PCM WAV bytes
    pcm = (final_mix * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, SR, pcm)
    buf.seek(0)

    st.success("✅ Generated devotional track!")
    st.audio(buf, format="audio/wav")

    # Downloads
    st.download_button(
        label="⬇️ Download Audio (WAV)",
        data=buf.getvalue(),
        file_name="mantra_devotional.wav",
        mime="audio/wav",
    )

    srt_text = "".join(srt_entries)
    st.download_button(
        label="⬇️ Download Subtitles (SRT)",
        data=srt_text.encode("utf-8"),
        file_name="mantra_subtitles.srt",
        mime="text/plain",
    )

st.markdown("""
---
**Notes**
- This app synthesizes a chant-like hum (not full TTS) to avoid ffmpeg/pydub issues on Streamlit Cloud.
- Paste your 108 names (Kannada or transliteration). Subtitles will use exactly your text.
- Use the sliders to change per-line duration and gap; timings reflect in the SRT.
""")
