import os
import io
import re
import time
import math
import tempfile
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import altair as alt
from faster_whisper import WhisperModel

# NOVO: biblioteke za vizuelizaciju zvuka
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# -----------------------------
# PODESIVI DEFAULTI
# -----------------------------
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")   # tiny/base/small/medium/large-v3
DEFAULT_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # "int8" (CPU), "float16" (GPU), "int8_float16" (GPU/CPU mix)


# -----------------------------
# POMOĆNE FUNKCIJE
# -----------------------------
def ensure_tmp_dir() -> Path:
    p = Path(tempfile.gettempdir()) / "st_transcribe"
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_uploaded_file(upl) -> Path:
    """
    Streamlit-ov UploadedFile (file_uploader i audio_input daju isti interfejs):
    .name, .type, .getbuffer() / .read()
    """
    tmp_dir = ensure_tmp_dir()
    suffix = Path(upl.name).suffix or ".wav"
    out = tmp_dir / f"upl_{int(time.time())}{suffix}"
    with open(out, "wb") as f:
        f.write(upl.getbuffer())
    return out

def fmt_ts(t: float) -> str:
    """ 12.345 -> 00:00:12,345 """
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments, dest: Path) -> Path:
    with open(dest, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{seg.text.strip()}\n\n")
    return dest

# --- NOVO: vizuelizacija zvuka ---
def load_audio_mono_16k(path: Path, target_sr: int = 16000):
    """
    Učitaj audio kao mono @ 16 kHz (librosa resampluje ako treba).
    Vraća: (y, sr)
    """
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y, sr

def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform"):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Vreme (s)")
    ax.set_ylabel("Amplituda")
    fig.tight_layout()
    return fig

def plot_spectrogram(y: np.ndarray, sr: int, title: str = "Spektrogram (dB)"):
    # STFT -> magnitude -> dB
    D = librosa.stft(y, n_fft=1024, hop_length=256, win_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# JEZIČKE STOP REČI (mini setovi)
# -----------------------------
STOPWORDS = {
    "sr": {
        "i","u","je","da","se","na","to","su","sam","smo","ste","nisam","nisi","nije","nismo","niste","nisu",
        "ali","ili","pa","kao","što","sto","što","ovo","ono","ova","ove","ovaj","taj","ta","to","tako","već","još",
        "od","do","za","po","pri","bez","sa","iz","o","kroz","preko","zbog","među","dok","jer","kada","kad",
        "koji","koja","koje","kojima","kojeg","koju","njih","njihov","njihova","njihovo","mi","ti","on","ona","oni","ono",
        "će","ću","ćemo","ćete","bi","bih","bismo","biste","bili","bila","bilo","biti","ne","ni","niti"
    },
    "hr": {"i","u","je","da","se","na","to","su","sam","smo","ste","ali","ili","pa","kao","što","ovo","ono","ova","ove","ovaj",
           "taj","tako","već","još","od","do","za","po","pri","bez","s","iz","o","kroz","preko","zbog","među","dok","jer","kad","kada",
           "koji","koja","koje","njih","mi","ti","on","ona","oni","ono","će","ću","ćemo","ćete","bi","bih","bismo","biste","bili","bila","bilo","biti","ne","ni"},
    "bs": {"i","u","je","da","se","na","to","su","sam","smo","ste","ali","ili","pa","kao","što","ovo","ono","ova","ove","ovaj",
           "taj","tako","već","još","od","do","za","po","pri","bez","sa","iz","o","kroz","preko","zbog","među","dok","jer","kad","kada",
           "koji","koja","koje","njih","mi","ti","on","ona","oni","ono","će","ću","ćemo","ćete","bi","bih","bismo","biste","bili","bila","bilo","biti","ne","ni"},
    "en": {"the","a","an","and","or","but","to","of","in","on","for","with","as","is","are","was","were","be","been","being","that","this","it","by","from","at","about","into","over","than","then","so","if","not"},
    "de": {"und","oder","aber","der","die","das","ein","eine","ist","im","in","zu","von","mit","für","auf","ан","als","auch","nicht","den"},
    "fr": {"et","ou","mais","le","la","les","un","une","est","dans","de","du","des","en","pour","avec","sur","par","au","aux","pas","que","qui"},
    "es": {"y","o","pero","el","la","los","las","un","una","es","en","de","del","para","con","por","sobre","al","no","que"},
    "it": {"e","o","ma","il","la","i","le","un","una","è","in","di","del","della","per","con","su","da","al","non","che"},
    "ru": {"и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у"},
    "mk": {"и","во","не","дека","тој","на","сум","сме","сте","ќе","да","се","тоа","ова","оној","како","или","но","со","од","до","за"},
    "sq": {"dhe","ose","por","një","një","në","me","për","si","është","janë","jo","që","kjo","ai","ajo","ne","ju"}
}

def tokenize(text: str):
    # Unicode reč: slova i brojevi; koristi lower i uklanja naglašene znakove samo ako treba
    # zadržavamo dijakritike (č, ć, š, ž, đ)
    return re.findall(r"\b[\w\-’']+\b", text.lower(), flags=re.UNICODE)

def split_sentences(text: str):
    parts = re.split(r"[\.!\?]+", text)
    return [p.strip() for p in parts if p.strip()]

def filter_tokens(tokens, lang_code="sr", use_stopwords=True, min_len=2):
    sw = STOPWORDS.get(lang_code, set()) if use_stopwords else set()
    cleaned = []
    for t in tokens:
        tt = t.strip("-'’_")
        if len(tt) < min_len:
            continue
        if tt.isnumeric():
            continue
        if tt in sw:
            continue
        cleaned.append(tt)
    return cleaned

def top_n(counter: Counter, n=10):
    return counter.most_common(n)

def as_csv(rows, header=("token","count")) -> bytes:
    out = io.StringIO()
    out.write(",".join(header) + "\n")
    for a, b in rows:
        out.write(f"{a},{b}\n")
    return out.getvalue().encode("utf-8")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Transkripcija (Streamlit + faster-whisper)", layout="centered")
st.title("🎙️ Transkripcija govora – Streamlit + faster-whisper")

with st.sidebar:
    st.subheader("⚙️ Podešavanja")
    model_size = st.selectbox(
        "Model",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=["tiny","base","small","medium","large-v3"].index(DEFAULT_MODEL) if DEFAULT_MODEL in ["tiny","base","small","medium","large-v3"] else 2
    )
    compute_type = st.selectbox(
        "Compute",
        ["int8", "int8_float16", "float16", "float32"],
        index=["int8","int8_float16","float16","float32"].index(DEFAULT_COMPUTE) if DEFAULT_COMPUTE in ["int8","int8_float16","float16","float32"] else 0,
        help="int8: najbrže na CPU; float16: brzo na GPU; int8_float16: kompromis"
    )
    lang = st.selectbox(
        "Jezik (auto detekcija ili fiksno)",
        ["auto","sr","en","de","fr","es","it","ru","hr","bs","mk","sq"],
        index=0
    )
    task = st.radio("Režim", ["transcribe","translate"], index=0)
    vad = st.checkbox("Uključi VAD (stabilnije segmentisanje)", value=True,
                      help="Filtrira tišine i pozadinu (Silero VAD kroz faster-whisper)")

    st.markdown("---")
    st.subheader("📊 Statistika (važi za transcribe)")
    top_n_words = st.slider("TOP N reči", min_value=5, max_value=30, value=10, step=1)
    min_len = st.slider("Minimalna dužina reči", min_value=2, max_value=6, value=2, step=1)
    use_sw = st.checkbox("Ukloni stop reči (preporučeno)", value=True)

    st.markdown("---")
    # NOVO: kontrola vizuelizacije
    show_viz = st.checkbox("Prikaži vizuelizaciju zvuka (waveform & spektrogram)", value=True)

st.markdown("**Opcija 1:** Snimi glas (mikrofon)  •  **Opcija 2:** Upload postojećeg fajla (mp3/wav/m4a/flac...)")

col1, col2 = st.columns(2, gap="large")
with col1:
    rec = st.audio_input("🎤 Snimi mikrofonom (WAV)")
with col2:
    upl = st.file_uploader("⬆️ Upload audio fajla", type=None)

go = st.button("▶️ Transkribuj", use_container_width=True)

# Lazy-load model (prvi put duže traje, posle je instant)
@st.cache_resource(show_spinner=True)
def load_model_cached(name: str, compute: str):
    return WhisperModel(name, compute_type=compute)

if go:
    if not rec and not upl:
        st.warning("Prvo snimi ili uploaduj fajl.")
        st.stop()

    # Sačuvaj ulaz u temp fajl
    try:
        st.info("Čuvam ulazni audio...")
        audio_path = save_uploaded_file(rec or upl)
    except Exception as e:
        st.error(f"Greška pri čuvanju ulaza: {e}")
        st.stop()

    # Učitavanje modela (cache_resource sprema instancu za ponovna pokretanja)
    with st.spinner(f"Učitavam model: {model_size} / {compute_type} ..."):
        model = load_model_cached(model_size, compute_type)

    st.success("Model spreman ✅")
    st.audio(str(audio_path))

    # Transkripcija
    with st.spinner("Transkribujem..."):
        segments, info = model.transcribe(
            str(audio_path),
            vad_filter=vad,
            vad_parameters=dict(min_silence_duration_ms=500),
            language=None if lang == "auto" else lang,
            task=task
        )

        # Skupi tekst i paralelno snimi segmente za .srt
        collected = []
        total_speech_time = 0.0
        for s in segments:
            collected.append(s)
            # zbir realnog govora preko segmenata
            if getattr(s, "start", None) is not None and getattr(s, "end", None) is not None:
                total_speech_time += max(0.0, (s.end - s.start))

        full_text = "".join(s.text for s in collected).strip()

    st.subheader("📝 Transkript")
    st.text_area("Rezultat", full_text, height=280)

    # Generiši .srt i .txt za preuzimanje
    tmp_dir = ensure_tmp_dir()
    srt_path = tmp_dir / f"transkript_{int(time.time())}.srt"
    txt_path = tmp_dir / f"transkript_{int(time.time())}.txt"
    try:
        segments_to_srt(collected, srt_path)
        txt_path.write_text(full_text, encoding="utf-8")
    except Exception as e:
        st.error(f"Greška pri kreiranju izlaznih fajlova: {e}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Preuzmi .srt",
                data=srt_path.read_bytes(),
                file_name=srt_path.name,
                mime="application/x-subrip",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "⬇️ Preuzmi .txt",
                data=txt_path.read_bytes(),
                file_name=txt_path.name,
                mime="text/plain",
                use_container_width=True
            )

    # Kratka metainfo
    st.caption(
        f"Model: **{model_size}**, compute: **{compute_type}**, "
        f"detektovan jezik: **{getattr(info, 'language', '?')}**, "
        f"procenjena tačnost: **{getattr(info, 'language_probability', 0):.2f}**"
    )

    # -----------------------------
    # NOVO: VIZUELIZACIJA ZVUKA
    # -----------------------------
    if show_viz and task == "transcribe":
        st.markdown("## 🎛️ Vizuelizacija zvuka")
        try:
            with st.spinner("Generišem waveform i spektrogram..."):
                y, sr = load_audio_mono_16k(audio_path)

                fig_wf = plot_waveform(y, sr, title="Waveform (mono, 16 kHz)")
                wf_png = fig_to_png_bytes(fig_wf)
                st.pyplot(fig_wf)
                st.download_button(
                    "⬇️ Preuzmi waveform (PNG)",
                    data=wf_png,
                    file_name="waveform.png",
                    mime="image/png",
                    use_container_width=True
                )

                fig_sp = plot_spectrogram(y, sr, title="Spektrogram (STFT, dB)")
                sp_png = fig_to_png_bytes(fig_sp)
                st.pyplot(fig_sp)
                st.download_button(
                    "⬇️ Preuzmi spektrogram (PNG)",
                    data=sp_png,
                    file_name="spectrogram.png",
                    mime="image/png",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Nije uspelo generisanje vizuelizacije: {e}")

    # -----------------------------
    # STATISTIKA (samo za transcribe)
    # -----------------------------
    if task == "transcribe":
        st.markdown("## 📊 Statistika transkripta")

        # osnovne metrike iz teksta
        sent_list = split_sentences(full_text)
        tokens_all = tokenize(full_text)

        lang_code = getattr(info, "language", None) or "sr"
        tokens_clean = filter_tokens(tokens_all, lang_code=lang_code, use_stopwords=use_sw, min_len=min_len)

        total_words = len(tokens_all)
        total_words_clean = len(tokens_clean)
        unique_words = len(set(tokens_clean))
        ttr = (unique_words / total_words_clean) if total_words_clean > 0 else 0.0
        avg_word_len = (sum(len(t) for t in tokens_clean) / total_words_clean) if total_words_clean > 0 else 0.0
        avg_sent_len = (total_words / max(1, len(sent_list)))

        # info o trajanju
        duration_all = float(getattr(info, "duration", 0.0) or 0.0)
        speech_time = total_speech_time if total_speech_time > 0 else duration_all
        wpm = (total_words / (duration_all / 60.0)) if duration_all > 0 else 0.0
        wpm_speech = (total_words / (speech_time / 60.0)) if speech_time > 0 else 0.0
        seg_count = len(collected)
        avg_seg_len = (speech_time / seg_count) if seg_count > 0 else 0.0

        # prikaz metrika
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Broj reči (ukupno)", f"{total_words}")
        m2.metric("Jedinstvene reči", f"{unique_words}")
        m3.metric("TTR (raznovrsnost)", f"{ttr:.2f}")
        m4.metric("Prosečna dužina reči", f"{avg_word_len:.2f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Broj rečenica", f"{len(sent_list)}")
        m6.metric("Prosečno reči / rečenici", f"{avg_sent_len:.1f}")
        m7.metric("Trajanje (ukupno)", fmt_ts(duration_all) if duration_all else "n/a")
        m8.metric("Reči / minut", f"{wpm:.1f}")

        m9, m10, m11, m12 = st.columns(4)
        m9.metric("Govor (suma segmenata)", fmt_ts(speech_time) if speech_time else "n/a")
        m10.metric("Reči / minut (samo govor)", f"{wpm_speech:.1f}")
        m11.metric("Broj segmenata", f"{seg_count}")
        m12.metric("Prosečna dužina segmenta", f"{avg_seg_len:.2f}s")

        # Top N reči
        cnt_words = Counter(tokens_clean)
        top_words = top_n(cnt_words, n=top_n_words)
        df_top = pd.DataFrame(top_words, columns=["token", "count"])

        st.subheader(f"TOP {top_n_words} najčešćih reči")
        chart = (
            alt.Chart(df_top)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Pojavljivanja"),
                y=alt.Y("token:N", sort='-x', title="Reč")
            )
            .properties(height=25 * len(df_top), width=650)
        )
        st.altair_chart(chart, use_container_width=True)
        st.download_button(
            "⬇️ Preuzmi TOP reči (CSV)",
            data=as_csv(top_words, header=("token","count")),
            file_name="top_words.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Top N bigrama (dvočlane fraze)
        bigrams = list(zip(tokens_clean, tokens_clean[1:]))
        cnt_bi = Counter([" ".join(bg) for bg in bigrams])
        top_bi = top_n(cnt_bi, n=top_n_words)
        df_bi = pd.DataFrame(top_bi, columns=["bigram", "count"])

        st.subheader(f"TOP {top_n_words} bigrama (dvočlanih fraza)")
        chart_bi = (
            alt.Chart(df_bi)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Pojavljivanja"),
                y=alt.Y("bigram:N", sort='-x', title="Bigram")
            )
            .properties(height=25 * len(df_bi), width=650)
        )
        st.altair_chart(chart_bi, use_container_width=True)
        st.download_button(
            "⬇️ Preuzmi TOP bigrame (CSV)",
            data=as_csv(top_bi, header=("bigram","count")),
            file_name="top_bigrams.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("🔎 Prikaz očišćenih tokena (debug)"):
            st.write(tokens_clean[:200])

    else:
        st.info("Statistika je dostupna kada je **Režim = transcribe**.")

    st.success("Gotovo ✅")


# Footer info
st.markdown("---")
st.caption("Tip: Ako je sporo na CPU, probaj **Model: base/tiny** ili ostavi **int8**. "
           "Za NVIDIA GPU stavi **compute=float16** i instaliraj odgovarajući PyTorch/CUDA (nije obavezno za faster-whisper).")
