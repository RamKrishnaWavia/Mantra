import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import os

st.title("🙏 Ganesha Ashtottara Chanting Generator")
st.markdown("Generate chanting audio with background music + subtitles")

# Upload or paste names
uploaded_file = st.file_uploader("Upload 108 Names (TXT file, one per line)", type=["txt"])
names_text = st.text_area("Or Paste 108 Names (one per line)")

if uploaded_file:
    names = uploaded_file.read().decode("utf-8").splitlines()
elif names_text:
    names = names_text.splitlines()
else:
    names = []

if st.button("Generate Chanting") and names:
    with st.spinner("Generating chanting... this may take a few minutes ⏳"):
        # Create base drone (tanpura-like)
        drone = Sine(220).to_audio_segment(duration=600000)  # 10 min
        drone = drone - 20  # lower volume

        final_audio = AudioSegment.silent(duration=0)
        srt_lines = []
        time_cursor = 0
        index = 1

        temp_dir = tempfile.mkdtemp()

        for name in names:
            if not name.strip():
                continue

            # Generate chanting using gTTS
            tts = gTTS(text=name, lang='sa')
            chant_path = os.path.join(temp_dir, f"chant_{index}.mp3")
            tts.save(chant_path)
            chant_audio = AudioSegment.from_mp3(chant_path)

            # Add pause
            chant_audio += AudioSegment.silent(duration=1200)
            final_audio += chant_audio

            # Subtitles
            start = time_cursor
            end = time_cursor + len(chant_audio)
            srt_lines.append(
                f"{index}\n"
                f"{start//3600000:02}:{(start//60000)%60:02}:{(start//1000)%60:02},{start%1000:03d} --> "
                f"{end//3600000:02}:{(end//60000)%60:02}:{(end//1000)%60:02},{end%1000:03d}\n"
                f"{name}\n\n"
            )

            time_cursor = end
            index += 1

        # Mix drone with chanting
        final_mix = drone.overlay(final_audio, loop=True)

        # Save outputs
        audio_path = os.path.join(temp_dir, "ganesha_ashtottara.mp3")
        srt_path = os.path.join(temp_dir, "ganesha_ashtottara.srt")

        final_mix.export(audio_path, format="mp3")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(srt_lines)

        # Streamlit downloads
        with open(audio_path, "rb") as f:
            st.download_button("⬇️ Download Chanting Audio", f, file_name="ganesha_ashtottara.mp3")

        with open(srt_path, "rb") as f:
            st.download_button("⬇️ Download Subtitles (SRT)", f, file_name="ganesha_ashtottara.srt")

        st.success("✅ Chanting + subtitles generated!")
