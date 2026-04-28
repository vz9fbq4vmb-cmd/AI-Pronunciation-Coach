import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
from gtts import gTTS

st.set_page_config(page_title="AI Pronunciation Coach 2.0", layout="wide")
st.title("🎙️ AI-Powered Pronunciation Coach 2.0")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("Артикуляцияны бақылау:")
    st.camera_input("Өз ерніңізді тексеріңіз")

with col2:
    target_text = st.text_input("Қайталауға арналған сөзді жазыңыз:", "Hello")
    
    # 🔊 Дұрыс айтылуын тыңдау батырмасы
    if st.button("🔊 Үлгіні тыңдау (Listen)"):
        tts = gTTS(text=target_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format='audio/mp3')

    # 🎤 Дыбыс жазу
    audio_value = st.audio_input("Сөзді айту үшін басыңыз")

    if audio_value:
        # Дыбысты тану (Speech-to-Text)
        recognizer = sr.Recognizer()
        try:
            audio_bytes = audio_value.read()
            audio_file = io.BytesIO(audio_bytes)
            
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                
                st.write(f"Сіз айттыңыз: **{student_text}**")

                # Түспен көрсету
                if student_text.lower().strip() == target_text.lower().strip():
                    st.markdown(f"<h2 style='color: #28a745;'>✅ Керемет! 100%</h2>", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"<h2 style='color: #dc3545;'>❌ Қате айтылды</h2>", unsafe_allow_html=True)

            # 📊 Спектрограмма (Түзетілген нұсқа)
            st.write("### 📊 Дыбыс картасы (Spectrogram)")
            
            # librosa арқылы оқу
            y, sr_lib = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr_lib, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_lib, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Анализ жасау мүмкін болмады: {e}")

st.divider()
st.write("### 🦷 Articulatory Heatmap")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Vowel_chart.png/300px-Vowel_chart.png")
