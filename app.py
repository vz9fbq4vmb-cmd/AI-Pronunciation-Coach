import streamlit as st
import speech_recognition as sr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from gtts import gTTS
import io

# Беттің дизайны
st.set_page_config(page_title="AI Pronunciation Coach 2.0", layout="wide")
st.title("🎙️ AI-Powered Pronunciation Coach 2.0")
st.subheader("Visualizing Phonetics & Haptic Feedback Simulation")

# Сол жақ баған: Камера және Нұсқаулық
col1, col2 = st.columns([1, 1])

with col1:
    st.info("Артикуляцияны бақылау үшін камераны қолданыңыз:")
    img_file = st.camera_input("Өз ерніңіздің қозғалысын тексеріңіз")

# Оң жақ баған: Дыбыс талдау
with col2:
    target_text = st.text_input("Қайталауға арналған сөзді жазыңыз:", "Hello world")
    
    if st.button("🎤 Дыбысты жазуды бастау"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Дауыстап айтыңыз...")
            audio_data = recognizer.listen(source)
            st.success("Дыбыс қабылданды!")

            try:
                # Дыбысты мәтінге айналдыру
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                st.write(f"Сіз айттыңыз: **{student_text}**")

                # Салыстыру алгоритмі
                target_words = target_text.lower().split()
                student_words = student_text.lower().split()

                st.write("### Талдау нәтижесі:")
                feedback_html = ""
                errors_found = False

                for word in target_words:
                    if word in student_words:
                        feedback_html += f"<span style='color:green; font-size:24px;'>{word} </span>"
                    else:
                        feedback_html += f"<span style='color:red; font-size:24px; text-decoration:underline;'>{word} </span>"
                        errors_found = True

                st.markdown(feedback_html, unsafe_allow_html=True)

                if errors_found:
                    st.error("⚠️ Қате анықталды! Діріл сигналы жіберілді (Haptic Feedback).")
                    st.write("Кеңес: Спектрограммаға қарап, тілдің орнын түзетіңіз.")
                else:
                    st.balloons()
                    st.success("Керемет! Pronunciation 100%")

                # Спектрограмма құру
                st.write("### 📊 Спектрограмма (Дыбыс картасы)")
                wav_data = io.BytesIO(audio_data.get_wav_data())
                y, sr_lib = librosa.load(wav_data)
                fig, ax = plt.subplots()
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr_lib, x_axis='time', y_axis='hz', ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error("Дыбысты тану мүмкін болмады. Қайта көріңіз.")

# Төменгі бөлім: Виртуалды 3D модель (сипаттама)
st.divider()
st.write("### 🦷 Articulatory Heatmap (Тіл позициясы)")
st.write("Қызыл зона - жоғары қысым, Көк зона - ауа ағыны.")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Vowel_chart.png/300px-Vowel_chart.png", caption="Тілдің орналасу картасы")
