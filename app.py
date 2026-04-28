import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import io

st.set_page_config(page_title="AI Pronunciation Coach 2.0", layout="wide")
st.title("🎙️ AI-Powered Pronunciation Coach 2.0")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("Артикуляцияны бақылау:")
    st.camera_input("Өз ерніңізді тексеріңіз")

with col2:
    target_text = st.text_input("Қайталауға арналған сөзді жазыңыз:", "Hello")
    
    # Дыбыс жазу интерфейсі
    audio_value = st.audio_input("Сөзді айту үшін басыңыз")

    if audio_value:
        st.audio(audio_value)
        
        # Дыбысты мәтінге айналдыру (Speech-to-Text)
        recognizer = sr.Recognizer()
        try:
            # AudioValue-ны файл ретінде оқу
            audio_file = io.BytesIO(audio_value.read())
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                # Google API арқылы тану
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                
                st.write(f"Сіз айттыңыз: **{student_text}**")

                # Салыстыру және түспен көрсету
                if student_text.lower().strip() == target_text.lower().strip():
                    st.markdown(f"<h2 style='color: #28a745; text-align: center;'>✅ Керемет! Дұрыс: {target_text}</h2>", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"<h2 style='color: #dc3545; text-align: center;'>❌ Қате! Қайталап көріңіз.</h2>", unsafe_allow_html=True)
                    st.write(f"Күтілген сөз: **{target_text}**, бірақ жүйе **{student_text}** деп естіді.")

            # Спектрограмма
            st.write("### 📊 Дыбыс анализы")
            y, sr_lib = librosa.load(audio_file)
            fig, ax = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr_lib, x_axis='time', y_axis='hz', ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.warning("Дыбысты тану мүмкін болмады. Анығырақ айтып көріңіз.")

st.divider()
st.write("### 🦷 Articulatory Heatmap")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Vowel_chart.png/300px-Vowel_chart.png")
