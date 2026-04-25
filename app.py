import streamlit as st
import speech_recognition as sr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="AI Pronunciation Coach 2.0", layout="wide")
st.title("🎙️ AI-Powered Pronunciation Coach 2.0")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("Артикуляцияны бақылау:")
    st.camera_input("Өз ерніңіздің қозғалысын тексеріңіз")

with col2:
    target_text = st.text_input("Қайталауға арналған сөзді жазыңыз:", "Hello world")
    
    # Микрофон қатесін болдырмау үшін стандартты Streamlit аудио жазбасын қолданамыз
    audio_value = st.audio_input("Дыбысты жазу үшін басыңыз")
    
    if audio_value:
        st.audio(audio_value)
        recognizer = sr.Recognizer()
        
        try:
            # Аудионы өңдеу
            with sr.AudioFile(audio_value) as source:
                audio_data = recognizer.record(source)
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                
                st.write(f"Сіз айттыңыз: **{student_text}**")
                
                # Салыстыру
                target_words = target_text.lower().split()
                student_words = student_text.lower().split()
                
                feedback_html = ""
                errors_found = False
                for word in target_words:
                    if word in student_words:
                        feedback_html += f"<span style='color:green; font-size:24px;'>{word} </span>"
                    else:
                        feedback_html += f"<span style='color:red; font-size:24px; text-decoration:underline;'>{word} </span>"
                        errors_found = True
                
                st.markdown(feedback_html, unsafe_allow_html=True)
                
                if not errors_found:
                    st.balloons()
                    st.success("Керемет! 100%")
                
                # Спектрограмма
                y, sr_lib = librosa.load(audio_value)
                fig, ax = plt.subplots()
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr_lib, x_axis='time', y_axis='hz', ax=ax)
                st.pyplot(fig)
                
        except Exception as e:
            st.error("Дыбысты тану мүмкін болмады.")

st.divider()
st.write("### 🦷 Articulatory Heatmap")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Vowel_chart.png/300px-Vowel_chart.png")
