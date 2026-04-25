import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Pronunciation Coach 2.0", layout="wide")
st.title("🎙️ AI-Powered Pronunciation Coach 2.0")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("Артикуляцияны бақылау:")
    st.camera_input("Өз ерніңізді тексеріңіз")

with col2:
    target_text = st.text_input("Қайталауға арналған сөз:", "Hello world")
    
    # Браузердің дыбыс жазу интерфейсі
    audio_value = st.audio_input("Дыбысты жазу үшін басыңыз")

    if audio_value:
        st.audio(audio_value)
        st.success("Дыбыс қабылданды! Талдау жасалуда...")
        
        try:
            # Дыбысты өңдеу (Спектрограмма)
            y, sr = librosa.load(audio_value)
            
            st.write("### 📊 Дыбыс картасы (Spectrogram)")
            fig, ax = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
            st.pyplot(fig)
            
            st.info("Кеңес: Спектрограммадағы толқындарға қарап, дыбыстың анықтығын тексеріңіз.")
            
        except Exception as e:
            st.error(f"Қате орын алды: {e}")

st.divider()
st.write("### 🦷 Articulatory Heatmap")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Vowel_chart.png/300px-Vowel_chart.png", caption="Тілдің позициясы")
