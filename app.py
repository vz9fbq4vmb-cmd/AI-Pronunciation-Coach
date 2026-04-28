import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
import difflib
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components

# Беттің реттеулері
st.set_page_config(page_title="AI Pronunciation Coach Pro", layout="wide")

# --- JAVASCRIPT ТЫҢДАУ ФУНКЦИЯСЫ (iPhone үшін) ---
def text_to_speech_js(text):
    js_code = f"""
    <script>
    function speak() {{
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.lang = 'en-US';
        msg.rate = 0.8;
        window.speechSynthesis.speak(msg);
    }}
    </script>
    <button onclick="speak()" style="
        background-color: #007bff; 
        color: white; 
        border: none; 
        padding: 12px 20px; 
        border-radius: 10px; 
        cursor: pointer;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    ">
        🔊 Үлгіні тыңдау (Native US Accent)
    </button>
    """
    components.html(js_code, height=70)

# --- SESSION STATE (Мәліметтерді сақтау) ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- ДЕРЕКТЕР ---
levels = {
    "Easy (Сөздер)": ["Apple", "Family", "Student", "School", "Teacher"],
    "Medium (Тіркестер)": ["Good morning", "How are you?", "Artificial Intelligence", "Knowledge is power"],
    "Hard (Скороговорки)": ["She sells seashells", "Red lory, yellow lory", "Peter Piper picked"]
}

# --- SIDEBAR (Dashboard) ---
with st.sidebar:
    st.title("📊 Dashboard")
    selected_level = st.selectbox("🎯 Деңгейді таңдаңыз:", list(levels.keys()))
    suggested_word = st.selectbox("📖 Сөз таңдаңыз:", levels[selected_level])
    
    st.divider()
    st.subheader("📜 Жетістік тарихы")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, hide_index=True)
        if st.button("Тарихты тазалау"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("Әлі нәтиже жоқ.")

# --- НЕГІЗГІ БЕТ ---
st.title("🎙️ AI Pronunciation Coach Pro")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Артикуляция")
    st.camera_input("Watch your lips", key="coach_cam")
    st.info("💡 **ЖИ Кеңесі:** Сөзді айтқанда ерніңіздің пішініне мән беріңіз.")

with col2:
    target_text = st.text_input("Жаттығатын сөзіңіз:", suggested_word)
    
    st.write("🔊 **Дұрыс айтылуы:**")
    text_to_speech_js(target_text)

    st.divider()
    
    # 🎤 Жазу
    audio_value = st.audio_input("Дауысыңызды жазыңыз")

    if audio_value:
        recognizer = sr.Recognizer()
        try:
            audio_bytes = audio_value.read()
            audio_file = io.BytesIO(audio_bytes)
            
            with sr.AudioFile(audio_file) as source:
                audio_data_rec = recognizer.record(source)
                student_text = recognizer.recognize_google(audio_data_rec, language="en-US")
                
                st.write(f"Сіз айттыңыз: **{student_text}**")

                # Ұпай есептеу
                similarity = difflib.SequenceMatcher(None, target_text.lower().strip(), student_text.lower().strip()).ratio()
                score = int(similarity * 100)

                # Нәтиже дизайны
                if score >= 90:
                    bg_color, status = "#28a745", "Керемет! Нағыз нетив спикер!"
                    st.balloons()
                elif score >= 60:
                    bg_color, status = "#ffc107", "Жақсы, бірақ анығырақ айтуға болады."
                else:
                    bg_color, status = "#dc3545", "Талпын! Үлгіні тағы бір тыңдап көр."

                st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
                        <h2 style="margin:0;">Ұпай: {score}%</h2>
                        <p style="margin:0;">{status}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Тарихқа сақтау
                st.session_state.history.append({
                    "Уақыт": datetime.now().strftime("%H:%M"), 
                    "Сөз": target_text, 
                    "Ұпай": f"{score}%"
                })

            # 📊 СПЕКТРОГРАММА (ҚАЙТА ҚОСЫЛДЫ)
            st.write("### 📊 Дыбыс анализы (Spectrogram)")
            # librosa арқылы өңдеу
            y, sr_lib = librosa.load(io.BytesIO(audio_bytes), sr=None)
            fig, ax = plt.subplots(figsize=(10, 4))
            # Mel-spectrogram жасау
            S = librosa.feature.melspectrogram(y=y, sr=sr_lib, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_lib, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Дыбысты өңдеу мүмкін болмады: {e}")

st.divider()
st.caption("AI-Powered Pronunciation Coach | Science Project 2026")
