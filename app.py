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

st.set_page_config(page_title="AI Pronunciation Coach Pro", layout="wide")

# --- JAVASCRIPT ТЫҢДАУ ФУНКЦИЯСЫ (iPhone үшін ең сенімді жол) ---
def text_to_speech_js(text):
    js_code = f"""
    <script>
    function speak() {{
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.lang = 'en-US';
        msg.rate = 0.9;
        window.speechSynthesis.speak(msg);
    }}
    </script>
    <button onclick="speak()" style="
        background-color: #007bff; 
        color: white; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 10px; 
        cursor: pointer;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
    ">
        🔊 Үлгіні тыңдау (Native US Accent)
    </button>
    """
    components.html(js_code, height=60)

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- ДЕРЕКТЕР ---
levels = {
    "Easy (Сөздер)": ["Apple", "Family", "Student", "School", "Teacher"],
    "Medium (Тіркестер)": ["Good morning", "How are you?", "Artificial Intelligence"],
    "Hard (Скороговорки)": ["She sells seashells", "Red lory, yellow lory"]
}

# --- SIDEBAR ---
with st.sidebar:
    st.title("📊 Dashboard")
    selected_level = st.selectbox("🎯 Деңгейді таңдаңыз:", list(levels.keys()))
    suggested_word = st.selectbox("📖 Сөз таңдаңыз:", levels[selected_level])
    
    st.divider()
    st.subheader("📜 Жетістік тарихы")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)

# --- НЕГІЗГІ БЕТ ---
st.title("🎙️ AI Pronunciation Coach Pro")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Артикуляция")
    st.camera_input("Watch your lips", key="coach_cam")

with col2:
    target_text = st.text_input("Жаттығатын сөзіңіз:", suggested_word)
    
    # 🇺🇸 ЖАҢА ТЫҢДАУ БАТЫРМАСЫ (JavaScript)
    st.write("🔊 **Үлгіні тыңдау:**")
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

                similarity = difflib.SequenceMatcher(None, target_text.lower().strip(), student_text.lower().strip()).ratio()
                score = int(similarity * 100)

                if score >= 90:
                    bg_color, status = "#28a745", "Керемет!"
                    st.balloons()
                elif score >= 60:
                    bg_color, status = "#ffc107", "Жақсы!"
                else:
                    bg_color, status = "#dc3545", "Талпын!"

                st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
                        <h2 style="margin:0;">Ұпай: {score}%</h2>
                        <p style="margin:0;">{status}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.session_state.history.append({
                    "Уақыт": datetime.now().strftime("%H:%M"), 
                    "Сөз": target_text, 
                    "Ұпай": f"{score}%"
                })

        except Exception as e:
            st.error("Дыбыс танылмады. Сәл қаттырақ сөйлеңіз.")
