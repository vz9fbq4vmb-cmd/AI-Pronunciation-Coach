import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
from gtts import gTTS
import difflib
import pandas as pd
from datetime import datetime

# Беттің реттеулері
st.set_page_config(page_title="AI Pronunciation Coach Pro", layout="wide", initial_sidebar_state="expanded")

# --- СТИЛЬ (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 10px; height: 3em; transition: 0.3s; }
    .stButton>button:hover { background-color: #007bff; color: white; border: none; }
    .score-box { padding: 20px; border-radius: 15px; text-align: center; color: white; font-weight: bold; margin-bottom: 20px; }
    .sidebar-content { padding: 10px; border-radius: 10px; background-color: #ffffff; margin-bottom: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE (Мәліметтерді сақтау) ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- ДЕРЕКТЕР: ДЕҢГЕЙЛЕР ЖӘНЕ ТАПСЫРМАЛАР ---
levels = {
    "Easy (Сөздер)": ["Apple", "Family", "Student", "School", "Teacher"],
    "Medium (Тіркестер)": ["Good morning", "How are you?", "Artificial Intelligence", "Science Project"],
    "Hard (Скороговорки)": ["She sells seashells", "Red lory, yellow lory", "Peter Piper picked", "I scream for ice cream"]
}

# Күнделікті тапсырма (Күнге байланысты ауысады)
daily_challenges = ["Everything happens for a reason", "Practice makes perfect", "Knowledge is power"]
day_of_year = datetime.now().timetuple().tm_yday
daily_task = daily_challenges[day_of_year % len(daily_challenges)]

# --- SIDEBAR (Статистика және Деңгейлер) ---
with st.sidebar:
    st.title("📊 Dashboard")
    st.markdown(f"**🗓 Күнделікті тапсырма:** \n\n `{daily_task}`")
    
    selected_level = st.selectbox("🎯 Деңгейді таңдаңыз:", list(levels.keys()))
    suggested_word = st.selectbox("📖 Сөз таңдаңыз:", levels[selected_level])
    
    st.divider()
    st.subheader("📜 Жетістік тарихы")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, hide_index=True)
        if st.button("Тарихты тазалау"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("Әлі нәтиже жоқ.")

# --- НЕГІЗГІ БЕТ ---
st.title("🎙️ AI Pronunciation Coach Pro")
st.write("Ағылшын тіліндегі айтылымды жақсартуға арналған жасанды интеллект көмекшісі.")

col_main1, col_main2 = st.columns([1, 1])

with col_main1:
    st.subheader("📹 Артикуляция")
    st.camera_input("Watch your lip movement", key="coach_cam")
    
    st.info("💡 **ЖИ Кеңесі:** Дыбысты шығарғанда ерніңіз бен тіліңіздің қозғалысына назар аударыңыз. Айнадағыдай бақылау — ең жақсы әдіс.")

with col_main2:
    target_text = st.text_input("Жаттығатын сөзіңіз:", suggested_word)
    
    # 🇺🇸 Американдық акцент
    if st.button("🔊 Үлгіні тыңдау (American Accent)"):
        tts = gTTS(text=target_text, lang='en', tld='us')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        st.audio(fp, format='audio/mp3')

    st.divider()
    
    # 🎤 Жазу
    audio_value = st.audio_input("Дауысыңызды жазыңыз")

    if audio_value:
        recognizer = sr.Recognizer()
        try:
            audio_bytes = audio_value.read()
            audio_file = io.BytesIO(audio_bytes)
            
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                
                st.write(f"Сіз айттыңыз: **{student_text}**")

                # Ұпай есептеу
                similarity = difflib.SequenceMatcher(None, target_text.lower().strip(), student_text.lower().strip()).ratio()
                score = int(similarity * 100)

                # Түсті блоктар және Кеңестер
                if score >= 90:
                    bg_color, status, feedback = "#28a745", "Керемет!", "Сіздің айтылымыңыз нағыз нетив-спикер сияқты!"
                    st.balloons()
                elif score >= 60:
                    bg_color, status, feedback = "#ffc107", "Жақсы!", "Кейбір әріптерді анығырақ айтуға тырысыңыз."
                else:
                    bg_color, status, feedback = "#dc3545", "Талпын!", "Үлгіні қайта тыңдап, дауысты дыбыстарды созып айтыңыз."

                st.markdown(f"""
                    <div class="score-box" style="background-color: {bg_color};">
                        <h2 style="margin:0;">Ұпай: {score}%</h2>
                        <p style="margin:0;">{status}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write(f"💬 **AI Кеңесі:** {feedback}")

                # Тарихқа сақтау
                now = datetime.now().strftime("%H:%M")
                st.session_state.history.append({"Уақыт": now, "Сөз": target_text, "Ұпай": f"{score}%"})

            # 📊 Спектрограмма
            with st.expander("📊 Дыбыс толқынын көру (Spectrogram)"):
                y, sr_lib = librosa.load(io.BytesIO(audio_bytes), sr=None)
                fig, ax = plt.subplots(figsize=(10, 3))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr_lib, x_axis='time', y_axis='hz', ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error("Дыбыс танылмады. Сәл қаттырақ және анық сөйлеңіз.")

st.divider()
st.markdown("<p style='text-align: center; color: grey;'>AI Pronunciation Coach Pro | 2026 Science Project</p>", unsafe_allow_html=True)
