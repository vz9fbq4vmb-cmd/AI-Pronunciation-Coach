import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
from gtts import gTTS
import difflib

st.set_page_config(page_title="AI Pronunciation Coach 3.0", layout="wide")

# Дизайнды жақсарту (CSS)
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .footer{ text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ AI-Powered Pronunciation Coach (US Accent)")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("📸 Visual Feedback (Articulation)")
    st.camera_input("Watch your lip movement")

with col2:
    target_text = st.text_input("Type a word or sentence to practice:", "Everything happens for a reason")
    
    # 🇺🇸 Native American Accent Generator
    if st.button("🔊 Listen to Native Speaker (US)"):
        try:
            # tld='us' нағыз американдық акцентті береді
            tts = gTTS(text=target_text, lang='en', tld='us', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            st.audio(audio_fp, format='audio/mp3')
        except Exception as e:
            st.error("Connection error. Please try again.")

    # 🎤 Recording Section
    st.subheader("Your Turn")
    audio_value = st.audio_input("Click to record your voice")

    if audio_value:
        recognizer = sr.Recognizer()
        try:
            audio_bytes = audio_value.read()
            audio_file = io.BytesIO(audio_bytes)
            
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                student_text = recognizer.recognize_google(audio_data, language="en-US")
                
                st.write(f"You said: **{student_text}**")

                # 📊 Accuracy Score Calculation
                # Сөздердің ұқсастығын пайызбен есептейміз
                similarity = difflib.SequenceMatcher(None, target_text.lower(), student_text.lower()).ratio()
                score = int(similarity * 100)

                # Түсті индикаторлар
                if score >= 90:
                    color = "#28a745" # Жасыл
                    status = "Excellent! Native level."
                    st.balloons()
                elif score >= 60:
                    color = "#ffc107" # Сары
                    status = "Good, but try to be clearer."
                else:
                    color = "#dc3545" # Қызыл
                    status = "Keep practicing! Listen to the sample again."

                st.markdown(f"""
                    <div style='padding:20px; border-radius:15px; background-color:{color}; color:white; text-align:center;'>
                        <h2 style='margin:0;'>Score: {score}%</h2>
                        <p style='margin:0; font-size:1.2em;'>{status}</p>
                    </div>
                """, unsafe_allow_html=True)

            # 📊 Spectrogram Analysis
            st.write("### 📊 Voice Spectrogram Analysis")
            y, sr_lib = librosa.load(io.BytesIO(audio_bytes), sr=None)
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr_lib, x_axis='time', y_axis='hz', ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.warning("Could not recognize audio. Speak louder and clear.")

st.divider()
st.caption("AI-Powered Pronunciation Coach | Developed for Science Project 2026")
