import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import ollama
import pyttsx3
from scipy.io.wavfile import write as write_wav
import io

# ========== Audio Recording Settings ==========
samplerate = 16000
duration = 5  # seconds per input

# ========== Load Whisper ==========
@st.cache_resource
def load_model():
    return whisper.load_model("base")
model = load_model()

# ========== Init TTS ==========
engine = pyttsx3.init()

def record_audio(duration=5, samplerate=16000):
    st.info("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio

def audio_to_wav_bytes(audio, samplerate):
    wav_buffer = io.BytesIO()
    write_wav(wav_buffer, samplerate, (audio * 32767).astype(np.int16))
    wav_buffer.seek(0)
    return wav_buffer

def transcribe_audio(audio, samplerate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name
        write_wav(temp_file, samplerate, (audio * 32767).astype(np.int16))
    result = model.transcribe(temp_file)
    os.remove(temp_file)
    return result["text"]

def chat_with_llama(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def tts_to_wav_bytes(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    with open(temp_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(temp_path)
    return wav_bytes

st.title("🗣️ Voice Assistant with Whisper & LLaMA")

if st.button("Record and Ask"):
    audio = record_audio(duration, samplerate)
    st.success("Recording complete!")
    st.audio(audio_to_wav_bytes(audio, samplerate), format="audio/wav")
    text = transcribe_audio(audio, samplerate)
    st.write("**You said:**", text)
    if text.strip().lower() in ["exit", "quit", "bye"]:
        st.write("👋 Goodbye!")
    else:
        reply = chat_with_llama(text)
        st.write("**LLaMA:**", reply)
        # Play in browser
        st.audio(tts_to_wav_bytes(reply), format="audio/wav")