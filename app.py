import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import ollama
import base64

samplerate = 16000
duration = 3  # Shorter duration for faster response

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")  # Use "tiny" for fastest transcription

model = load_model()

def record_audio(duration=3, samplerate=16000):
    st.info("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio

def transcribe_audio(audio, samplerate=16000):
    import tempfile
    from scipy.io.wavfile import write as write_wav
    import os
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
    import pyttsx3
    import tempfile
    import os
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    with open(temp_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(temp_path)
    return wav_bytes

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'''
    <audio id="llama-audio" autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    '''
    st.markdown(md, unsafe_allow_html=True)

st.title("⚡ Fast Voice Assistant")

# Session state to track if a reply is being spoken
if 'speaking' not in st.session_state:
    st.session_state['speaking'] = False

if st.button("Record and Ask"):
    audio = record_audio(duration, samplerate)
    text = transcribe_audio(audio, samplerate)
    st.write("**You said:**", text)
    if text.strip().lower() in ["exit", "quit", "bye"]:
        st.write("👋 Goodbye!")
        st.session_state['speaking'] = False
    else:
        reply = chat_with_llama(text)
        st.write("**LLaMA:**", reply)
        autoplay_audio(tts_to_wav_bytes(reply))
        st.session_state['speaking'] = True

# Show Stop Speaking button only if currently speaking
if st.session_state.get('speaking', False):
    if st.button("Stop Speaking"):
        # Inject JS to stop and remove the audio element
        stop_js = """
        <script>
        var audio = document.getElementById('llama-audio');
        if (audio) {
            audio.pause();
            audio.currentTime = 0;
            audio.remove();
        }
        </script>
        """
        st.markdown(stop_js, unsafe_allow_html=True)
        st.session_state['speaking'] = False