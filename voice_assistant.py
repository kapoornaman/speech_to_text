import whisper
import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
import ollama
import pyttsx3
conversation_history = []


# ========== Audio Recording Settings ==========
samplerate = 16000
duration = 5  # seconds per input
q = queue.Queue()

# ========== Load Whisper ==========
model = whisper.load_model("base")  # or "small", "medium", "large"

# ========== Init TTS ==========
engine = pyttsx3.init()

# ========== Audio Callback ==========
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ========== Record & Transcribe ==========
def record_and_transcribe():
    print("🎙️ Listening... Speak now!")
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        audio = np.empty((0, 1), dtype=np.float32)
        for _ in range(0, int(samplerate * duration / 1024)):
            audio_chunk = q.get()
            audio = np.append(audio, audio_chunk, axis=0)

    print("🔁 Transcribing...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name
        from scipy.io.wavfile import write as write_wav
        write_wav(temp_file, samplerate, audio)


    result = model.transcribe(temp_file)
    os.remove(temp_file)
    return result["text"]

# ========== Query LLaMA ==========
def chat_with_llama(prompt):
    print("🤖 Thinking...")
    
    # Append user input to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    response = ollama.chat(
        model="llama3",
        messages=conversation_history
    )

    reply = response["message"]["content"]
    
    # Append bot's reply to history
    conversation_history.append({"role": "assistant", "content": reply})
    
    return reply


# ========== Speak ==========
def speak(text):
    print("🗣️ Responding...")
    engine.say(text)
    engine.runAndWait()

# ========== Main Loop ==========
if __name__ == "__main__":
    while True:
        try:
            user_input = record_and_transcribe()
            print("You said:", user_input)

            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break
            if user_input.strip().lower() in ["start over", "reset chat"]:
                conversation_history.clear()
                print("🔄 Chat history cleared.")
                continue


            reply = chat_with_llama(user_input)
            print("LLaMA:", reply)

            speak(reply)

        except KeyboardInterrupt:
            print("🚪 Exiting...")
            break
