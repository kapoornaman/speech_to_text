import whisper
import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
import ollama
import pyttsx3
import json
import time
from scipy.io.wavfile import write as write_wav

# ========== Audio Recording Settings ==========
samplerate = 16000
channels = 1
q = queue.Queue()
silence_threshold = 0.01  # Adjust this value based on your microphone
silence_duration = 1.0  # seconds of silence to stop recording
min_recording_duration = 0.5  # minimum recording duration in seconds

# ========== Load Whisper ==========
model = whisper.load_model("base")

# ========== Init TTS ==========
engine = pyttsx3.init()

# ========== Memory ==========
conversation_history = [
    {"role": "system", "content": (
        "You are a friendly and knowledgeable car dealer. "
        "You only talk about cars, car models, features, and related topics. "
        "For anything else, you should say 'I'm sorry, I can only talk about cars.' "
        "Start every new conversation by asking the user's name. "
        "Remember the user's name and their car interests for future conversations. "
        "If a returning user provides their name, greet them and ask about their previous interests. "
        "Pay attention to any car-related interests the user mentions during the conversation "
        "and remember these for future interactions. "
        "If you detect a new car interest, acknowledge it and add it to the user's profile."
    )}
]

USER_DATA_FILE = "user_profiles.json"

def load_user_profiles():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_profiles(profiles):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(profiles, f)

def update_user_interests(user_name, new_interests):
    if user_name not in user_profiles:
        user_profiles[user_name] = {"interests": []}
    
    current_interests = set(user_profiles[user_name]["interests"])
    if isinstance(new_interests, str):
        new_interests = [new_interests]
    
    for interest in new_interests:
        if interest.lower() not in [i.lower() for i in current_interests]:
            current_interests.add(interest)
    
    user_profiles[user_name]["interests"] = list(current_interests)
    save_user_profiles(user_profiles)

# ========== Audio Callback ==========
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ========== Record & Transcribe ==========
def record_and_transcribe():
    print("🎙️ Listening... Speak now!")
    print("(Press Ctrl+C to stop recording)")
    
    audio = np.empty((0, channels), dtype=np.float32)
    silence_counter = 0
    last_sound_time = time.time()
    recording_started = False
    
    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            while True:
                audio_chunk = q.get()
                current_time = time.time()
                
                # Check if there's sound in the chunk
                if np.abs(audio_chunk).mean() > silence_threshold:
                    if not recording_started:
                        recording_started = True
                        print("Recording started...")
                    last_sound_time = current_time
                    silence_counter = 0
                else:
                    silence_counter += len(audio_chunk) / samplerate
                
                # Add chunk to audio if we're recording
                if recording_started:
                    audio = np.append(audio, audio_chunk, axis=0)
                
                # Stop if we've had enough silence after starting to record
                if recording_started and silence_counter >= silence_duration:
                    if current_time - last_sound_time >= min_recording_duration:
                        break
                    else:
                        silence_counter = 0
                
    except KeyboardInterrupt:
        print("\nStopping recording...")
    
    if len(audio) == 0:
        print("No audio recorded!")
        return ""
    
    print("🔁 Transcribing...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name
        write_wav(temp_file, samplerate, audio)

    result = model.transcribe(temp_file)
    os.remove(temp_file)
    return result["text"]

# ========== Query LLaMA ==========
def chat_with_llama(prompt, user_name):
    print("🤖 Thinking...")
    
    # Add user prompt to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    response = ollama.chat(
        model="llama3",
        messages=conversation_history
    )

    reply = response["message"]["content"]
    
    # Add assistant response to memory
    conversation_history.append({"role": "assistant", "content": reply})
    
    # Extract potential car interests from the conversation
    # This is a simple heuristic - you might want to make this more sophisticated
    car_keywords = ["interested in", "like", "prefer", "looking for", "want", "considering"]
    for keyword in car_keywords:
        if keyword in prompt.lower():
            # Extract the phrase after the keyword
            parts = prompt.lower().split(keyword)
            if len(parts) > 1:
                potential_interest = parts[1].strip().split('.')[0].strip()
                if any(car_term in potential_interest.lower() for car_term in ["car", "vehicle", "model", "brand", "make"]):
                    update_user_interests(user_name, potential_interest)
    
    return reply

# ========== Speak ==========
def speak(text):
    print("🗣️ Responding...")
    engine.say(text)
    engine.runAndWait()

# ========== Main Loop ==========
if __name__ == "__main__":
    user_profiles = load_user_profiles()
    current_user = None

    while True:
        try:
            if not current_user:
                current_user = input("🚗 Welcome! What's your name? ").strip()
                if current_user in user_profiles:
                    interests = user_profiles[current_user]["interests"]
                    if interests:
                        print(f"👋 Welcome back, {current_user}!")
                        print(f"Based on our previous conversations, you're interested in: {', '.join(interests)}")
                    else:
                        print(f"👋 Welcome back, {current_user}!")
                else:
                    print(f"👋 Nice to meet you, {current_user}!")
                    user_profiles[current_user] = {"interests": []}
                    save_user_profiles(user_profiles)

            user_input = input("\n💬 Type your message (or press Enter to speak): ").strip()

            # Exit condition
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            # Reset context
            if user_input.lower() in ["start over", "reset chat"]:
                conversation_history.clear()
                print("🔄 Chat history cleared.")
                continue

            # Use voice if input is empty
            if not user_input:
                user_input = record_and_transcribe()
                print("🎧 You said:", user_input)

            if not user_input.strip():
                continue  # Skip if still empty

            # Add user name and interests to the prompt for context
            user_context = f"User name: {current_user}"
            if current_user in user_profiles and user_profiles[current_user]["interests"]:
                user_context += f". Previous interests: {', '.join(user_profiles[current_user]['interests'])}"
            full_prompt = f"{user_context}\n{user_input}"

            reply = chat_with_llama(full_prompt, current_user)
            print("🤖 Car Dealer:", reply)
            speak(reply)

        except KeyboardInterrupt:
            print("\n🚪 Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue
