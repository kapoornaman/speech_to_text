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
from collections import deque

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

# ========== Conversation Management ==========
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in memory
SUMMARY_THRESHOLD = 5    # Number of messages before summarizing

def summarize_conversation(messages):
    """Summarize the conversation history using LLaMA."""
    if not messages:
        return ""
    
    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation between a car dealer and a customer, focusing on key points about car interests, preferences, and important details. Keep it concise but informative."},
        {"role": "user", "content": "Please summarize this conversation:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])}
    ]
    
    try:
        response = ollama.chat(
            model="llama3",
            messages=summary_prompt
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error summarizing conversation: {str(e)}")
        return ""

def get_conversation_context(messages, summary):
    """Get the current conversation context including summary and recent messages."""
    context = []
    
    # Add system message
    context.append(messages[0])  # Original system message
    
    # Add conversation summary if available
    if summary:
        context.append({
            "role": "system",
            "content": f"Previous conversation summary: {summary}"
        })
    
    # Add recent messages
    for msg in messages[-MAX_HISTORY_LENGTH:]:
        if msg["role"] != "system":  # Skip system messages as they're already included
            context.append(msg)
    
    return context

# ========== Memory ==========
conversation_history = deque(maxlen=MAX_HISTORY_LENGTH)
conversation_summary = ""
conversation_history.append({"role": "system", "content": (
    "You are a friendly and knowledgeable car dealer. "
    "You only talk about cars, car models, features, and related topics. "
    "For anything else, you should say 'I'm sorry, I can only talk about cars.' "
    "Start every new conversation by asking the user's name. "
    "Remember the user's name and their car interests for future conversations. "
    "If a returning user provides their name, greet them and ask about their previous interests. "
    "Pay attention to any car-related interests the user mentions during the conversation "
    "and remember these for future interactions. "
    "If you detect a new car interest, acknowledge it and add it to the user's profile."
)})

USER_DATA_FILE = "user_profiles.json"

def load_user_profiles():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_profiles(profiles):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(profiles, f)

def is_valid_car_interest(text):
    """Validate if the interest is car-related, budget-related, or makes sense."""
    # Keywords that indicate car-related interests
    car_terms = [
        "car", "vehicle", "suv", "sedan", "truck", "van", "crossover", "hatchback",
        "brand", "make", "model", "year", "mileage", "fuel", "electric", "hybrid",
        "automatic", "manual", "transmission", "engine", "horsepower", "torque",
        "awd", "4wd", "fwd", "rwd", "drivetrain", "safety", "features", "price",
        "budget", "cost", "affordable", "luxury", "premium", "economy", "efficient",
        "spacious", "compact", "family", "sport", "performance", "comfort", "reliable"
    ]
    
    # Keywords that indicate budget-related terms
    budget_terms = [
        "budget", "price", "cost", "affordable", "expensive", "cheap", "value",
        "payment", "finance", "lease", "loan", "down payment", "monthly",
        "dollar", "euro", "currency", "expensive", "luxury", "premium", "economy"
    ]
    
    text_lower = text.lower()
    
    # Check if the text contains any car-related or budget-related terms
    has_car_term = any(term in text_lower for term in car_terms)
    has_budget_term = any(term in text_lower for term in budget_terms)
    
    # Additional validation rules
    is_reasonable_length = 3 <= len(text.split()) <= 15  # Not too short, not too long
    has_no_special_chars = all(c.isalnum() or c.isspace() or c in ".,-" for c in text)
    
    # The interest is valid if it's either car-related or budget-related, and meets other criteria
    return (has_car_term or has_budget_term) and is_reasonable_length and has_no_special_chars

def extract_car_interests(text):
    """Extract and validate potential car interests from any text."""
    car_keywords = ["interested in", "like", "prefer", "looking for", "want", "considering", "thinking about"]
    interests = set()  # Use a set to automatically handle duplicates
    
    for keyword in car_keywords:
        if keyword in text.lower():
            parts = text.lower().split(keyword)
            if len(parts) > 1:
                # Split by common sentence endings and clean up
                potential_interests = [p.strip().split('.')[0].strip() for p in parts[1:]]
                for interest in potential_interests:
                    # Clean up the interest text
                    interest = ' '.join(interest.split())  # Normalize whitespace
                    if is_valid_car_interest(interest):
                        interests.add(interest)
    
    return list(interests)

def update_user_interests(user_name, new_interests):
    """Update user interests with deduplication and validation."""
    if user_name not in user_profiles:
        user_profiles[user_name] = {"interests": []}
    
    # Convert existing interests to lowercase for comparison
    current_interests = {interest.lower(): interest for interest in user_profiles[user_name]["interests"]}
    
    # Process new interests
    for interest in new_interests:
        interest_lower = interest.lower()
        
        # Skip if it's a duplicate (case-insensitive)
        if interest_lower in current_interests:
            continue
            
        # Validate the interest
        if is_valid_car_interest(interest):
            # If we have a similar interest, update it with the new one if it's more specific
            similar_exists = False
            for existing_lower, existing in current_interests.items():
                # Check if one is a subset of the other
                if (interest_lower in existing_lower or existing_lower in interest_lower):
                    # Keep the more specific version
                    if len(interest) > len(existing):
                        current_interests[interest_lower] = interest
                    similar_exists = True
                    break
            
            # If no similar interest exists, add the new one
            if not similar_exists:
                current_interests[interest_lower] = interest
    
    # Update the user profile with the deduplicated and validated interests
    user_profiles[user_name]["interests"] = list(current_interests.values())
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

def chat_with_llama(prompt, user_name):
    print("🤖 Thinking...")
    global conversation_summary
    
    # Extract and validate interests before adding to conversation history
    interests = extract_car_interests(prompt)
    if interests:
        update_user_interests(user_name, interests)
        # Add a system message about the updated interests
        if interests:
            interest_update = f"User has expressed interest in: {', '.join(interests)}"
            conversation_history.append({"role": "system", "content": interest_update})
    
    # Add user prompt to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Get current context including summary and recent messages
    current_context = get_conversation_context(list(conversation_history), conversation_summary)
    
    response = ollama.chat(
        model="llama3",
        messages=current_context
    )

    reply = response["message"]["content"]
    
    # Add assistant response to memory
    conversation_history.append({"role": "assistant", "content": reply})
    
    # Update summary if we've reached the threshold
    if len(conversation_history) >= SUMMARY_THRESHOLD:
        conversation_summary = summarize_conversation(list(conversation_history))
        # Clear old messages but keep the system message and summary
        while len(conversation_history) > 1:
            conversation_history.popleft()
        conversation_history.appendleft({"role": "system", "content": conversation_summary})
    
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
    conversation_summary = ""  # Initialize conversation summary

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
                
                # Reset conversation history for new user
                conversation_history.clear()
                conversation_history.append({"role": "system", "content": (
                    "You are a friendly and knowledgeable car dealer. "
                    "You only talk about cars, car models, features, and related topics. "
                    "For anything else, you should say 'I'm sorry, I can only talk about cars.' "
                    "Start every new conversation by asking the user's name and their car interests. "
                    "Remember the user's name and their car interests for future conversations. "
                    "If a returning user provides their name, greet them and ask about their previous interests. "
                    "Pay attention to any car-related interests the user mentions during the conversation "
                    "and remember these for future interactions. "
                    "If you detect a new car interest, acknowledge it and add it to the user's profile."
                )})
                conversation_summary = ""

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
