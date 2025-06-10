# Voice Assistant Car Dealer

An interactive voice assistant that acts as a car dealer, capable of having natural conversations about cars using speech recognition and text-to-speech capabilities.

## Features

- 🎙️ Voice input using Whisper for accurate speech-to-text
- 🤖 Natural language processing using Ollama (LLaMA model)
- 🗣️ Text-to-speech responses using pyttsx3
- 👤 User profile management with persistent memory
- 🚗 Specialized in car-related conversations

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- A working microphone
- Speakers or headphones

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running and the llama3 model is downloaded:
```bash
ollama pull llama3
```

## Usage

1. Activate the virtual environment if not already activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the assistant:
```bash
python voice_assistant.py
```

3. Follow the prompts:
   - Enter your name when asked
   - Type your message or press Enter to use voice input
   - Type "exit" or "quit" to end the conversation
   - Type "start over" or "reset chat" to clear conversation history

## Commands

- Press Enter: Switch to voice input mode
- Type "exit" or "quit": End the program
- Type "start over" or "reset chat": Clear conversation history

## Note

The assistant is specialized in car-related topics and will politely decline to discuss other subjects.

**Let me know if you want a sample `README.md` or have any issues with these steps!**