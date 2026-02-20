# Personal AI Assistant (Voice Enabled)

Includes:

- Speech-to-Text using Whisper
- Text-to-Speech using pyttsx3
- RAG with FAISS
- SQLite memory
- Optional OpenAI backend

Setup:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

(Optional) export OPENAI_API_KEY="your-key"

python personal_assistant.py

Open:
http://127.0.0.1:5000/
