
import os
import sqlite3
import time
from pathlib import Path
from flask import Flask, request, jsonify
import numpy as np

import whisper
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write

from sentence_transformers import SentenceTransformer
import faiss
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "assistant_memory.db"
DOCS_DIR = Path("knowledge")

embed_model = SentenceTransformer(EMBED_MODEL)
sample_vec = embed_model.encode(["test"])
index = faiss.IndexFlatL2(sample_vec.shape[1])
docs = []

whisper_model = whisper.load_model("base")
tts_engine = pyttsx3.init()

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        role TEXT,
        content TEXT,
        ts REAL
    )""")
    conn.commit()
    return conn

db = init_db()

def save_message(role, content):
    db.execute("INSERT INTO messages (role, content, ts) VALUES (?, ?, ?)",
               (role, content, time.time()))
    db.commit()

def ingest_docs():
    texts = []
    for f in DOCS_DIR.glob("*.txt"):
        txt = f.read_text(encoding="utf-8")
        texts.append(txt)
        docs.append(txt)
    if texts:
        emb = embed_model.encode(texts)
        index.add(np.array(emb).astype("float32"))

def retrieve(query, k=3):
    if index.ntotal == 0:
        return ""
    qv = embed_model.encode([query])
    D, I = index.search(np.array(qv).astype("float32"), k)
    results = []
    for idx in I[0]:
        if idx < len(docs):
            results.append(docs[idx][:500])
    return "\n".join(results)

def generate(prompt):
    if OPENAI_API_KEY:
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6
        )
        return resp.choices[0].text.strip()
    return "OpenAI key not set."

def answer(user_input):
    context = retrieve(user_input)
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nAssistant:"
    response = generate(prompt)
    save_message("user", user_input)
    save_message("assistant", response)
    return response

def record_audio(filename="input.wav", duration=5, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    return filename

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

HTML = """
<!doctype html>
<html>
<head>
<title>Personal AI Assistant (Voice Enabled)</title>
</head>
<body style="font-family:Arial; background:#111; color:white; padding:20px;">
<h2>Personal AI Assistant (Voice Enabled)</h2>
<textarea id="input" rows="3" cols="60"></textarea><br>
<button onclick="send()">Send</button>
<button onclick="voice()">Speak</button>
<div id="output" style="margin-top:20px;"></div>

<script>
async function send(){
    const text = document.getElementById("input").value;
    const res = await fetch("/api/query",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({q:text})
    });
    const data = await res.json();
    document.getElementById("output").innerText = data.answer;
}

async function voice(){
    const res = await fetch("/api/voice");
    const data = await res.json();
    document.getElementById("output").innerText = data.answer;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    ans = answer(data["q"])
    return jsonify({"answer": ans})

@app.route("/api/voice")
def api_voice():
    file = record_audio()
    text = transcribe_audio(file)
    ans = answer(text)
    speak(ans)
    return jsonify({"answer": ans})

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    ingest_docs()
    app.run(port=5000, debug=True)
