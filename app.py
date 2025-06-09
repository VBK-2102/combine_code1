import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime
from googletrans import Translator
import tempfile
import google.generativeai as genai
import pymupdf
from dotenv import load_dotenv
from fpdf import FPDF
import traceback
import base64
from flask_socketio import SocketIO, emit
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ========== Configuration ==========
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in .env file")
    raise ValueError("GOOGLE_API_KEY not found in .env file")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {str(e)}")
    model = None

# ========== Translator Configuration ==========
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Kannada": "kn", "Marathi": "mr", "Tamil": "ta",
    "Telugu": "te", "Spanish": "es", "French": "fr", "Haryanvi": "bgc", "Arabic": "ar",
    "Bengali": "bn", "Chinese": "zh", "German": "de", "Japanese": "ja", "Portuguese": "pt",
    "Russian": "ru", "Urdu": "ur", "Punjabi": "pa", "Gujarati": "gu", "Malayalam": "ml",
    "Odia": "or", "Italian": "it", "Dutch": "nl", "Korean": "ko", "Turkish": "tr", 
    "Vietnamese": "vi", "Thai": "th"
}

# Ensure the history directory exists
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
history_file = os.path.join(HISTORY_DIR, "translated_output.txt")

# ========== Conversation Analyzer ==========
class ConversationAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise

    def _read_file_content(self, uploaded_file):
        filename = uploaded_file.filename.lower()
        try:
            if filename.endswith('.pdf'):
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                return "\n".join([page.get_text() for page in doc])
            elif filename.endswith('.txt'):
                return uploaded_file.read().decode('utf-8')
            else:
                raise ValueError("Unsupported file format. Only .txt and .pdf are allowed.")
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            raise

    def analyze_conversation(self, input_data, max_length: int = 200, temperature: float = 0.7):
        try:
            if hasattr(input_data, "filename"):
                conversation_text = self._read_file_content(input_data)
            elif isinstance(input_data, str):
                conversation_text = input_data
            else:
                raise ValueError("Unsupported input data type for analysis")

            if not conversation_text.strip():
                raise ValueError("Input conversation is empty.")

            prompt = f"""Analyze the following doctor-patient conversation and provide a structured summary:

1. Identify the patient's main concerns or symptoms
2. Note any diagnoses or assessments made by the doctor
3. List any recommended treatments or next steps
4. Highlight important follow-up information

Conversation:
{conversation_text.strip()}

Analysis Summary:
"""
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )

            full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            summary_start = full_output.find("Analysis Summary:") + len("Analysis Summary:")
            summary = full_output[summary_start:].strip()

            if "Conversation:" in summary:
                summary = summary.split("Conversation:")[0].strip()

            return summary
        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")
            raise

# Initialize components with error handling
try:
    model_processor = ConversationAnalyzer(model_name="gpt2")
except Exception as e:
    model_processor = None
    logger.error(f"Error initializing Conversation Analyzer: {e}")

try:
    translator_engine = Translator()
except Exception as e:
    logger.error(f"Error initializing translator: {e}")
    translator_engine = None

# ========== Utility Functions ==========
def get_language_code(language_name):
    return LANGUAGES.get(language_name, "en")

def recognize_speech():
    try:
        # Check if we're in a server environment without audio
        if os.environ.get('SERVER_ENVIRONMENT') == 'true':
            return "Microphone not available in server mode"
            
        recognizer = sr.Recognizer()
        mic_list = sr.Microphone.list_microphone_names()
        
        if not mic_list:
            return "Microphone not detected"
            
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Listening...")
            recognizer.pause_threshold = 1
            audio = recognizer.listen(source, phrase_time_limit=10)
            
        logger.info("Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech Recognition Error: {e}"
    except OSError as e:
        logger.error(f"Microphone error: {e}")
        return "Microphone not available"
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}")
        return "Speech recognition failed"

def translate_text(text, target_lang_code):
    try:
        if not text or text.startswith(("Speech Recognition Error", "Microphone")):
            return text
            
        if translator_engine is None:
            return "[Translation service not available]"
            
        translated = translator_engine.translate(text, dest=target_lang_code)
        return translated.text
    except Exception as e:
        logger.error(f"Error in translate_text: {e}")
        return "[Translation error]"

def speak_text(text, lang_code):
    if not text or text.startswith(("Speech Recognition Error", "Microphone")):
        return
        
    try:
        text_to_voice(text, lang_code)
    except Exception as e:
        logger.error(f"Error in speak_text: {e}")

def translate_and_speak(speaker, target_lang_name):
    try:
        target_lang_code = get_language_code(target_lang_name)
        if not target_lang_code:
            return f"Invalid target language: {target_lang_name}"
            
        spoken_text = recognize_speech()
        if not spoken_text:
            return f"{speaker} said nothing or speech could not be recognized."
        if spoken_text.startswith(("Speech Recognition Error", "Microphone")):
            return spoken_text

        translated = translate_text(spoken_text, target_lang_code)
        speak_text(translated, target_lang_code)

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_entry = f"{timestamp} {speaker} said: {spoken_text}\n{timestamp} Translated: {translated}"

        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except IOError as e:
            logger.error(f"Error writing to history file: {e}")

        return log_entry
    except Exception as e:
        logger.error(f"Error in translate_and_speak: {e}")
        return f"Error in translation process: {str(e)}"

def text_to_voice(text_data, to_language):
    try:
        if not text_data:
            return
            
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_path = temp_audio.name
        
        myobj = gTTS(text=text_data, lang=to_language, slow=False)
        myobj.save(temp_path)
        
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        socketio.emit('audio_data', {
            'audio_data': audio_base64,
            'language': to_language,
            'text': text_data
        })
        
        os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error in text_to_voice: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def read_conversation_history():
    conversation = []
    try:
        if not os.path.exists(history_file):
            return conversation
            
        with open(history_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_entry = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "said:" in line:
                if current_entry:
                    conversation.append(current_entry)
                parts = line.split(" said: ")
                timestamp = parts[0].split("]")[0] + "]"
                speaker = parts[0].split(" ")[-1].lower()
                text = parts[1] if len(parts) > 1 else ""
                current_entry = {
                    "speaker": speaker,
                    "original": text,
                    "timestamp": timestamp,
                    "translated": ""
                }
            elif "Translated:" in line and current_entry:
                parts = line.split("Translated: ")
                current_entry["translated"] = parts[1] if len(parts) > 1 else ""
                
        if current_entry:
            conversation.append(current_entry)
            
        return conversation[-50:]  # Return last 50 entries
    except Exception as e:
        logger.error(f"Error reading conversation history: {e}")
        return []

# ========== Routes ==========
@app.route("/")
def home():
    return redirect(url_for("translator"))

@app.route("/translator", methods=["GET", "POST"])
def translator():
    try:
        if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            doctor_lang = request.form.get("doctor_lang", "English")
            patient_lang = request.form.get("patient_lang", "Hindi")
            action = request.form.get("action")

            if action == "doctor":
                message = translate_and_speak("Doctor", patient_lang)
                return jsonify({
                    "message": message,
                    "sender": "doctor",
                    "doctor_lang": doctor_lang,
                    "patient_lang": patient_lang
                })
            elif action == "patient":
                message = translate_and_speak("Patient", doctor_lang)
                return jsonify({
                    "message": message,
                    "sender": "patient",
                    "doctor_lang": doctor_lang,
                    "patient_lang": patient_lang
                })
            elif action == "clear":
                try:
                    open(history_file, "w").close()
                    return jsonify({"status": "cleared"})
                except IOError as e:
                    return jsonify({"status": "error", "message": str(e)}), 500

        doctor_lang = request.args.get("doctor_lang", "English")
        patient_lang = request.args.get("patient_lang", "Hindi")
        conversation = read_conversation_history()
        
        return render_template("translator.html",
                            message="",
                            sender="",
                            doctor_lang=doctor_lang,
                            patient_lang=patient_lang,
                            languages=LANGUAGES,
                            conversation=conversation)
    except Exception as e:
        logger.error(f"Error in translator route: {e}")
        return render_template("error.html", error=str(e)), 500

@app.route("/download_file")
def download_file():
    try:
        if not os.path.exists(history_file):
            return "No conversation history found", 404
            
        return send_file(
            history_file,
            as_attachment=True,
            download_name="conversation_history.txt",
            mimetype='text/plain'
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return str(e), 500

# ========== WebSocket Routes ==========
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('speak_text')
def handle_speak_text(data):
    try:
        text = data.get('text', '')
        lang_code = data.get('lang_code', 'en')
        if text:
            text_to_voice(text, lang_code)
            return {'status': 'success'}
        return {'status': 'error', 'message': 'No text provided'}
    except Exception as e:
        logger.error(f"Error in handle_speak_text: {e}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
