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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ========== Configuration ==========
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ========== Translator Configuration ==========
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Kannada": "kn", "Marathi": "mr", "Tamil": "ta",
    "Telugu": "te", "Spanish": "es", "French": "fr", "Haryanvi": "bgc", "Arabic": "ar",
    "Bengali": "bn", "Chinese": "zh", "German": "de", "Japanese": "ja", "Portuguese": "pt",
    "Russian": "ru", "Urdu": "ur", "Punjabi": "pa", "Gujarati": "gu", "Malayalam": "ml",
    "Odia": "or", "Italian": "it", "Dutch": "nl", "Korean": "ko", "Turkish": "tr", 
    "Vietnamese": "vi", "Thai": "th"
}

history_file = "translated_output.txt"

# ========== Conversation Analyzer ==========
class ConversationAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _read_file_content(self, uploaded_file):
        filename = uploaded_file.filename.lower()
        if filename.endswith('.pdf'):
            try:
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                return "\n".join([page.get_text() for page in doc])
            except Exception as e:
                raise ValueError(f"Failed to read PDF file: {e}")
        elif filename.endswith('.txt'):
            try:
                return uploaded_file.read().decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to read TXT file: {e}")
        else:
            raise ValueError("Unsupported file format. Only .txt and .pdf are allowed.")

    def analyze_conversation(self, input_data, max_length: int = 200, temperature: float = 0.7):
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

# Initialize components
try:
    model_processor = ConversationAnalyzer(model_name="gpt2")
except Exception as e:
    model_processor = None
    print(f"Error initializing Conversation Analyzer: {e}")

translator_engine = Translator()

# ========== Utility Functions ==========
def get_language_code(language_name):
    return LANGUAGES.get(language_name, "en")

def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        if not sr.Microphone.list_microphone_names():
            return "Microphone not detected"
            
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            recognizer.pause_threshold = 1
            audio = recognizer.listen(source, phrase_time_limit=10)
            
        print("Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"Speech Recognition Error: {e}"
    except OSError as e:
        print(f"Microphone error: {e}")
        return "Microphone not available"
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return "Speech recognition failed"

def translate_text(text, target_lang_code):
    try:
        if text.startswith(("Speech Recognition Error", "Microphone")):
            return text
        translated = translator_engine.translate(text, dest=target_lang_code)
        return translated.text
    except Exception as e:
        print(f"Error in translate_text: {e}")
        return "[Translation error]"

def speak_text(text, lang_code):
    if not text or text.startswith(("Speech Recognition Error", "Microphone")):
        return
    text_to_voice(text, lang_code)

def translate_and_speak(speaker, target_lang_name):
    target_lang_code = get_language_code(target_lang_name)
    spoken_text = recognize_speech()
    
    if not spoken_text:
        return f"{speaker} said nothing or speech could not be recognized."
    if spoken_text.startswith(("Speech Recognition Error", "Microphone")):
        return spoken_text

    translated = translate_text(spoken_text, target_lang_code)
    speak_text(translated, target_lang_code)

    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {speaker} said: {spoken_text}\n")
        f.write(f"{timestamp} Translated: {translated}\n")

    return f"{timestamp} {speaker} said: {spoken_text}\n{timestamp} Translated: {translated}"

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
        print(f"Error in text_to_voice: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def read_conversation_history():
    conversation = []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    line1 = lines[i].strip()
                    line2 = lines[i + 1].strip()
                    speaker = "doctor" if "Doctor said" in line1 else "patient" if "Patient said" in line1 else "unknown"
                    text_part = line2.split("Translated: ")[1] if "Translated: " in line2 else line2
                    timestamp = line2.split("]")[0] + "]" if "]" in line2 else ""
                    conversation.append({"speaker": speaker, "text": f"{timestamp} {text_part}"})
    except FileNotFoundError:
        conversation = []
    return conversation

# ========== Routes ==========
@app.route("/")
def home():
    return redirect(url_for("translator"))

@app.route("/translator", methods=["GET", "POST"])
def translator():
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
            open(history_file, "w").close()
            return jsonify({"status": "cleared"})

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

@app.route("/download_file")
def download_file():
    return send_file(history_file, as_attachment=True)

@app.route("/llm", methods=["GET", "POST"])
def llm():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        model_name = request.form.get("model", "gpt2")
        max_length = int(request.form.get("max_length", 150))
        temperature = float(request.form.get("temperature", 0.7))

        if not uploaded_file or uploaded_file.filename == '':
            return render_template("llm.html", 
                                available_models=["gpt2"], 
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error="Please select a file")

        if not (uploaded_file.filename.endswith('.pdf') or uploaded_file.filename.endswith('.txt')):
            return render_template("llm.html", 
                                available_models=["gpt2"], 
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error="Only PDF and TXT files are allowed")

        try:
            if uploaded_file.filename.endswith('.pdf'):
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
            else:
                text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            return render_template("llm.html", 
                                available_models=["gpt2"], 
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error=f"Error reading file: {str(e)}")

        if not text.strip():
            return render_template("llm.html", 
                                available_models=["gpt2"], 
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error="File is empty")

        if model_processor is None:
            return render_template("llm.html",
                                available_models=["gpt2"],
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error="The analysis model could not be loaded. Please check server logs.")

        try:
            summary = model_processor.analyze_conversation(text, max_length=max_length, temperature=temperature)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            formatted_output = f"""=== Conversation Analysis Report ===
Timestamp: {timestamp}
Model: {model_name}
Temperature: {temperature}
File: {uploaded_file.filename}

--- Summary ---
{summary}
"""
            return render_template("llm.html", 
                                 available_models=["gpt2"], 
                                 default_model=model_name,
                                 default_length=max_length,
                                 default_temp=temperature,
                                 result=formatted_output)
        except Exception as e:
            return render_template("llm.html", 
                                available_models=["gpt2"], 
                                default_model=model_name,
                                default_length=max_length,
                                default_temp=temperature,
                                error=str(e))

    return render_template("llm.html", 
                         available_models=["gpt2"], 
                         default_model="gpt2", 
                         default_length=150, 
                         default_temp=0.7)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not (uploaded_file.filename.endswith('.pdf') or uploaded_file.filename.endswith('.txt')):
            return jsonify({"error": "Invalid file type. Only PDF and TXT allowed"}), 400

        file_content = ""
        try:
            if uploaded_file.filename.endswith('.pdf'):
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                file_content = "\n".join([page.get_text() for page in doc])
            else:
                file_content = uploaded_file.read().decode('utf-8')
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400

        if not file_content.strip():
            return jsonify({"error": "Empty file content"}), 400

        prompt = """Analyze this doctor-patient conversation and extract the following information:
        - Patient Name: [extract full name]
        - Age: [extract age]
        - Symptoms: [list all symptoms]
        - Diagnosis: [medical condition]
        - Prescribed Medication: [list medications]
        - Follow-up Date: [extract date if mentioned]

        Return the information in this exact format:
        Patient Name: [value]
        Age: [value]
        Symptoms: [value]
        Diagnosis: [value]
        Prescribed Medication: [value]
        Follow-up Date: [value]

        Conversation:
        """ + file_content[:2000]

        try:
            response = model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from AI model")
                
            extracted = response.text

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            pdf.cell(200, 10, txt="Medical Consultation Summary", ln=1, align='C')
            pdf.ln(10)
            
            for line in extracted.splitlines():
                if ':' in line:
                    field, value = line.split(':', 1)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(50, 10, txt=f"{field}:", ln=0)
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, txt=value.strip())
                    pdf.ln(5)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                pdf.output(temp_pdf.name)
                temp_path = temp_pdf.name

            return jsonify({
                "success": True,
                "form_data": extracted,
                "download_link": f"/download_report?file={os.path.basename(temp_path)}"
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/download_report")
def download_report():
    try:
        file_name = request.args.get("file")
        if not file_name:
            return "Missing file name", 400

        if not file_name.endswith('.pdf'):
            return "Invalid file type", 400

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        
        if not os.path.exists(file_path):
            return "File not found", 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name="medical_report.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Download error: {e}")
        traceback.print_exc()
        return f"Download error: {str(e)}", 500

# ========== WebSocket Routes ==========
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

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
        print(f"Error in handle_speak_text: {e}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
