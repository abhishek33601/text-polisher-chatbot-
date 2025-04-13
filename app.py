from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import speech_recognition as sr
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime

app = Flask(__name__)

# Hardcoded API key
GEMINI_API_KEY = "AIzaSyABrufViTD3BTCqDP89hGGUvnIdxVIRy4U"

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store chat history
chat_history = {}

def process_text_with_gemini(text):
    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    # Create a comprehensive prompt for text improvement
    prompt = f"""You are an advanced text improvement AI. Please analyze and improve the following text based on its type and context.

Input text: "{text}"

Please follow these steps:
1. First identify the type of text (casual request, formal statement, question, command, etc.)
2. Then improve it according to these rules:

For casual requests or informal text:
- Make it polite and clear while keeping it casual
- Add necessary context and courtesy words
- Keep the friendly tone
- Make it sound natural

For formal text:
- Improve grammar and structure
- Enhance clarity and professionalism
- Maintain the appropriate tone
- Fix any technical terminology if present

For questions:
- Make them clear and specific
- Add necessary context
- Keep the appropriate level of formality

For commands or instructions:
- Make them clear and actionable
- Add courtesy when appropriate
- Maintain authority when needed

General improvements:
- Fix grammar and spelling
- Improve sentence structure
- Enhance clarity and readability
- Keep the original intent and meaning
- Make it appropriate for the context

Please provide:
1. The improved version
2. A brief explanation of what was improved
3. The type of text identified"""

    try:
        response = text_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Low temperature for consistent results
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        if not response.text:
            raise Exception("No response from Gemini AI")
        
        # Parse the response
        parts = response.text.split('\n\n')
        improved_text = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else "Text has been improved for clarity and appropriateness"
        text_type = parts[2].strip() if len(parts) > 2 else "General text"
        
        return {
            'improved_text': improved_text,
            'explanation': explanation,
            'text_type': text_type
        }
        
    except Exception as e:
        raise Exception(f"Error processing text: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fix_text', methods=['POST'])
def fix_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        session_id = data.get('session_id', 'default')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Process the text
        result = process_text_with_gemini(text)
        
        # Store in chat history
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        chat_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'original': text,
            'fixed': result['improved_text'],
            'explanation': result['explanation'],
            'text_type': result['text_type']
        })
        
        return jsonify({
            'original_text': text,
            'fixed_text': result['improved_text'],
            'explanation': result['explanation'],
            'text_type': result['text_type']
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the image temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image with Gemini
        image = Image.open(filepath)
        response = vision_model.generate_content([
            "Analyze this image and provide: 1. A detailed description, 2. Suggestions for improvement if applicable, 3. Any interesting observations:",
            image
        ])
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'description': response.text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        audio_data = request.files['audio']
        session_id = request.form.get('session_id', 'default')
        recognizer = sr.Recognizer()
        
        # Convert audio to text
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        # Process the text with improved logic
        result = process_text_with_gemini(text)
        
        # Store in chat history
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        chat_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'original': text,
            'fixed': result['improved_text'],
            'explanation': result['explanation'],
            'text_type': result['text_type'],
            'type': 'voice'
        })
        
        return jsonify({
            'original_text': text,
            'fixed_text': result['improved_text'],
            'explanation': result['explanation'],
            'text_type': result['text_type'],
            'session_id': session_id
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        session_id = request.args.get('session_id', 'default')
        if session_id in chat_history:
            return jsonify(chat_history[session_id])
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        session_id = request.json.get('session_id', 'default')
        if session_id in chat_history:
            chat_history[session_id] = []
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 