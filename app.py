#!/usr/bin/env python3
"""
AI Audio Pipeline Server with AWS Bedrock Integration
Handles audio transcription, text generation via AWS, and text-to-speech
Uses AWS Bedrock for text generation, local models for audio processing
"""

import os
import io
import json
import base64
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration
)
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np

# Configure FFmpeg path before importing pydub to avoid warnings
import os
ffmpeg_path = None
common_paths = [
    "C:\\Users\\Jake\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe",
    "C:\\ffmpeg\\bin\\ffmpeg.exe",
    "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe", 
    "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe"
]
for path in common_paths:
    if os.path.exists(path):
        ffmpeg_path = path
        break

if ffmpeg_path:
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment
from pydub.utils import which

# Load environment variables from .env file
load_dotenv()

# Configure FFmpeg for pydub immediately
def configure_ffmpeg():
    """Configure FFmpeg paths for pydub"""
    ffmpeg_path = which("ffmpeg")
    if not ffmpeg_path:
        # Try common Windows FFmpeg paths
        common_paths = [
            "C:\\Users\\Jake\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe", 
            "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe"
        ]
        for path in common_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
    
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe") if "ffmpeg.exe" in ffmpeg_path else ffmpeg_path.replace("ffmpeg", "ffprobe")
        print(f"FFmpeg configured at: {ffmpeg_path}")
    else:
        print("Warning: FFmpeg not found in common locations")

# Configure FFmpeg immediately
configure_ffmpeg()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class AudioPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.load_models()
        
        # Initialize AWS Bedrock client for text generation
        self.setup_aws_client()
        
    def setup_aws_client(self):
        """Initialize AWS Bedrock client for text generation"""
        try:
            # Check for bearer token (alternative auth method)
            bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'ABSKd2F3YXdld2FoLWF0LTA5ODg2MDA1ODM0ODpRaEw5UlozcEdrR2VDMVFKWDRqWkZ0eCtZQ1h4RGhvSzdxQy9uMktMWDE3MUw5Z1NCVUxoRElSdndyWT0=')
            
            if bearer_token:
                logger.info("🔐 Using AWS Bearer Token for authentication")
                # For bearer token, we need to set up custom headers
                # This requires a different approach with boto3
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1')
                )
                # We'll handle the bearer token in the request headers
                self.use_bearer_token = True
                self.bearer_token = bearer_token
            else:
                logger.info("🔑 Using standard AWS credentials")
                # Initialize Bedrock client with standard credentials
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1')
                )
                self.use_bearer_token = False
            
            # Test connection
            self.test_aws_connection()
            self.aws_available = True
            logger.info("✅ AWS Bedrock client initialized successfully!")
            
        except NoCredentialsError:
            logger.warning("❌ AWS credentials not found. Please configure AWS credentials.")
            logger.warning("   You can set up credentials using:")
            logger.warning("   1. AWS CLI: aws configure")
            logger.warning("   2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            logger.warning("   3. IAM roles (if running on EC2)")
            self.aws_available = False
            
        except Exception as e:
            logger.warning(f"❌ AWS Bedrock initialization failed: {e}")
            logger.warning("   Falling back to simple response generation")
            self.aws_available = False
    
    def test_aws_connection(self):
        """Test AWS Bedrock connection"""
        try:
            # Try a minimal test with Claude 3.5 Sonnet v2 inference profile
            response = self.bedrock_client.invoke_model(
                modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, can you respond?"
                        }
                    ],
                    "max_tokens": 10,
                    "temperature": 0.7
                })
            )
            logger.info("AWS Bedrock connection test successful")
        except Exception as e:
            logger.warning(f"AWS Bedrock connection test failed: {e}")
            raise e
        
    def load_models(self):
        """Load local AI models (Whisper for STT, keep TTS local)"""
        logger.info("Loading local AI models...")
        
        # Load Whisper for speech-to-text (keeping local for better privacy/speed)
        logger.info("Loading Whisper model...")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.whisper_model.to(self.device)
        
        # Remove local language model loading since we're using AWS
        logger.info("Text generation will use AWS Bedrock")
        
        # Initialize TTS (keeping local)
        logger.info("Loading TTS model...")
        try:
            # Using gTTS (Google Text-to-Speech) as it's Python 3.13 compatible
            # Test gTTS availability
            test_tts = gTTS(text="test", lang='en')
            self.tts_available = True
            logger.info("gTTS (Google Text-to-Speech) initialized successfully!")
        except Exception as e:
            logger.warning(f"Could not initialize gTTS: {e}. TTS will be disabled.")
            self.tts_available = False
        
        logger.info("Local models loaded successfully!")
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Convert audio to text using Whisper"""
        try:
            # Ensure audio is the right format
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
            
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Process with Whisper
            input_features = self.whisper_processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
            
            return transcription.strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def generate_response(self, text):
        """Generate AI response using AWS Bedrock or fallback to simple response"""
        try:
            if self.aws_available:
                return self.generate_aws_response(text)
            else:
                return self.generate_fallback_response(text)
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm sorry, I couldn't process that request at the moment."
    
    def generate_aws_response(self, text):
        """Generate response using AWS Bedrock Claude model"""
        try:
            # Prepare the message for Claude 3.5 Sonnet v2
            messages = [
                {
                    "role": "user",
                    "content": f"Please respond to this message in a conversational and helpful manner. Keep your response concise but informative: {text}"
                }
            ]
            
            # Prepare the request body for Claude
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            })
            
            # Call AWS Bedrock
            if hasattr(self, 'use_bearer_token') and self.use_bearer_token:
                # Use direct HTTP request with bearer token
                response = self.invoke_bedrock_with_bearer_token(body)
            else:
                # Use standard boto3 method
                response = self.bedrock_client.invoke_model(
                    modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                    body=body
                )
                # Debug the response
                response_body_raw = response['body'].read()
                logger.info(f"Raw AWS response: {response_body_raw}")
                
                # Parse the Claude response
                response_body = json.loads(response_body_raw)
                logger.info(f"Parsed AWS response: {response_body}")
                assistant_response = response_body['content'][0]['text']
                
            if hasattr(self, 'use_bearer_token') and self.use_bearer_token:
                # Response is already parsed from bearer token method
                assistant_response = response
            else:
                # Response already parsed above
                pass
            
            logger.info(f"AWS Bedrock response generated successfully")
            return assistant_response.strip()
            
        except Exception as e:
            logger.error(f"AWS Bedrock error: {e}")
            # Fallback to simple response
            return self.generate_fallback_response(text)
    
    def invoke_bedrock_with_bearer_token(self, body):
        """Make direct HTTP request to Bedrock using bearer token"""
        import requests
        
        # AWS Bedrock endpoint for Claude 3.5 Sonnet v2 inference profile
        region = os.getenv('AWS_REGION', 'us-east-1')
        model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
        url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.bearer_token}',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.post(url, headers=headers, data=body, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data['content'][0]['text']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Bearer token request failed: {e}")
            raise e
    
    def generate_fallback_response(self, text):
        """Generate a simple fallback response when AWS is not available"""
        # Simple rule-based responses for common patterns
        text_lower = text.lower()
        
        if any(greeting in text_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! How can I help you today?"
        elif any(question in text_lower for question in ['how are you', 'how do you do']):
            return "I'm doing well, thank you for asking! How can I assist you?"
        elif any(thanks in text_lower for thanks in ['thank you', 'thanks', 'appreciate']):
            return "You're welcome! Is there anything else I can help you with?"
        elif 'weather' in text_lower:
            return "I don't have access to current weather data, but you might want to check a weather app or website for the most accurate forecast."
        elif any(question_word in text_lower for question_word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return f"That's an interesting question about '{text}'. I'd be happy to help, but I'm currently running in fallback mode. For more detailed responses, AWS Bedrock integration would be needed."
        else:
            return f"I understand you mentioned '{text}'. I'm currently running in a simple fallback mode. For more intelligent responses, please configure AWS Bedrock credentials."
    
    def text_to_speech(self, text):
        """Convert text to speech using gTTS"""
        try:
            if not self.tts_available:
                logger.warning("TTS not available")
                return None
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Save to temporary file
            tts.save(temp_path)
            
            # Convert MP3 to WAV for better compatibility
            audio = AudioSegment.from_mp3(temp_path)
            wav_path = temp_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            
            # Read the WAV file
            with open(wav_path, "rb") as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            os.unlink(wav_path)
            
            return audio_data
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

# Initialize the pipeline
pipeline = AudioPipeline()

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "device": pipeline.device,
        "aws_available": pipeline.aws_available,
        "tts_available": pipeline.tts_available
    })

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio through the complete pipeline"""
    try:
        # Get audio file from request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Read the file data
        file_data = audio_file.read()
        
        # Convert audio to a standard format using pydub
        # This handles many different audio formats (mp3, wav, m4a, etc.)
        try:
            # Try to load with pydub first (handles more formats)
            audio_segment = AudioSegment.from_file(io.BytesIO(file_data))
            
            # Convert to WAV format for librosa
            wav_data = io.BytesIO()
            audio_segment.export(wav_data, format="wav")
            wav_data.seek(0)
            
            # Now load with librosa
            audio_data, sample_rate = librosa.load(wav_data, sr=None)
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            # Fallback: try librosa directly
            try:
                audio_data, sample_rate = librosa.load(io.BytesIO(file_data), sr=None)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return jsonify({"error": f"Could not process audio format: {str(e)}"}), 400
        
        # Step 1: Transcribe audio to text
        logger.info("Transcribing audio...")
        transcription = pipeline.transcribe_audio(audio_data, sample_rate)
        logger.info(f"Transcription: {transcription}")
        
        if not transcription:
            return jsonify({"error": "Could not transcribe audio"}), 400
        
        # Step 2: Generate AI response
        logger.info("Generating response...")
        response_text = pipeline.generate_response(transcription)
        logger.info(f"Response: {response_text}")
        
        # Step 3: Convert response to speech
        logger.info("Converting to speech...")
        audio_response = pipeline.text_to_speech(response_text)
        
        result = {
            "transcription": transcription,
            "response_text": response_text,
            "audio_available": audio_response is not None,
            "aws_used": pipeline.aws_available
        }
        
        if audio_response:
            # Encode audio as base64 for JSON response
            audio_b64 = base64.b64encode(audio_response).decode('utf-8')
            result["audio_data"] = audio_b64
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test_response.html')
def test_response():
    """Serve the test response page"""
    return send_file('test_response.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('status', {'message': 'Connected to AI pipeline server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the server
    logger.info("Starting AI Pipeline Server with AWS Bedrock integration...")
    
    # For production use, disable Werkzeug reloader and use proper production settings
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, 
                 allow_unsafe_werkzeug=True, use_reloader=False)
