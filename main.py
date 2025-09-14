import os
import json
import asyncio
import numpy as np
import pandas as pd
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from groq import Groq
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import logging
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
import threading
from queue import Queue
import io
import wave
from TTS.api import TTS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This is CRITICAL for Render

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Configuration with all model options
@dataclass
class Config:
    # API Keys (loaded via .env)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Groq Model Options
    GROQ_MODELS: Dict[str, str] = {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
        "mixtral-8x7b": "mixtral-8x7b-32768"
    }
    CURRENT_GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    # Gemini Model Options
    GEMINI_MODELS: Dict[str, str] = {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash"
    }
    CURRENT_GEMINI_MODEL: str = "gemini-1.5-pro"
    
    # Whisper Model Options
    WHISPER_MODELS: List[str] = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    CURRENT_WHISPER_MODEL: str = "medium"
    
    # TTS Model (XTTS - will fallback if GPU unavailable)
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v1"
    
    # Other Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Performance Settings
    SIMILARITY_THRESHOLD: float = 0.65
    MAX_RESPONSE_LENGTH: int = 300
    LANGUAGE_DETECTION_THRESHOLD: float = 0.7
    CORPUS_FILE: str = "enhanced_college_corpus.csv"
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    AUDIO_FORMAT: str = "wav"
    
    # Threading and Concurrency
    MAX_WORKERS: int = 4
    WEBSOCKET_TIMEOUT: int = 300
    
    # College Information (Updated with new data from website)
    COLLEGE_INFO: Dict[str, Any] = {
        "name": "JSPM's Jayawantrao Sawant College of Engineering (JSCOE)",
        "location": "Hadapsar, Pune, Maharashtra, India",
        "address": "Survey No. 58, Hadapsar, Pune - 411028, Maharashtra, India",
        "phone": "+91-20-26998000",
        "email": "info@jspmjscoe.edu.in",
        "website": "https://jspmjscoe.edu.in",
        "established": "2004",
        "affiliation": "Savitribai Phule Pune University (SPPU)",
        "approval": "AICTE Approved"
    }

config = Config()

# Thread pool for CPU intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.embedding_cache = {}
        self.load_lock = threading.Lock()
    
    def get_whisper_model(self):
        model_key = f"whisper_{config.CURRENT_WHISPER_MODEL}"
        if model_key not in self.models:
            with self.load_lock:
                if model_key not in self.models:
                    logger.info(f"Loading Whisper model: {config.CURRENT_WHISPER_MODEL}")
                    self.models[model_key] = whisper.load_model(config.CURRENT_WHISPER_MODEL)
        return self.models[model_key]
    
    def get_embedding_model(self):
        if "embedding" not in self.models:
            with self.load_lock:
                if "embedding" not in self.models:
                    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
                    self.models["embedding"] = SentenceTransformer(config.EMBEDDING_MODEL)
        return self.models["embedding"]

model_manager = ModelManager()

# Initialize API clients
def initialize_clients():
    try:
        groq_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
        
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(config.GEMINI_MODELS[config.CURRENT_GEMINI_MODEL])
        else:
            gemini_model = None
            
        return groq_client, gemini_model
    except Exception as e:
        logger.error(f"Error initializing API clients: {e}")
        return None, None

groq_client, gemini_model = initialize_clients()

class CorpusManager:
    def __init__(self):
        self.corpus_data = None
        self.corpus_embeddings = None
        self.load_corpus()
    
    def load_corpus(self):
        try:
            if os.path.exists(config.CORPUS_FILE):
                self.corpus_data = pd.read_csv(config.CORPUS_FILE)
                questions = self.corpus_data['Question'].tolist()
                
                embedding_model = model_manager.get_embedding_model()
                self.corpus_embeddings = embedding_model.encode(questions)
                logger.info(f"Loaded corpus with {len(questions)} questions")
            else:
                logger.warning(f"{config.CORPUS_FILE} not found. Creating sample corpus...")
                self.create_sample_corpus()
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            self.create_sample_corpus()
    
    def create_sample_corpus(self):
        """Create sample corpus with updated info from jspmjscoe.edu.in"""
        sample_data = {
            'Category': [
                'General', 'Admissions', 'Courses', 'Facilities', 'Contact',
                'Placements', 'Faculty', 'Campus', 'Fees', 'Scholarships',
                'Events', 'News'
            ],
            'Question': [
                'What is JSPM JSCOE?',
                'How to get admission in JSCOE?',
                'What courses are offered?',
                'What facilities are available?',
                'What are the contact details?',
                'How are the placements?',
                'Tell me about faculty',
                'Describe the campus',
                'What are the fees?',
                'Are scholarships available?',
                'What are the upcoming events?',
                'What recent news has JSCOE published?'
            ],
            'Answer': [
                f"{config.COLLEGE_INFO['name']} is an engineering college established in {config.COLLEGE_INFO['established']} located in {config.COLLEGE_INFO['location']}. It is affiliated with {config.COLLEGE_INFO['affiliation']} and {config.COLLEGE_INFO['approval']}.",
                "Admissions are based on MHT-CET scores and JEE Main scores. Visit our website for detailed admission process and eligibility criteria.",
                "We offer B.Tech programs in Computer Engineering, Information Technology, Electronics & Telecommunication, Mechanical Engineering, Civil Engineering, and Electrical Engineering.",
                "Our campus features modern laboratories, library, hostels, sports facilities, auditorium, cafeteria, and Wi-Fi enabled campus with excellent infrastructure.",
                f"Contact us at {config.COLLEGE_INFO['phone']}, email: {config.COLLEGE_INFO['email']}, website: {config.COLLEGE_INFO['website']}. Address: {config.COLLEGE_INFO['address']}",
                "We have excellent placement records with top companies like TCS, Infosys, Wipro, Cognizant, Accenture visiting our campus. Average package ranges from 3-8 LPA.",
                "Our faculty consists of experienced professors and industry experts with PhD and M.Tech qualifications, providing quality education and mentorship.",
                "The campus is spread over acres with green environment, modern buildings, well-equipped labs, sports ground, and all necessary amenities for students.",
                "Fees vary by program. Engineering programs typically range from 80,000 to 1,50,000 per year. Contact admission office for detailed fee structure.",
                "Yes, we provide merit-based scholarships, government scholarships, and financial assistance for deserving students. Apply during admission process.",
                "Upcoming events include Industrial Visits for SE & TE students on August 7-8, 2025. Check our website for more updates.",
                "Recent news includes industrial visits, recognition of Dr. Prakash Kadam as a Lean Manufacturing Expert, and outstanding alumni placements at Kantar, Zensar, Infosys, and more."
            ]
        }
        
        self.corpus_data = pd.DataFrame(sample_data)
        self.corpus_data.to_csv(config.CORPUS_FILE, index=False)
        
        embedding_model = model_manager.get_embedding_model()
        self.corpus_embeddings = embedding_model.encode(sample_data['Question'])
        logger.info("Created and saved sample corpus data with latest info")

corpus_manager = CorpusManager()

class AudioProcessor:
    @staticmethod
    def process_audio_chunk(audio_data: bytes) -> np.ndarray:
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return np.array([])

    @staticmethod
    def speech_to_text(audio_data: np.ndarray) -> str:
        try:
            whisper_model = model_manager.get_whisper_model()
            result = whisper_model.transcribe(audio_data)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return ""

class TTSProcessor:
    _tts = None
    
    @classmethod
    def get_tts(cls):
        if cls._tts is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._tts = TTS(model_name=config.TTS_MODEL).to(device)
                logger.info(f"Loaded TTS model: {config.TTS_MODEL} on {device}")
            except Exception as e:
                logger.error(f"Error loading TTS model: {e}")
                cls._tts = None
        return cls._tts
    
    @staticmethod
    def text_to_speech(text: str) -> bytes:
        if TTSProcessor.get_tts() is None:
            logger.error("TTS model not available")
            return b""
            
        try:
            temp_file = "temp_audio.wav"
            TTSProcessor.get_tts().tts_to_file(text=text, file_path=temp_file)
            
            with open(temp_file, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

class ResponseGenerator:
    @staticmethod
    def retrieve_from_corpus(query: str) -> Optional[str]:
        try:
            if corpus_manager.corpus_embeddings is None or len(corpus_manager.corpus_embeddings) == 0:
                return None
                
            embedding_model = model_manager.get_embedding_model()
            query_embedding = embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, corpus_manager.corpus_embeddings)[0]
            max_idx = np.argmax(similarities)
            
            if similarities[max_idx] >= config.SIMILARITY_THRESHOLD:
                logger.info(f"Found corpus match with similarity {similarities[max_idx]:.3f}")
                return corpus_manager.corpus_data.iloc[max_idx]['Answer']
            return None
        except Exception as e:
            logger.error(f"Error retrieving from corpus: {e}")
            return None
    
    @staticmethod
    def is_education_related(query: str) -> bool:
        education_keywords = [
            'education', 'college', 'university', 'course', 'degree', 'admission',
            'career', 'job', 'placement', 'engineering', 'technology', 'computer',
            'jspm', 'jscoe', 'hadapsar', 'fee', 'scholarship', 'faculty', 'campus',
            'event', 'news', 'industrial visit', 'lean manufacturing', 'placements',
            'kantar', 'zensar', 'infosys', 'prakash kadam'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in education_keywords)
    
    @staticmethod
    async def generate_groq_response(query: str, context: Optional[str] = None) -> str:
        try:
            if not groq_client:
                return "Groq client not available. Please check API configuration."
                
            system_prompt = f"""
            You are JSPM Bot, an AI assistant for {config.COLLEGE_INFO['name']} located in {config.COLLEGE_INFO['location']}.
            Provide accurate information about college courses, admissions, facilities, placements, faculty, events, and news.
            Be polite, professional, and conversational. Keep responses concise and under 80 words.
            
            College Details:
            - Address: {config.COLLEGE_INFO['address']}
            - Phone: {config.COLLEGE_INFO['phone']}
            - Email: {config.COLLEGE_INFO['email']}
            - Website: {config.COLLEGE_INFO['website']}
            """
            
            if context:
                system_prompt += f"\n\nUse this context: {context}"
            
            chat_completion = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    model=config.GROQ_MODELS[config.CURRENT_GROQ_MODEL],
                    temperature=0.7,
                    max_tokens=config.MAX_RESPONSE_LENGTH
                )
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating Groq response: {e}")
            return "I'm having trouble processing your request. Please try again."
    
    @staticmethod
    async def generate_gemini_response(query: str) -> str:
        try:
            if not gemini_model:
                return "Gemini client not available. Please check API configuration."
                
            prompt = f"""
            You are an educational assistant for {config.COLLEGE_INFO['name']}.
            Provide helpful information about education, placements, events, and college life.
            Query: {query}
            Keep response under 80 words and be conversational.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: gemini_model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return "I'm having trouble generating a response. Please try again."

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"Client connected. Total connections: {self.connection_count}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_count -= 1
            logger.info(f"Client disconnected. Total connections: {self.connection_count}")

manager = ConnectionManager()

# FastAPI app
app = FastAPI(title="JSPM JSCOE Voice AI Agent", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend.html at root
@app.get("/")
async def get_frontend():
    return FileResponse("frontend.html")

# Mount static files if needed (optional)
app.mount("/static", StaticFiles(directory="."), name="static")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "models_loaded": len(model_manager.models),
        "corpus_size": len(corpus_manager.corpus_data) if corpus_manager.corpus_data is not None else 0,
        "tts_available": TTSProcessor.get_tts() is not None,
        "groq_configured": bool(config.GROQ_API_KEY),
        "gemini_configured": bool(config.GEMINI_API_KEY),
    }

# Websocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            try:
                audio_data = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=config.WEBSOCKET_TIMEOUT
                )
            except asyncio.TimeoutError:
                await websocket.send_text("TIMEOUT: Session expired due to inactivity")
                break
            
            start_time = time.time()
            
            audio_array = AudioProcessor.process_audio_chunk(audio_data)
            if len(audio_array) == 0:
                await websocket.send_text("ERROR: Invalid audio data")
                continue
            
            user_query = await asyncio.get_event_loop().run_in_executor(
                executor, AudioProcessor.speech_to_text, audio_array
            )
            
            if not user_query.strip():
                await websocket.send_text("AI: I didn't catch that. Could you please repeat?")
                continue
            
            await websocket.send_text(f"USER: {user_query}")
            
            corpus_answer = await asyncio.get_event_loop().run_in_executor(
                executor, ResponseGenerator.retrieve_from_corpus, user_query
            )
            
            if corpus_answer:
                response_text = await ResponseGenerator.generate_groq_response(user_query, corpus_answer)
            else:
                if ResponseGenerator.is_education_related(user_query):
                    response_text = await ResponseGenerator.generate_gemini_response(user_query)
                else:
                    response_text = f"I specialize in information about {config.COLLEGE_INFO['name']}. How can I help you with your academic journey?"
            
            # Convert response to audio
            audio_bytes = TTSProcessor.text_to_speech(response_text)
            if audio_bytes:
                await websocket.send_bytes(audio_bytes)
            else:
                # Fallback to text if TTS fails (common on Render)
                await websocket.send_text(f"AI: {response_text}")
            
            processing_time = time.time() - start_time
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        await websocket.send_text(f"ERROR: {str(e)}")
        manager.disconnect(websocket)

# ✅ COMMENT OUT OR DELETE THIS ENTIRE BLOCK FOR RENDER DEPLOYMENT
# if __name__ == "__main__":
#     logger.info("Starting JSPM JSCOE Voice AI Agent Backend...")
#     logger.info(f"Configuration: Groq({config.CURRENT_GROQ_MODEL}), Gemini({config.CURRENT_GEMINI_MODEL}), Whisper({config.CURRENT_WHISPER_MODEL})")
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000, 
#         log_level="info",
#         access_log=True
#     )
