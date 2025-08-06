import asyncio
import json
import base64
import io
import logging
from pathlib import Path
from typing import Dict, Optional
import uuid
import tempfile
import os
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

# Voxtral imports
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import torchaudio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))

manager = ConnectionManager()

async def load_model():
    """Load the Voxtral model and processor"""
    global model, processor
    
    try:
        logger.info("Loading Voxtral-Mini-3B-2507...")
        repo_id = "mistralai/Voxtral-Mini-3B-2507"
        
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

async def process_audio(audio_data: bytes, client_id: str):
    """Process audio data and return response"""
    try:
        # Convert audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_webm_path = temp_file.name
        
        # Convert WebM to WAV using FFmpeg
        temp_wav_path = temp_webm_path.replace('.webm', '.wav')
        
        try:
            # Use ffmpeg to convert webm to wav
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', temp_webm_path, 
                '-ar', '16000', 
                '-ac', '1', 
                '-f', 'wav', 
                temp_wav_path, 
                '-y'
            ], check=True, capture_output=True, text=True)
            
            # Load the converted WAV file
            waveform, sample_rate = torchaudio.load(temp_wav_path)
            
            # Ensure mono and correct sample rate
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to numpy array and ensure it's 1D
            audio_array = waveform.squeeze().numpy()
            
            # Check if audio has content
            if len(audio_array) == 0 or np.all(np.abs(audio_array) < 0.001):
                await manager.send_message(client_id, {
                    "type": "error", 
                    "message": "No audio content detected. Please speak louder or check your microphone."
                })
                return
            
            logger.info(f"Audio processed: {len(audio_array)} samples, max amplitude: {np.max(np.abs(audio_array))}")
            
            # Prepare conversation for Voxtral - FIXED FORMAT
            conversation = [{
                "role": "user", 
                "content": [{
                    "type": "audio",
                    "audio": audio_array
                }]
            }]
            
            # Apply chat template
            inputs = processor.apply_chat_template(
                conversation, 
                return_tensors="pt"
            )
            
            # Move to correct device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=500, 
                        temperature=0.7, 
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    input_length = inputs["input_ids"].shape[1]
                else:
                    outputs = model.generate(
                        inputs, 
                        max_new_tokens=500, 
                        temperature=0.7, 
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    input_length = inputs.shape[1]
                
                # Decode only the new tokens
                new_tokens = outputs[:, input_length:]
                decoded_outputs = processor.tokenizer.batch_decode(
                    new_tokens, 
                    skip_special_tokens=True
                )
            
            response_text = decoded_outputs[0] if decoded_outputs and decoded_outputs[0].strip() else "I'm sorry, I couldn't process that audio. Could you please try again?"
            
            # Clean up response text
            response_text = response_text.strip()
            if not response_text:
                response_text = "I heard your audio but couldn't generate a response. Please try speaking more clearly."
            
            await manager.send_message(client_id, {
                "type": "response",
                "text": response_text,
                "status": "success"
            })
            
        finally:
            # Clean up temp files
            for path in [temp_webm_path, temp_wav_path]:
                if os.path.exists(path):
                    os.unlink(path)
                    
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e}")
        await manager.send_message(client_id, {
            "type": "error", 
            "message": "Audio format conversion failed. Please try again."
        })
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        await manager.send_message(client_id, {
            "type": "error", 
            "message": f"Error processing audio: {str(e)}"
        })

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    success = await load_model()
    if not success:
        logger.error("Failed to load model on startup")
    yield
    # Shutdown (if needed)

app = FastAPI(title="Voxtral Audio Chat", version="1.0.0", lifespan=lifespan)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "audio":
                # Check if audio data exists
                if not data.get("audio"):
                    await manager.send_message(client_id, {
                        "type": "error", 
                        "message": "No audio data received"
                    })
                    continue
                
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(data["audio"])
                    
                    # Check if we have actual audio data
                    if len(audio_bytes) < 1000:  # Very small file, likely empty
                        await manager.send_message(client_id, {
                            "type": "error", 
                            "message": "Audio recording too short. Please record for at least 1 second."
                        })
                        continue
                    
                    logger.info(f"Received audio data: {len(audio_bytes)} bytes")
                    
                    # Send processing status
                    await manager.send_message(client_id, {
                        "type": "status", 
                        "message": "Processing audio..."
                    })
                    
                    # Process audio in background
                    asyncio.create_task(process_audio(audio_bytes, client_id))
                    
                except Exception as e:
                    logger.error(f"Error decoding audio: {e}")
                    await manager.send_message(client_id, {
                        "type": "error", 
                        "message": "Failed to decode audio data"
                    })
                
            elif data["type"] == "ping":
                await manager.send_message(client_id, {"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

@app.get("/")
async def get_index():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voxtral Audio Chat</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎤 Voxtral Audio Chat</h1>
                <p>Speak and get AI responses in real-time</p>
            </header>
            
            <main>
                <div class="controls">
                    <button id="startBtn" class="btn btn-primary">🎤 Start Recording</button>
                    <button id="stopBtn" class="btn btn-secondary" disabled>⏹️ Stop Recording</button>
                    <button id="continuousBtn" class="btn btn-tertiary">🔄 Continuous Mode</button>
                    <span id="status" class="status">Ready</span>
                </div>
                
                <div class="chat-container">
                    <div id="messages" class="messages"></div>
                </div>
                
                <div class="info">
                    <p><strong>Supported Languages:</strong> English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian</p>
                    <p><strong>Manual Mode:</strong> Click "Start Recording", speak, then click "Stop Recording".</p>
                    <p><strong>Continuous Mode:</strong> Click "Continuous Mode" for hands-free operation with voice activity detection.</p>
                    <p><strong>Tips:</strong> Ensure your microphone is working and speak in a quiet environment for best results.</p>
                </div>
            </main>
        </div>
        
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Get the TCP port from environment variables (RunPod specific)
    port = int(os.environ.get("RUNPOD_TCP_PORT_8000", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
