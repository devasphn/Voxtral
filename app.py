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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Voxtral imports
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import torchaudio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voxtral Audio Chat", version="1.0.0")

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
            device_map=device
        )
        
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

async def process_audio(audio_data: bytes, client_id: str, mode: str = "chat"):
    """Process audio data and return response"""
    try:
        # Convert audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(temp_file_path)
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed (Voxtral expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Prepare conversation for the model
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio": waveform.numpy().flatten()
                }]
            }]
            
            # Process with model
            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.2, top_p=0.95)
                decoded_outputs = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            
            response_text = decoded_outputs[0] if decoded_outputs else "Sorry, I couldn't process that audio."
            
            await manager.send_message(client_id, {
                "type": "response",
                "text": response_text,
                "status": "success"
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        await manager.send_message(client_id, {
            "type": "error", 
            "message": f"Error processing audio: {str(e)}"
        })

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = await load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "audio":
                # Decode base64 audio
                audio_bytes = base64.b64decode(data["audio"])
                
                # Send processing status
                await manager.send_message(client_id, {
                    "type": "status", 
                    "message": "Processing audio..."
                })
                
                # Process audio in background
                asyncio.create_task(process_audio(audio_bytes, client_id))
                
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
                    <span id="status" class="status">Ready</span>
                </div>
                
                <div class="chat-container">
                    <div id="messages" class="messages"></div>
                </div>
                
                <div class="info">
                    <p><strong>Supported Languages:</strong> English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian</p>
                    <p><strong>Instructions:</strong> Click "Start Recording" to begin. Speak clearly and click "Stop Recording" when done.</p>
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
