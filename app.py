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
            # Use ffmpeg to convert webm to wav with normalization
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', temp_webm_path, 
                '-ar', '16000', 
                '-ac', '1', 
                '-filter:a', 'volume=15dB,highpass=f=80,lowpass=f=8000',  # Enhanced filtering
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
            audio_array = waveform.squeeze().numpy().astype(np.float32)
            
            # Enhanced audio processing for Voxtral compatibility
            if len(audio_array) > 0:
                # Remove DC offset
                audio_array = audio_array - np.mean(audio_array)
                
                # Normalize to prevent clipping
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array = audio_array / max_val * 0.8  # Scale to 80% of max
                
                # Ensure minimum amplitude for Voxtral
                current_max = np.max(np.abs(audio_array))
                if current_max < 0.1:
                    # Apply significant gain boost
                    audio_array = audio_array * (0.3 / current_max)
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                
                # Add very slight noise to ensure non-zero signal
                noise = np.random.normal(0, 0.001, len(audio_array))
                audio_array = audio_array + noise
                audio_array = np.clip(audio_array, -1.0, 1.0)
            
            # Check if audio has sufficient content after processing
            final_amplitude = np.max(np.abs(audio_array))
            if len(audio_array) == 0 or final_amplitude < 0.01:
                await manager.send_message(client_id, {
                    "type": "error", 
                    "message": "Audio signal insufficient. Please record for longer and speak much louder."
                })
                return
            
            logger.info(f"Audio processed: {len(audio_array)} samples, final amplitude: {final_amplitude}")
            
            # Ensure minimum duration (Voxtral requires substantial audio)
            min_samples = 16000  # 1 second at 16kHz
            if len(audio_array) < min_samples:
                # Pad with silence
                padding = min_samples - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
                logger.info(f"Audio padded to minimum length: {len(audio_array)} samples")
            
            # CRITICAL: Convert to PyTorch tensor for Voxtral
            audio_tensor = torch.from_numpy(audio_array).float()
            
            # Create the conversation in the exact format Voxtral expects
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio_tensor  # Use tensor, not list
                        }
                    ]
                }
            ]
            
            logger.info("Applying chat template with tensor format...")
            
            try:
                # Apply chat template with tensor input
                inputs = processor.apply_chat_template(
                    conversation,
                    return_tensors="pt"
                )
                logger.info(f"Template applied successfully with tensor format")
                
            except Exception as e:
                logger.error(f"Chat template with tensor failed: {e}")
                
                # Try with numpy array format
                try:
                    conversation_numpy = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": audio_array  # Use numpy array
                                }
                            ]
                        }
                    ]
                    
                    inputs = processor.apply_chat_template(
                        conversation_numpy,
                        return_tensors="pt"
                    )
                    logger.info("Template applied with numpy format")
                    
                except Exception as e2:
                    logger.error(f"Both tensor and numpy formats failed: {e2}")
                    
                    # Final fallback: Direct processor call
                    try:
                        inputs = processor(
                            audio=audio_array,
                            return_tensors="pt",
                            sampling_rate=16000
                        )
                        logger.info("Used direct processor call")
                    except Exception as e3:
                        logger.error(f"All processing methods failed: {e3}")
                        raise e
            
            # Move inputs to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            logger.info("Generating response...")
            
            # Generate response
            with torch.no_grad():
                try:
                    if isinstance(inputs, dict) and "input_ids" in inputs:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=500,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                        input_length = inputs["input_ids"].shape[1]
                        generated_tokens = outputs[:, input_length:]
                    else:
                        # Handle case where inputs might be different format
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=500,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )
                        generated_tokens = outputs
                    
                    # Decode the response
                    response_text = processor.tokenizer.decode(
                        generated_tokens[0], 
                        skip_special_tokens=True
                    ).strip()
                    
                except Exception as gen_error:
                    logger.error(f"Generation error: {gen_error}")
                    response_text = "I processed your audio but encountered an error during response generation. Please try again."
                
            logger.info(f"Generated response: {response_text}")
            
            if not response_text:
                response_text = "I heard your audio but couldn't generate a meaningful response. Please try speaking more clearly and for a longer duration."
            
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
        import traceback
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "error", 
            "message": f"Error processing audio. Please try recording for longer and speaking louder."
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
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["type"] == "audio":
                if not data.get("audio"):
                    await manager.send_message(client_id, {
                        "type": "error", 
                        "message": "No audio data received"
                    })
                    continue
                
                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    
                    if len(audio_bytes) < 5000:  # Increased minimum size
                        await manager.send_message(client_id, {
                            "type": "error", 
                            "message": "Audio recording too short. Please record for at least 3-4 seconds and speak loudly."
                        })
                        continue
                    
                    logger.info(f"Received audio data: {len(audio_bytes)} bytes")
                    
                    await manager.send_message(client_id, {
                        "type": "status", 
                        "message": "Processing audio..."
                    })
                    
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
                    <p><strong>Recording Instructions:</strong> Record for 3-4 seconds minimum, speak VERY LOUDLY and clearly.</p>
                    <p><strong>Manual Mode:</strong> Click "Start Recording", speak loudly, then click "Stop Recording".</p>
                    <p><strong>Continuous Mode:</strong> Click "Continuous Mode" for hands-free operation.</p>
                    <p><strong>⚠️ Important:</strong> This model requires high-quality audio input. Speak much louder than normal conversation volume!</p>
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
    port = int(os.environ.get("RUNPOD_TCP_PORT_8000", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
