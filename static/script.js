class VoxtralChat {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.clientId = this.generateClientId();
        
        this.initializeElements();
        this.initializeWebSocket();
        this.setupEventListeners();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.status = document.getElementById('status');
        this.messages = document.getElementById('messages');
    }
    
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('Connected - Ready to record');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('Disconnected - Refresh to reconnect');
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection error');
        };
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Recording...', 'recording');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.addMessage('Error: Could not access microphone. Please allow microphone access and try again.', 'error');
            this.updateStatus('Error accessing microphone');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.updateStatus('Processing...', 'processing');
        }
    }
    
    async processRecording() {
        if (this.audioChunks.length === 0) {
            this.addMessage('No audio recorded', 'error');
            this.updateStatus('Ready');
            return;
        }
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Convert to base64
        const reader = new FileReader();
        reader.onload = () => {
            const base64Audio = reader.result.split(',')[1];
            this.sendAudio(base64Audio);
        };
        reader.readAsDataURL(audioBlob);
    }
    
    sendAudio(base64Audio) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.addMessage('🎤 Audio sent for processing...', 'info');
            
            this.ws.send(JSON.stringify({
                type: 'audio',
                audio: base64Audio
            }));
        } else {
            this.addMessage('Error: Not connected to server', 'error');
            this.updateStatus('Connection error');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'response':
                this.addMessage(data.text, 'assistant');
                this.updateStatus('Ready');
                break;
                
            case 'status':
                this.updateStatus(data.message, 'processing');
                break;
                
            case 'error':
                this.addMessage(`Error: ${data.message}`, 'error');
                this.updateStatus('Ready');
                break;
                
            case 'pong':
                console.log('Received pong from server');
                break;
        }
    }
    
    addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = text;
        
        this.messages.appendChild(messageDiv);
        this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    updateStatus(text, className = '') {
        this.status.textContent = text;
        this.status.className = `status ${className}`;
    }
}

// Initialize the chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoxtralChat();
});
