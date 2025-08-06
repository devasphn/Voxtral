class VoxtralChat {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isContinuousMode = false;
        this.clientId = this.generateClientId();
        this.recordingStartTime = null;
        this.silenceTimer = null;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        
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
        this.continuousBtn = document.getElementById('continuousBtn');
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
        this.continuousBtn.addEventListener('click', () => this.toggleContinuousMode());
    }
    
    async toggleContinuousMode() {
        if (!this.isContinuousMode) {
            // Start continuous mode
            this.isContinuousMode = true;
            this.continuousBtn.textContent = '⏹️ Stop Continuous';
            this.continuousBtn.classList.add('active');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = true;
            
            await this.startContinuousRecording();
        } else {
            // Stop continuous mode
            this.isContinuousMode = false;
            this.continuousBtn.textContent = '🔄 Continuous Mode';
            this.continuousBtn.classList.remove('active');
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            
            this.stopContinuousRecording();
        }
    }
    
    async startContinuousRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // Setup audio context for voice activity detection
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            const microphone = this.audioContext.createMediaStreamSource(stream);
            microphone.connect(this.analyser);
            
            this.analyser.fftSize = 256;
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            this.updateStatus('Continuous mode active - Speak anytime', 'recording');
            this.startVoiceActivityDetection(stream);
            
        } catch (error) {
            console.error('Error starting continuous recording:', error);
            this.addMessage('Error: Could not access microphone for continuous mode.', 'error');
            this.isContinuousMode = false;
            this.toggleContinuousMode();
        }
    }
    
    startVoiceActivityDetection(stream) {
        const checkVoiceActivity = () => {
            if (!this.isContinuousMode) return;
            
            this.analyser.getByteFrequencyData(this.dataArray);
            const average = this.dataArray.reduce((a, b) => a + b) / this.dataArray.length;
            
            // Voice activity threshold (adjust as needed)
            const threshold = 20;
            
            if (average > threshold && !this.isRecording) {
                // Voice detected, start recording
                this.startAutoRecording(stream);
            }
            
            requestAnimationFrame(checkVoiceActivity);
        };
        
        checkVoiceActivity();
    }
    
    async startAutoRecording(stream) {
        if (this.isRecording) return;
        
        try {
            // Check if the browser supports the preferred format
            let options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'audio/webm' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options = { mimeType: 'audio/wav' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        options = {}; // Let the browser choose
                    }
                }
            }
            
            this.mediaRecorder = new MediaRecorder(stream, options);
            this.audioChunks = [];
            this.recordingStartTime = Date.now();
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            this.mediaRecorder.start(100);
            this.isRecording = true;
            
            this.updateStatus('Recording detected speech...', 'recording');
            
            // Auto-stop after silence or max duration
            this.silenceTimer = setTimeout(() => {
                if (this.isRecording) {
                    this.stopAutoRecording();
                }
            }, 5000); // Stop after 5 seconds
            
        } catch (error) {
            console.error('Error starting auto recording:', error);
        }
    }
    
    stopAutoRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            if (this.silenceTimer) {
                clearTimeout(this.silenceTimer);
                this.silenceTimer = null;
            }
            
            this.updateStatus('Processing detected speech...', 'processing');
        }
    }
    
    stopContinuousRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        this.updateStatus('Ready');
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // Check if the browser supports the preferred format
            let options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'audio/webm' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options = { mimeType: 'audio/wav' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        options = {}; // Let the browser choose
                    }
                }
            }
            
            this.mediaRecorder = new MediaRecorder(stream, options);
            this.audioChunks = [];
            this.recordingStartTime = Date.now();
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    console.log('Audio chunk received:', event.data.size, 'bytes');
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Start recording with data intervals
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Recording... (speak now)', 'recording');
            
            // Auto-stop after 30 seconds to prevent too long recordings
            setTimeout(() => {
                if (this.isRecording && !this.isContinuousMode) {
                    this.stopRecording();
                }
            }, 30000);
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.addMessage('Error: Could not access microphone. Please allow microphone access and try again.', 'error');
            this.updateStatus('Error accessing microphone');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            const recordingDuration = Date.now() - this.recordingStartTime;
            
            // Check if recording is too short
            if (recordingDuration < 500) {
                this.addMessage('Recording too short. Please record for at least 1 second.', 'error');
                this.mediaRecorder.stop();
                this.isRecording = false;
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                this.updateStatus('Ready');
                return;
            }
            
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.updateStatus('Processing recording...', 'processing');
        }
    }
    
    async processRecording() {
        console.log('Processing recording with', this.audioChunks.length, 'chunks');
        
        if (this.audioChunks.length === 0) {
            this.addMessage('No audio data recorded. Please try again.', 'error');
            this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
            return;
        }
        
        // Create blob from recorded chunks
        const audioBlob = new Blob(this.audioChunks, { 
            type: this.mediaRecorder.mimeType || 'audio/webm' 
        });
        
        console.log('Audio blob created:', audioBlob.size, 'bytes, type:', audioBlob.type);
        
        if (audioBlob.size < 1000) {
            this.addMessage('Recording too small. Please speak louder and longer.', 'error');
            this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
            return;
        }
        
        // Convert to base64
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const base64Audio = reader.result.split(',')[1];
                console.log('Base64 audio length:', base64Audio.length);
                this.sendAudio(base64Audio);
            } catch (error) {
                console.error('Error processing audio blob:', error);
                this.addMessage('Error processing audio. Please try again.', 'error');
                this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
            }
        };
        
        reader.onerror = () => {
            console.error('Error reading audio blob');
            this.addMessage('Error reading audio. Please try again.', 'error');
            this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
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
                this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
                break;
                
            case 'status':
                this.updateStatus(data.message, 'processing');
                break;
                
            case 'error':
                this.addMessage(`Error: ${data.message}`, 'error');
                this.updateStatus(this.isContinuousMode ? 'Continuous mode active - Speak anytime' : 'Ready');
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
