/**
 * VoiceClone AI Studio - Audio Recorder Module
 * Handles browser microphone recording, MediaRecorder, live timer, and audio blob creation.
 */

export class AudioRecorder {
  constructor({ onStart, onStop, onTick, analyserTarget }) {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.audioContext = null;
    this.analyser = null;
    this.timerInterval = null;
    this.secondsRecorded = 0;
    this.isRecording = false;

    this.onStart = onStart || (() => {});
    this.onStop = onStop || (() => {});
    this.onTick = onTick || (() => {});
    this.analyserTarget = analyserTarget || null;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Web Audio setup for live visualization
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
      const source = this.audioContext.createMediaStreamSource(this.stream);
      this.analyser = this.audioContext.createAnalyser();
      source.connect(this.analyser);

      if (this.analyserTarget) {
        this.analyserTarget.attach(this.analyser);
      }

      // Determine supported mimeType
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
      let chosenMime = '';
      for (const m of mimeTypes) {
        if (MediaRecorder.isTypeSupported(m)) {
          chosenMime = m;
          break;
        }
      }

      this.mediaRecorder = new MediaRecorder(this.stream, chosenMime ? { mimeType: chosenMime } : {});
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: chosenMime || 'audio/webm' });
        this.onStop(audioBlob, this.secondsRecorded);
      };

      this.mediaRecorder.start(200); // 200ms slices
      this.isRecording = true;
      this.secondsRecorded = 0;

      // Timer
      this.timerInterval = setInterval(() => {
        this.secondsRecorded++;
        this.onTick(this.formatTime(this.secondsRecorded), this.secondsRecorded);
      }, 1000);

      this.onStart();
    } catch (err) {
      console.error('Microphone access failed:', err);
      alert('Could not access microphone: ' + err.message);
    }
  }

  stop() {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.analyserTarget) {
      this.analyserTarget.stop();
    }
  }

  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
}
