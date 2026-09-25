/**
 * VoiceClone AI Studio - Audio Player Module
 * Waveform canvas rendering, scrubbing, playhead, and multi-track audio playback.
 */

export class WaveformAudioPlayer {
  constructor(canvasId, { onTimeUpdate, onEnded, onPlayStateChange } = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.audio = new Audio();
    this.audioBuffer = null;
    this.peaks = [];
    this.isPlaying = false;
    this.duration = 0;
    this.currentTime = 0;
    this.animationId = null;

    this.onTimeUpdate = onTimeUpdate || (() => {});
    this.onEnded = onEnded || (() => {});
    this.onPlayStateChange = onPlayStateChange || (() => {});

    this.activeTrackUrl = null;
    this.refTrackUrl = null;
    this.currentTrackType = 'output'; // 'output' or 'reference'

    this._bindEvents();
  }

  _bindEvents() {
    this.audio.addEventListener('timeupdate', () => {
      this.currentTime = this.audio.currentTime;
      this.onTimeUpdate(this.currentTime, this.duration);
    });

    this.audio.addEventListener('loadedmetadata', () => {
      this.duration = this.audio.duration;
      this.onTimeUpdate(0, this.duration);
      this.draw();
    });

    this.audio.addEventListener('ended', () => {
      this.isPlaying = false;
      this.onPlayStateChange(false);
      this.onEnded();
      this.draw();
    });

    this.audio.addEventListener('play', () => {
      this.isPlaying = true;
      this.onPlayStateChange(true);
      this._startRenderLoop();
    });

    this.audio.addEventListener('pause', () => {
      this.isPlaying = false;
      this.onPlayStateChange(false);
      this._stopRenderLoop();
      this.draw();
    });

    // Canvas click to scrub
    if (this.canvas) {
      this.canvas.addEventListener('click', (e) => {
        if (!this.duration) return;
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, clickX / rect.width));
        this.seek(ratio * this.duration);
      });
    }
  }

  async loadTrack(url, trackType = 'output') {
    this.currentTrackType = trackType;
    if (trackType === 'output') {
      this.activeTrackUrl = url;
    } else {
      this.refTrackUrl = url;
    }

    this.audio.src = url;
    this.audio.load();

    // Fetch and decode audio to get waveform peaks
    try {
      const resp = await fetch(url);
      const arrayBuffer = await resp.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      this.audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      this._extractPeaks();
      ctx.close();
    } catch (err) {
      console.warn('Could not decode audio buffer for waveform:', err);
      // Generate synthetic peaks if decoding fails
      this.peaks = Array.from({ length: 120 }, () => Math.random() * 0.7 + 0.2);
    }

    this.draw();
  }

  _extractPeaks(numBars = 120) {
    if (!this.audioBuffer) return;
    const channelData = this.audioBuffer.getChannelData(0);
    const blockSize = Math.floor(channelData.length / numBars);
    this.peaks = [];

    for (let i = 0; i < numBars; i++) {
      const start = i * blockSize;
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(channelData[start + j] || 0);
      }
      this.peaks.push(sum / blockSize);
    }

    // Normalize peaks
    const max = Math.max(...this.peaks, 1e-4);
    this.peaks = this.peaks.map((p) => Math.max(0.1, p / max));
  }

  togglePlay() {
    if (!this.audio.src) return;
    if (this.isPlaying) {
      this.audio.pause();
    } else {
      this.audio.play();
    }
  }

  seek(timeInSeconds) {
    this.currentTime = timeInSeconds;
    this.audio.currentTime = timeInSeconds;
    this.draw();
  }

  setPlaybackRate(rate) {
    this.audio.playbackRate = rate;
  }

  setVolume(vol) {
    this.audio.volume = Math.max(0, Math.min(1, vol));
  }

  _startRenderLoop() {
    const loop = () => {
      if (this.isPlaying) {
        this.draw();
        this.animationId = requestAnimationFrame(loop);
      }
    };
    loop();
  }

  _stopRenderLoop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  draw() {
    if (!this.canvas) return;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();

    if (this.canvas.width !== rect.width * dpr || this.canvas.height !== rect.height * dpr) {
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const peaks = this.peaks.length > 0 ? this.peaks : Array.from({ length: 100 }, () => 0.2);
    const numBars = peaks.length;
    const barWidth = Math.max(2, (width / numBars) * 0.65);
    const gap = (width - numBars * barWidth) / (numBars - 1);
    const progressRatio = this.duration > 0 ? this.currentTime / this.duration : 0;
    const progressX = progressRatio * width;

    // Draw baseline
    const centerY = height / 2;

    for (let i = 0; i < numBars; i++) {
      const x = i * (barWidth + gap);
      const barHeight = Math.max(4, peaks[i] * (height * 0.8));
      const isPlayed = x <= progressX;

      // Color scheme based on track type
      let fill;
      if (isPlayed) {
        fill = this.currentTrackType === 'reference' ? '#8b5cf6' : '#06b6d4';
      } else {
        fill = 'rgba(255, 255, 255, 0.12)';
      }

      ctx.fillStyle = fill;
      // Draw symmetrical mirrored rounded bar
      const topY = centerY - barHeight / 2;
      ctx.beginPath();
      ctx.roundRect(x, topY, barWidth, barHeight, 2);
      ctx.fill();
    }

    // Draw scrubber playhead cursor line
    if (this.duration > 0) {
      ctx.strokeStyle = this.currentTrackType === 'reference' ? '#c4b5fd' : '#67e8f9';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 0);
      ctx.lineTo(progressX, height);
      ctx.stroke();

      // Scrubber head circle
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(progressX, centerY, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
}
