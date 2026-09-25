/**
 * VoiceClone AI Studio - Visualizer Module
 * Handles Radar Chart rendering and Live Audio Spectrum / VU Meter.
 */

export class AcousticRadarChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.labels = ['Pitch', 'Brightness', 'Resonance', 'Warmth', 'Articulation', 'Stability'];
    this.defaultValues = [50, 50, 50, 50, 50, 50];
  }

  draw(data = null) {
    if (!this.canvas) return;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 34;

    ctx.clearRect(0, 0, width, height);

    const numAxes = this.labels.length;
    const angleStep = (Math.PI * 2) / numAxes;

    // Draw concentric polygon grid rings (20%, 40%, 60%, 80%, 100%)
    const rings = [0.2, 0.4, 0.6, 0.8, 1.0];
    rings.forEach((rRatio) => {
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * (radius * rRatio);
        const y = centerY + Math.sin(angle) * (radius * rRatio);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = rRatio === 1.0 ? 'rgba(255, 255, 255, 0.15)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw axis lines and labels
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numAxes; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      // Axis line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.stroke();

      // Label text
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 18);
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(this.labels[i], labelX, labelY);
    }

    // Prepare data points (0 - 100 scale)
    const values = this.labels.map(l => (data && data[l] !== undefined ? data[l] : 50));

    // Draw data polygon with glowing gradient fill
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const valRatio = Math.max(0.1, Math.min(1.0, values[i] / 100));
      const x = centerX + Math.cos(angle) * (radius * valRatio);
      const y = centerY + Math.sin(angle) * (radius * valRatio);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();

    // Fill gradient
    const grad = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, radius);
    grad.addColorStop(0, 'rgba(6, 182, 212, 0.55)');
    grad.addColorStop(1, 'rgba(139, 92, 246, 0.25)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Outline stroke with cyan glow
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

    // Draw point dots
    for (let i = 0; i < numAxes; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const valRatio = Math.max(0.1, Math.min(1.0, values[i] / 100));
      const x = centerX + Math.cos(angle) * (radius * valRatio);
      const y = centerY + Math.sin(angle) * (radius * valRatio);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#67e8f9';
      ctx.fill();
      ctx.strokeStyle = '#0e131f';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

export class LiveAudioVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.animationId = null;
    this.analyser = null;
    this.dataArray = null;
  }

  attach(analyserNode) {
    this.analyser = analyserNode;
    this.analyser.fftSize = 64;
    const bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(bufferLength);
    this.start();
  }

  start() {
    if (!this.canvas || !this.analyser) return;
    const render = () => {
      this.animationId = requestAnimationFrame(render);
      this.analyser.getByteFrequencyData(this.dataArray);

      const ctx = this.ctx;
      const width = this.canvas.width;
      const height = this.canvas.height;
      ctx.clearRect(0, 0, width, height);

      const barWidth = (width / this.dataArray.length) * 1.5;
      let x = 0;

      for (let i = 0; i < this.dataArray.length; i++) {
        const barHeight = (this.dataArray[i] / 255) * height;
        const grad = ctx.createLinearGradient(0, height, 0, 0);
        grad.addColorStop(0, '#06b6d4');
        grad.addColorStop(1, '#f43f5e');

        ctx.fillStyle = grad;
        ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    };
    render();
  }

  stop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }
}
