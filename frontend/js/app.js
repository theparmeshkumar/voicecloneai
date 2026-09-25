import { AcousticRadarChart, LiveAudioVisualizer } from './visualizer.js';
import { AudioRecorder } from './audio_recorder.js';
import { WaveformAudioPlayer } from './audio_player.js';

// Application State
const state = {
  currentStage: 'stage-clone', // 'stage-clone' or 'stage-studio'
  activeClone: null,           // Current loaded voice profile
  selectedSampleFile: null,    // Uploaded audio blob/file for cloning
  stsRecordedBlob: null,       // Audio blob for Speech-to-Speech
  allProfiles: [],
  history: [],
};

// Preset Scripts
const SCRIPT_PRESETS = {
  podcast: "Welcome back to the podcast. Today, we're diving deep into the fascinating world of artificial intelligence and voice synthesis technology.",
  keynote: "Good morning everyone. Today, I am thrilled to introduce a groundbreaking breakthrough that will completely reshape how we communicate with machines.",
  story: "The old library was quiet, save for the ticking of a grandfather clock in the corner. She turned the yellowed page, uncovering a secret lost for centuries.",
  documentary: "High above the Arctic tundra, the wind howls across vast plains of ice. Here, in one of the most unforgiving environments on Earth, life finds a way.",
  assistant: "I've reviewed your schedule for today. You have three meetings this morning, and the weather forecast is mostly sunny with a light breeze."
};

// Toast Notification Helper
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ'}</span> ${message}`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Format Seconds to MM:SS
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Visualizers & Players
  const radarChart = new AcousticRadarChart('acousticRadarCanvas');
  radarChart.draw(); // initial neutral radar

  const micVisualizer = new LiveAudioVisualizer('micVisualizerCanvas');
  const stsVisualizer = new LiveAudioVisualizer('stsMicVisualizerCanvas');

  // Master Waveform Player
  const masterPlayer = new WaveformAudioPlayer('masterWaveformCanvas', {
    onTimeUpdate: (cur, dur) => {
      const curEl = document.getElementById('playerTimeCurrent');
      const durEl = document.getElementById('playerTimeDuration');
      if (curEl) curEl.textContent = formatTime(cur);
      if (durEl) durEl.textContent = formatTime(dur);
    },
    onPlayStateChange: (isPlaying) => {
      const playBtn = document.getElementById('btnMasterPlay');
      if (playBtn) {
        playBtn.innerHTML = isPlaying
          ? `<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`
          : `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
      }
    }
  });

  // Reference Audio Player (in Step 1 profile card)
  const refPlayer = new WaveformAudioPlayer('refWaveformCanvas', {
    onTimeUpdate: (cur, dur) => {
      const curEl = document.getElementById('refTimeCurrent');
      const durEl = document.getElementById('refTimeDuration');
      if (curEl) curEl.textContent = formatTime(cur);
      if (durEl) durEl.textContent = formatTime(dur);
    },
    onPlayStateChange: (isPlaying) => {
      const playBtn = document.getElementById('btnRefPlay');
      if (playBtn) {
        playBtn.innerHTML = isPlaying ? '❚❚' : '▶';
      }
    }
  });

  // Microphones
  const sampleRecorder = new AudioRecorder({
    analyserTarget: micVisualizer,
    onStart: () => {
      document.getElementById('btnRecordSample').classList.add('recording');
      document.getElementById('btnRecordSample').innerHTML = '<span>■</span> Stop Recording';
      document.getElementById('micVisualizerCanvas').style.display = 'block';
    },
    onStop: (blob, duration) => {
      document.getElementById('btnRecordSample').classList.remove('recording');
      document.getElementById('btnRecordSample').innerHTML = '<span>🎙</span> Record via Microphone';
      document.getElementById('micVisualizerCanvas').style.display = 'none';

      state.selectedSampleFile = new File([blob], `mic_sample_${Date.now()}.webm`, { type: blob.type });
      showSelectedSamplePreview(state.selectedSampleFile, duration);
      showToast(`Recorded ${formatTime(duration)} voice sample`, 'success');
    },
    onTick: (formatted) => {
      document.getElementById('sampleRecTimer').textContent = formatted;
    }
  });

  const stsRecorder = new AudioRecorder({
    analyserTarget: stsVisualizer,
    onStart: () => {
      document.getElementById('btnStsRecord').classList.add('recording');
      document.getElementById('btnStsRecord').innerHTML = '<span>■</span> Stop Recording';
      document.getElementById('stsMicVisualizerCanvas').style.display = 'block';
    },
    onStop: (blob, duration) => {
      document.getElementById('btnStsRecord').classList.remove('recording');
      document.getElementById('btnStsRecord').innerHTML = '<span>🎙</span> Record Voice Script';
      document.getElementById('stsMicVisualizerCanvas').style.display = 'none';

      state.stsRecordedBlob = blob;
      document.getElementById('stsAudioPreview').style.display = 'block';
      document.getElementById('stsAudioTag').src = URL.createObjectURL(blob);
      showToast(`Recorded ${formatTime(duration)} audio script`, 'success');

      // Auto-trigger whisper transcription
      autoTranscribeSts(blob);
    },
    onTick: (formatted) => {
      document.getElementById('stsRecTimer').textContent = formatted;
    }
  });

  // Bind UI Controls
  setupNavigation();
  setupSampleUpload(sampleRecorder);
  setupCloneAction(radarChart, refPlayer);
  setupStudioControls(masterPlayer, stsRecorder);
  setupModals();
  loadSettings();
  loadProfiles();
  loadHistory();

  // Navigation Logic
  function setupNavigation() {
    const tabClone = document.getElementById('tabCloneVoice');
    const tabStudio = document.getElementById('tabStudioGen');
    const panelClone = document.getElementById('panelCloneVoice');
    const panelStudio = document.getElementById('panelStudioGen');

    tabClone.addEventListener('click', () => {
      tabClone.classList.add('active');
      tabStudio.classList.remove('active');
      panelClone.style.display = 'grid';
      panelStudio.style.display = 'none';
      state.currentStage = 'stage-clone';
    });

    tabStudio.addEventListener('click', () => {
      if (!state.activeClone) {
        showToast('Please first analyze and clone a voice, or select one from the library!', 'error');
        return;
      }
      tabStudio.classList.add('active');
      tabClone.classList.remove('active');
      panelStudio.style.display = 'grid';
      panelClone.style.display = 'none';
      state.currentStage = 'stage-studio';
      updateActiveCloneBanner();
    });
  }

  // Sample Upload (Dropzone + File Input + Mic)
  function setupSampleUpload(sampleRec) {
    const dropzone = document.getElementById('sampleDropzone');
    const fileInput = document.getElementById('sampleFileInput');
    const btnRecord = document.getElementById('btnRecordSample');

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-active');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('drag-active');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-active');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    btnRecord.addEventListener('click', () => {
      if (sampleRec.isRecording) {
        sampleRec.stop();
      } else {
        sampleRec.start();
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.startsWith('audio/') && !file.name.match(/\.(wav|mp3|m4a|ogg|webm|flac)$/i)) {
      showToast('Please select a valid audio file (WAV, MP3, M4A, OGG, WebM)', 'error');
      return;
    }
    state.selectedSampleFile = file;
    showSelectedSamplePreview(file);
    showToast(`Loaded "${file.name}"`, 'success');
  }

  function showSelectedSamplePreview(file, knownDuration = null) {
    const previewCard = document.getElementById('samplePreviewCard');
    const fileNameEl = document.getElementById('samplePreviewFileName');
    const fileSizeEl = document.getElementById('samplePreviewSize');
    const audioEl = document.getElementById('samplePreviewAudio');
    const btnAnalyze = document.getElementById('btnAnalyzeClone');

    previewCard.style.display = 'flex';
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    audioEl.src = URL.createObjectURL(file);
    btnAnalyze.disabled = false;

    // Suggest default clone name from file
    const cloneNameInput = document.getElementById('cloneNameInput');
    if (!cloneNameInput.value) {
      const baseName = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
      cloneNameInput.value = baseName.charAt(0).toUpperCase() + baseName.slice(1);
    }
  }

  // Clone Voice Action (Upload -> Feature Extraction -> Profile)
  function setupCloneAction(radar, refPlayerInstance) {
    const btnAnalyze = document.getElementById('btnAnalyzeClone');
    const scanOverlay = document.getElementById('scanOverlay');
    const profileCard = document.getElementById('profileResultCard');

    btnAnalyze.addEventListener('click', async () => {
      if (!state.selectedSampleFile) {
        showToast('Please upload or record an audio sample first!', 'error');
        return;
      }

      const cloneName = document.getElementById('cloneNameInput').value.trim();
      const cloneDesc = document.getElementById('cloneDescInput').value.trim();
      const cloneGender = document.getElementById('cloneGenderSelect').value;
      const cloneAccent = document.getElementById('cloneAccentSelect').value;

      const formData = new FormData();
      formData.append('audio_file', state.selectedSampleFile);
      formData.append('name', cloneName);
      formData.append('description', cloneDesc);
      if (cloneGender !== 'auto') formData.append('gender', cloneGender);
      if (cloneAccent !== 'auto') formData.append('accent', cloneAccent);

      scanOverlay.style.display = 'flex';
      btnAnalyze.disabled = true;

      try {
        const resp = await fetch('/api/clone/upload', {
          method: 'POST',
          body: formData
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || 'Voice analysis failed');
        }

        const cloneData = await resp.json();
        state.activeClone = cloneData;
        showToast(`Voice "${cloneData.name}" successfully cloned!`, 'success');

        // Populate Profile UI
        displayCloneProfile(cloneData, radar, refPlayerInstance);
        profileCard.style.display = 'flex';

        // Unlock Step 2 Tab and switch to it after a brief moment
        document.getElementById('tabStudioGen').classList.remove('disabled');
        loadProfiles();

        // Switch to Studio Generation tab
        setTimeout(() => {
          document.getElementById('tabStudioGen').click();
        }, 1200);

      } catch (err) {
        console.error('Clone creation error:', err);
        showToast(`Error: ${err.message}`, 'error');
      } finally {
        scanOverlay.style.display = 'none';
        btnAnalyze.disabled = false;
      }
    });
  }

  function displayCloneProfile(cloneData, radar, refPlayerInstance) {
    const profile = cloneData.profile;
    document.getElementById('profileHeroName').textContent = cloneData.name;
    document.getElementById('profileHeroTag').textContent = profile.timbre.tone_description;

    // Display traits badges
    const traitsRow = document.getElementById('profileTraitsRow');
    const badgeGender = document.getElementById('badgeGender');
    const badgeAccent = document.getElementById('badgeAccent');
    if (traitsRow && badgeGender && badgeAccent) {
      traitsRow.style.display = 'flex';
      badgeGender.textContent = cloneData.gender || profile.pitch.inferred_gender || 'Voice';
      badgeAccent.textContent = cloneData.accent || 'Regional Voice';
    }

    // Metrics
    document.getElementById('metricRegister').textContent = profile.pitch.vocal_register;
    document.getElementById('metricGender').textContent = cloneData.gender ? `${cloneData.gender} (${profile.pitch.inferred_gender})` : profile.pitch.inferred_gender;
    document.getElementById('metricMeanF0').textContent = `${profile.pitch.median_f0_hz || profile.pitch.mean_f0_hz} Hz`;
    document.getElementById('metricFormants').textContent = `F1: ${profile.formants.f1_hz} | F2: ${profile.formants.f2_hz}`;
    document.getElementById('metricBrightness').textContent = `${profile.timbre.spectral_centroid_hz} Hz`;
    document.getElementById('metricPacing').textContent = `${profile.pacing.estimated_tempo_bpm} BPM (${profile.pacing.words_per_minute} wpm)`;

    // Draw Radar Chart
    radar.draw(profile.radar);

    // Load Reference Track into Ref Player
    if (cloneData.ref_audio_file) {
      const refUrl = `/api/audio/profile/${cloneData.ref_audio_file}`;
      refPlayerInstance.loadTrack(refUrl, 'reference');
      document.getElementById('btnRefPlay').onclick = () => refPlayerInstance.togglePlay();
    }
  }

  function updateActiveCloneBanner() {
    if (!state.activeClone) return;
    const bannerName = document.getElementById('activeCloneName');
    const bannerTag = document.getElementById('activeCloneTag');
    if (bannerName) bannerName.textContent = state.activeClone.name;
    if (bannerTag) {
      const accent = state.activeClone.accent || 'Natural Voice';
      const gender = state.activeClone.gender || state.activeClone.profile.pitch.inferred_gender;
      bannerTag.textContent = `${gender} • ${accent} • ${state.activeClone.profile.timbre.tone_description}`;
    }
  }

  // Setup Studio Generation Controls (TTS & STS)
  function setupStudioControls(masterPl, stsRec) {
    const modeTabs = document.querySelectorAll('.mode-tab');
    const ttsArea = document.getElementById('ttsControlsArea');
    const stsArea = document.getElementById('stsControlsArea');

    let currentMode = 'tts';

    modeTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        modeTabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        currentMode = tab.dataset.mode;
        if (currentMode === 'tts') {
          ttsArea.style.display = 'block';
          stsArea.style.display = 'none';
        } else {
          ttsArea.style.display = 'none';
          stsArea.style.display = 'block';
        }
      });
    });

    // Script Presets
    const presetChips = document.querySelectorAll('.preset-chip');
    const scriptInput = document.getElementById('scriptTextInput');
    presetChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const key = chip.dataset.preset;
        if (SCRIPT_PRESETS[key]) {
          scriptInput.value = SCRIPT_PRESETS[key];
          updateScriptCounter();
        }
      });
    });

    scriptInput.addEventListener('input', updateScriptCounter);

    function updateScriptCounter() {
      const words = scriptInput.value.trim().split(/\s+/).filter(Boolean).length;
      document.getElementById('scriptWordCount').textContent = `${words} words`;
      const estSecs = Math.round((words / 130) * 60);
      document.getElementById('scriptEstDuration').textContent = `~${estSecs}s speaking`;
    }

    // Sliders & Engine Selection
    const speedSlider = document.getElementById('speedSlider');
    const speedVal = document.getElementById('speedValue');
    speedSlider.addEventListener('input', () => {
      speedVal.textContent = `${parseFloat(speedSlider.value).toFixed(2)}x`;
    });

    const pitchSlider = document.getElementById('pitchSlider');
    const pitchVal = document.getElementById('pitchValue');
    if (pitchSlider && pitchVal) {
      pitchSlider.addEventListener('input', () => {
        const val = parseFloat(pitchSlider.value);
        pitchVal.textContent = `${val > 0 ? '+' : ''}${val.toFixed(1)} st`;
      });
    }

    const warmthSlider = document.getElementById('warmthSlider');
    const warmthVal = document.getElementById('warmthValue');
    warmthSlider.addEventListener('input', () => {
      warmthVal.textContent = `${Math.round(warmthSlider.value * 100)}%`;
    });

    const synthEngineSelect = document.getElementById('synthEngineSelect');
    const synthEngineBadge = document.getElementById('synthEngineBadge');
    if (synthEngineSelect && synthEngineBadge) {
      synthEngineSelect.addEventListener('change', () => {
        const selected = synthEngineSelect.value;
        if (selected === 'auto') synthEngineBadge.textContent = 'Auto-Optimal';
        else if (selected === 'edge_tts') synthEngineBadge.textContent = 'Edge-TTS Studio';
        else if (selected === 'kokoro') synthEngineBadge.textContent = 'Kokoro ONNX';
        else if (selected === 'elevenlabs') synthEngineBadge.textContent = 'ElevenLabs 1:1';
      });
    }

    // STS Recorder
    const btnStsRecord = document.getElementById('btnStsRecord');
    btnStsRecord.addEventListener('click', () => {
      if (stsRec.isRecording) {
        stsRec.stop();
      } else {
        stsRec.start();
      }
    });

    // STS File Upload
    const stsFileInput = document.getElementById('stsFileInput');
    stsFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        const file = e.target.files[0];
        state.stsRecordedBlob = file;
        document.getElementById('stsAudioPreview').style.display = 'block';
        document.getElementById('stsAudioTag').src = URL.createObjectURL(file);
        showToast(`Loaded audio file: ${file.name}`, 'success');
        autoTranscribeSts(file);
      }
    });

    // Generate Button Click
    const btnGenerate = document.getElementById('btnGenerateSpeech');
    btnGenerate.addEventListener('click', async () => {
      if (!state.activeClone) {
        showToast('Please select or create a cloned voice first!', 'error');
        return;
      }

      btnGenerate.disabled = true;
      btnGenerate.innerHTML = `<span class="scanner-ring" style="width:18px;height:18px;border-width:2px;"></span> Generating Speech...`;

      try {
        let result;
        const chosenEngine = synthEngineSelect ? synthEngineSelect.value : 'auto';
        const pitchShift = pitchSlider ? parseFloat(pitchSlider.value) : 0.0;

        if (currentMode === 'tts') {
          const text = scriptInput.value.trim();
          if (!text) {
            throw new Error('Please enter text script to synthesize!');
          }
          const resp = await fetch('/api/generate/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              clone_id: state.activeClone.id,
              text: text,
              engine: chosenEngine,
              speed: parseFloat(speedSlider.value),
              pitch_shift: pitchShift,
              warmth_mix: parseFloat(warmthSlider.value),
              lang: document.getElementById('languageSelect').value
            })
          });

          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'TTS generation failed');
          }
          result = await resp.json();
        } else {
          // STS Mode
          if (!state.stsRecordedBlob) {
            throw new Error('Please record or upload audio to convert into cloned voice!');
          }
          const customScript = document.getElementById('stsTranscriptInput').value.trim();

          const formData = new FormData();
          formData.append('clone_id', state.activeClone.id);
          formData.append('audio_file', state.stsRecordedBlob);
          formData.append('custom_script', customScript);
          formData.append('engine', chosenEngine);
          formData.append('speed', parseFloat(speedSlider.value));
          formData.append('pitch_shift', pitchShift);
          formData.append('warmth_mix', parseFloat(warmthSlider.value));

          const resp = await fetch('/api/generate/sts', {
            method: 'POST',
            body: formData
          });

          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Speech-to-Speech conversion failed');
          }
          result = await resp.json();
        }

        showToast(`Generated speech via ${result.engine || 'AI Engine'} in ${result.processing_time_seconds || result.total_pipeline_time_seconds}s!`, 'success');
        loadGeneratedIntoMasterPlayer(result, masterPl);
        loadHistory();

      } catch (err) {
        console.error('Generation error:', err);
        showToast(err.message, 'error');
      } finally {
        btnGenerate.disabled = false;
        btnGenerate.innerHTML = `<span>⚡</span> Synthesize Cloned Voice`;
      }
    });

    // Master Player Controls
    document.getElementById('btnMasterPlay').onclick = () => masterPl.togglePlay();
    document.getElementById('btnMasterReplay').onclick = () => { masterPl.seek(0); masterPl.audio.play(); };

    // Compare Track Switching
    const btnCompOutput = document.getElementById('btnCompareOutput');
    const btnCompRef = document.getElementById('btnCompareRef');

    btnCompOutput.addEventListener('click', () => {
      btnCompOutput.classList.add('active');
      btnCompRef.classList.remove('active');
      if (masterPl.activeTrackUrl) {
        masterPl.loadTrack(masterPl.activeTrackUrl, 'output');
        masterPl.audio.play();
      }
    });

    btnCompRef.addEventListener('click', () => {
      if (!state.activeClone || !state.activeClone.ref_audio_file) {
        showToast('No original reference sample available', 'error');
        return;
      }
      btnCompRef.classList.add('active');
      btnCompOutput.classList.remove('active');
      const refUrl = `/api/audio/profile/${state.activeClone.ref_audio_file}`;
      masterPl.loadTrack(refUrl, 'reference');
      masterPl.audio.play();
    });

    // Playback Speed Selector in Master Player
    document.getElementById('playerSpeedSelect').addEventListener('change', (e) => {
      masterPl.setPlaybackRate(parseFloat(e.target.value));
    });
  }

  function loadGeneratedIntoMasterPlayer(genResult, masterPl) {
    const card = document.getElementById('masterPlayerCard');
    card.style.display = 'block';

    const trackTitle = document.getElementById('playerTrackTitle');
    const trackSubtitle = document.getElementById('playerTrackSubtitle');
    const downloadBtn = document.getElementById('btnDownloadWav');

    trackTitle.textContent = `${genResult.clone_name} — Cloned Generation`;
    const engineText = genResult.engine ? `Engine: ${genResult.engine} • ` : '';
    trackSubtitle.textContent = `${engineText}Duration: ${genResult.duration_seconds}s • Generated in ${genResult.processing_time_seconds || genResult.total_pipeline_time_seconds}s`;

    downloadBtn.href = genResult.audio_url;
    downloadBtn.download = genResult.audio_filename;

    masterPl.loadTrack(genResult.audio_url, 'output');
    document.getElementById('btnCompareOutput').classList.add('active');
    document.getElementById('btnCompareRef').classList.remove('active');

    // Scroll to player smoothly
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function autoTranscribeSts(audioBlob) {
    const box = document.getElementById('stsTranscriptBox');
    const input = document.getElementById('stsTranscriptInput');
    const langBadge = document.getElementById('stsDetectedLang');

    box.style.display = 'block';
    input.value = 'Transcribing speech with Whisper...';

    const formData = new FormData();
    formData.append('audio_file', audioBlob);

    try {
      const resp = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      });
      if (resp.ok) {
        const data = await resp.json();
        input.value = data.text;
        langBadge.textContent = `Lang: ${data.language.toUpperCase()} (${Math.round(data.language_probability * 100)}%)`;
      } else {
        input.value = '';
      }
    } catch (err) {
      console.warn('Transcription failed:', err);
      input.value = '';
    }
  }

  // Modals & Voice Library
  function setupModals() {
    const libModal = document.getElementById('libraryModal');
    const histModal = document.getElementById('historyModal');
    const settingsModal = document.getElementById('settingsModal');
    const editTraitsModal = document.getElementById('editTraitsModal');

    // Library Modal
    document.getElementById('btnOpenLibrary').onclick = () => {
      libModal.classList.add('show');
      renderLibraryGrid();
    };
    document.getElementById('btnCloseLibrary').onclick = () => libModal.classList.remove('show');

    // History Modal
    document.getElementById('btnOpenHistory').onclick = () => {
      histModal.classList.add('show');
      renderHistoryList();
    };
    document.getElementById('btnCloseHistory').onclick = () => histModal.classList.remove('show');

    // Settings Modal
    const btnOpenSettings = document.getElementById('btnOpenSettings');
    const btnCloseSettings = document.getElementById('btnCloseSettings');
    const btnSaveSettings = document.getElementById('btnSaveSettings');
    if (btnOpenSettings && settingsModal) {
      btnOpenSettings.onclick = async () => {
        await loadSettings();
        settingsModal.classList.add('show');
      };
    }
    if (btnCloseSettings && settingsModal) {
      btnCloseSettings.onclick = () => settingsModal.classList.remove('show');
    }
    if (btnSaveSettings) {
      btnSaveSettings.onclick = async () => {
        const engine = document.getElementById('settingDefaultEngine').value;
        const elevenKey = document.getElementById('settingElevenKey').value.trim();
        try {
          const resp = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ default_engine: engine, elevenlabs_api_key: elevenKey })
          });
          if (resp.ok) {
            showToast('Settings saved successfully!', 'success');
            settingsModal.classList.remove('show');
            const synthSelect = document.getElementById('synthEngineSelect');
            if (synthSelect && engine !== 'auto') {
              synthSelect.value = engine;
              synthSelect.dispatchEvent(new Event('change'));
            }
          }
        } catch (err) {
          showToast('Failed to save settings: ' + err.message, 'error');
        }
      };
    }

    // Quick Edit Characteristics Modal
    const btnQuickEdit = document.getElementById('btnQuickEditTraits');
    const btnCloseEdit = document.getElementById('btnCloseEditTraits');
    const btnSaveTraits = document.getElementById('btnSaveProfileTraits');
    if (btnQuickEdit && editTraitsModal) {
      btnQuickEdit.onclick = () => {
        if (!state.activeClone) {
          showToast('No active cloned voice selected', 'error');
          return;
        }
        document.getElementById('editProfileNameInput').value = state.activeClone.name || '';
        document.getElementById('editProfileGenderSelect').value = state.activeClone.gender || state.activeClone.profile.pitch.inferred_gender || 'Male';
        document.getElementById('editProfileAccentSelect').value = state.activeClone.accent || 'Indian English / Hindi';
        editTraitsModal.classList.add('show');
      };
    }
    if (btnCloseEdit && editTraitsModal) {
      btnCloseEdit.onclick = () => editTraitsModal.classList.remove('show');
    }
    if (btnSaveTraits) {
      btnSaveTraits.onclick = async () => {
        if (!state.activeClone) return;
        const newName = document.getElementById('editProfileNameInput').value.trim();
        const newGender = document.getElementById('editProfileGenderSelect').value;
        const newAccent = document.getElementById('editProfileAccentSelect').value;

        btnSaveTraits.disabled = true;
        btnSaveTraits.textContent = 'Updating Voice...';
        try {
          const resp = await fetch(`/api/profiles/${state.activeClone.id}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName, gender: newGender, accent: newAccent })
          });
          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Failed to update profile');
          }
          const updated = await resp.json();
          state.activeClone = updated;
          displayCloneProfile(updated, radarChart, refPlayer);
          updateActiveCloneBanner();
          loadProfiles();
          editTraitsModal.classList.remove('show');
          showToast(`Updated "${updated.name}" traits (${newGender}, ${newAccent})`, 'success');
        } catch (err) {
          showToast('Error updating traits: ' + err.message, 'error');
        } finally {
          btnSaveTraits.disabled = false;
          btnSaveTraits.textContent = 'Update Voice Profile';
        }
      };
    }

    [libModal, histModal, settingsModal, editTraitsModal].filter(Boolean).forEach((m) => {
      m.addEventListener('click', (e) => {
        if (e.target === m) m.classList.remove('show');
      });
    });
  }

  async function loadSettings() {
    try {
      const resp = await fetch('/api/settings');
      if (resp.ok) {
        const data = await resp.json();
        const engSelect = document.getElementById('settingDefaultEngine');
        const keyInput = document.getElementById('settingElevenKey');
        if (engSelect && data.default_engine) engSelect.value = data.default_engine;
        if (keyInput && data.elevenlabs_api_key) keyInput.value = data.elevenlabs_api_key;
        if (data.default_engine && data.default_engine !== 'auto') {
          const synthSelect = document.getElementById('synthEngineSelect');
          if (synthSelect) {
            synthSelect.value = data.default_engine;
            synthSelect.dispatchEvent(new Event('change'));
          }
        }
      }
    } catch (err) {
      console.warn('Could not load settings:', err);
    }
  }

  async function loadProfiles() {
    try {
      const resp = await fetch('/api/profiles');
      if (resp.ok) {
        state.allProfiles = await resp.json();
        // If no active clone yet, set first as active
        if (!state.activeClone && state.allProfiles.length > 0) {
          state.activeClone = state.allProfiles[0];
          document.getElementById('tabStudioGen').classList.remove('disabled');
          displayCloneProfile(state.activeClone, radarChart, refPlayer);
          updateActiveCloneBanner();
        }
      }
    } catch (err) {
      console.warn('Could not load profiles:', err);
    }
  }

  function renderLibraryGrid() {
    const grid = document.getElementById('libraryGrid');
    grid.innerHTML = '';

    if (state.allProfiles.length === 0) {
      grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">No cloned voices saved yet. Create your first clone in Step 1!</div>`;
      return;
    }

    state.allProfiles.forEach((p) => {
      const isSelected = state.activeClone && state.activeClone.id === p.id;
      const item = document.createElement('div');
      item.className = `library-item ${isSelected ? 'selected' : ''}`;
      item.innerHTML = `
        <div class="library-item-top">
          <div class="avatar-disc" style="width:40px; height:40px; font-size:1.1rem;">🎙</div>
          <div>
            <div style="font-weight:700; color:#fff;">${p.name}</div>
            <div class="library-item-meta">${p.profile.pitch.vocal_register} • ${p.sample_duration}s</div>
          </div>
        </div>
        <div style="font-size:0.82rem; color:var(--text-muted);">${p.description || p.profile.timbre.tone_description}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
          <button class="btn-action btn-action-primary" style="padding:0.4rem 0.8rem; font-size:0.78rem;">
            ${isSelected ? 'Active in Studio' : 'Select Voice'}
          </button>
          <button class="btn-delete" style="background:none; border:none; color:var(--accent-rose); cursor:pointer; font-size:0.8rem;" title="Delete Profile">✕</button>
        </div>
      `;

      item.querySelector('.btn-action-primary').onclick = (e) => {
        e.stopPropagation();
        state.activeClone = p;
        displayCloneProfile(p, radarChart, refPlayer);
        updateActiveCloneBanner();
        document.getElementById('libraryModal').classList.remove('show');
        showToast(`Activated "${p.name}" in Studio`, 'success');
      };

      item.querySelector('.btn-delete').onclick = async (e) => {
        e.stopPropagation();
        if (confirm(`Delete cloned voice "${p.name}"?`)) {
          await fetch(`/api/profiles/${p.id}`, { method: 'DELETE' });
          await loadProfiles();
          renderLibraryGrid();
          showToast(`Deleted "${p.name}"`, 'info');
        }
      };

      grid.appendChild(item);
    });
  }

  async function loadHistory() {
    try {
      const resp = await fetch('/api/history');
      if (resp.ok) {
        state.history = await resp.json();
      }
    } catch (err) {
      console.warn('Could not load history:', err);
    }
  }

  function renderHistoryList() {
    const list = document.getElementById('historyList');
    list.innerHTML = '';

    if (state.history.length === 0) {
      list.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No speech generations yet.</div>`;
      return;
    }

    state.history.forEach((h) => {
      const card = document.createElement('div');
      card.className = 'history-card';
      card.innerHTML = `
        <div>
          <div style="font-weight:600; color:#fff; font-size:0.95rem;">${h.clone_name} (${h.mode.toUpperCase()})</div>
          <div style="font-size:0.8rem; color:var(--text-muted); max-width:450px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${h.text || h.original_transcript || 'Speech output'}</div>
          <div style="font-size:0.72rem; color:var(--text-dim);">${h.timestamp || ''} • ${h.duration_seconds}s audio</div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <button class="btn-action btn-play-hist" style="padding:0.45rem 0.85rem; font-size:0.8rem;">▶ Play</button>
          <a href="${h.audio_url}" download="${h.audio_filename}" class="btn-action" style="padding:0.45rem 0.85rem; font-size:0.8rem;">⬇</a>
        </div>
      `;

      card.querySelector('.btn-play-hist').onclick = () => {
        loadGeneratedIntoMasterPlayer(h, masterPlayer);
        document.getElementById('historyModal').classList.remove('show');
        masterPlayer.audio.play();
      };

      list.appendChild(card);
    });
  }
});
