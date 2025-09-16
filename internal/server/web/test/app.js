(() => {
  const recordBtn = document.getElementById('recordBtn');
  const transcriptEl = document.getElementById('transcript');
  const logEl = document.getElementById('log');
  const embMetaEl = document.getElementById('embMeta');
  const embPreviewEl = document.getElementById('embPreview');
  const modelSelect = document.getElementById('modelSelect');
  const statusEl = document.getElementById('status');
  const embedBtn = document.getElementById('embedBtn');
  const embedText = document.getElementById('embedText');
  const ttsBtn = document.getElementById('ttsBtn');
  const ttsText = document.getElementById('ttsText');
  const ttsVoice = document.getElementById('ttsVoice');
  const ttsAudio = document.getElementById('ttsAudio');

  let recording = false;
  let audioCtx;
  let mediaStream;
  let processor;
  let input;
  let buffers = [];
  const sampleRate = 16000; // target 16kHz mono

  function log(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
    statusEl.textContent = msg;
  }

  recordBtn.addEventListener('click', async () => {
    if (!recording) {
      transcriptEl.textContent = '';
      embMetaEl.textContent = '';
      embPreviewEl.textContent = '';
      buffers = [];
      try {
        await startRecording();
        recordBtn.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
      } catch (e) {
        log('Failed to start recording: ' + e.message);
      }
    } else {
      await stopRecording();
      recordBtn.textContent = 'Start Recording';
      recordBtn.classList.remove('recording');
      log('Encoding audio (WAV)...');
      const wav = encodeWAV(buffers, sampleRate);
      const base64 = await blobToBase64(new Blob([wav], { type: 'audio/wav' }));
      log('Connecting to STT and streaming...');
      await runSTT(base64);
    }
  });

  embedBtn.addEventListener('click', async () => {
    let text = (embedText.value || '').trim();
    if (!text) {
      text = (transcriptEl.textContent || '').trim();
    }
    if (!text) {
      log('Nothing to embed — provide text or record speech first.');
      return;
    }
    log('Embedding provided text...');
    await fetchEmbeddings(text);
  });

  ttsBtn.addEventListener('click', async () => {
    let text = (ttsText.value || transcriptEl.textContent || '').trim();
    if (!text) { log('Nothing to speak — provide text or record speech first.'); return; }
    const voice = ttsVoice.value || 'en_US-amy-medium';
    log(`Requesting TTS with voice ${voice}...`);
    await speakWS(text, voice);
  });

  async function startRecording() {
    recording = true;
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
    const source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    input = source;
    input.connect(processor);
    processor.connect(audioCtx.destination);
    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const ch = e.inputBuffer.getChannelData(0);
      buffers.push(new Float32Array(ch));
    };
    log('Recording started');
  }

  async function stopRecording() {
    recording = false;
    try { processor && processor.disconnect(); } catch {}
    try { input && input.disconnect(); } catch {}
    try { mediaStream && mediaStream.getTracks().forEach(t => t.stop()); } catch {}
    try { audioCtx && audioCtx.close(); } catch {}
    log('Recording stopped');
  }

  function flattenBuffers(buffers) {
    let length = 0;
    for (const b of buffers) length += b.length;
    const out = new Float32Array(length);
    let offset = 0;
    for (const b of buffers) { out.set(b, offset); offset += b.length; }
    return out;
  }

  function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  function encodeWAV(bufs, sampleRate) {
    const samples = flattenBuffers(bufs);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return buffer;
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async function runSTT(audioB64) {
    transcriptEl.textContent = '';
    log('Connecting WS for STT...');
    const wsURL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/stt';
    const ws = new WebSocket(wsURL);
    ws.onopen = () => {
      log('WS connected, sending audio to transcribe...');
      ws.send(JSON.stringify({ filename: 'recording.wav', model: modelSelect.value, audio_base64: audioB64, stream: true }));
    };
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === 'status' && msg.message) {
          log(msg.message);
        } else if (msg.event === 'data') {
          if (transcriptEl.textContent.length === 0) log('Receiving transcription...');
          transcriptEl.textContent += msg.text + '\n';
        } else if (msg.event === 'done') {
          ws.close();
          const finalText = transcriptEl.textContent.trim();
          if (finalText) {
            log('Transcription complete. Fetching embeddings...');
            await fetchEmbeddings(finalText);
          } else {
            log('Transcription complete. No text.');
          }
        } else if (msg.ok && msg.text) {
          transcriptEl.textContent = msg.text;
          log('Transcription complete. Fetching embeddings...');
          await fetchEmbeddings(msg.text);
        } else if (msg.error) {
          log('STT error: ' + msg.error);
        }
      } catch (e) { log('WS parse error: ' + e.message); }
    };
    ws.onerror = (e) => log('WS error: ' + e.message);
  }

  async function fetchEmbeddings(text) {
    log('Requesting embeddings...');
    const resp = await fetch('/v1/embeddings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ input: text }) });
    if (!resp.ok) { log('Embeddings error: ' + resp.status); return; }
    const data = await resp.json();
    const vec = (data.embeddings && data.embeddings[0]) || [];
    embMetaEl.textContent = `Model: ${data.model} | Dim: ${vec.length}`;
    embPreviewEl.textContent = JSON.stringify(vec.slice(0, 16)) + (vec.length > 16 ? ' ...' : '');
    log('Embeddings received');
  }

  async function speakWS(text, voice) {
    const wsURL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/tts';
    const ws = new WebSocket(wsURL);
    ws.onopen = () => {
      ws.send(JSON.stringify({ text, voice }));
    };
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.ok && msg.audio_base64) {
          const b64 = msg.audio_base64;
          ttsAudio.src = 'data:' + (msg.mime || 'audio/wav') + ';base64,' + b64;
          ttsAudio.play();
          log('TTS audio received');
        } else if (msg.error) {
          log('TTS error: ' + msg.error);
        }
      } catch (e) { log('WS parse error: ' + e.message); }
    };
    ws.onerror = (e) => log('TTS WS error: ' + e.message);
  }
})();
