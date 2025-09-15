STT API (Whisper)

Overview
- Local transcription via whisper.cpp binaries and GGML models.
- Models: tiny|base|small|medium|large-v2|large-v3
- Endpoints support non-streaming and streaming (SSE) plus WebSocket.

REST Endpoints
- POST `/v1/audio/transcriptions?model=base`
  - multipart form-data
  - Fields: `file` or `audio` = audio file
  - Response: `{ "text": "...", "model": "base" }`

- POST `/v1/audio/transcriptions/stream?model=base`
  - multipart form-data
  - Response: `text/event-stream`
    - Emits: `data: <line>` events as text is produced
    - Terminates with: `event: done` + `data: `

WebSocket
- `ws://<host>:<port>/<prefix>/stt`
  - Send (non-streamed): `{ "filename":"a.wav", "model":"base", "audio_base64":"<...>" }`
    - Receive: `{ "ok": true, "text": "...", "model": "base" }`
  - Send (streamed): `{ "filename":"a.wav", "model":"base", "audio_base64":"<...>", "stream": true }`
    - Receive events:
      - `{ "event": "status", "message": "starting transcription" }`
      - `{ "event": "data", "text": "..." }` repeated for partials
      - `{ "event": "done" }` when finished

Notes
- First run downloads the whisper binary and requested model.
- Audio formats supported by the bundled binaries are accepted; WAV/MP3/M4A common.
- Model defaults can be set in config; query param overrides per request.

