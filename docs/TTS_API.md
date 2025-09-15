TTS API (Piper)

Overview
- Local text-to-speech via Piper binaries.
- Voices fetched from rhasspy/piper-voices on Hugging Face.
- Default voice: `en_US-amy-medium` (configurable).

REST Endpoint
- POST `/v1/tts`
  - Request JSON:
    - `{ "text": "Hello there", "voice": "en_US-amy-medium" }`
  - Response body:
    - `audio/wav` bytes
  - Example:
    - `curl -X POST http://localhost:9000/v1/tts -H "Content-Type: application/json" -o out.wav -d '{"text":"Hello there","voice":"en_US-amy-medium"}'`

WebSocket
- `ws://<host>:<port>/<prefix>/tts`
  - Send: `{ "text": "Hello there", "voice": "en_US-amy-medium" }`
  - Receive: `{ "ok": true, "mime": "audio/wav", "audio_base64": "..." }`

Notes
- First request downloads Piper binary for the platform and the selected voice model (ONNX + JSON).
- Voices follow the path scheme: `<lang>/<locale>/<voice>/<quality>/<voice>.<ext>` — for example:
  - `en/en_US/amy/medium/en_US-amy-medium.onnx`
  - `en/en_US/amy/medium/en_US-amy-medium.onnx.json`
- To add more voices, use the correct name as published in rhasspy/piper-voices.

