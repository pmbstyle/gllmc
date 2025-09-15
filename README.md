<p align="center">
<img width="250" height="250" alt="gllmc" src="https://github.com/user-attachments/assets/1d45be74-d6c8-46fb-803c-854fc83f8111" />
</p>

## Go LLM Core — Local, Modular LLM Backend

A single Go binary that runs local LLM APIs on Windows, macOS, and Linux.

Modular services enabled by flags featuring:
  - STT (Whisper)
  - TTS (Piper) 
  - Embeddings (all‑MiniLM‑L6‑v2)

### Prerequisites
- Go 1.21+
- Internet access on first run to download binaries/models

### Build
- From repo root:
  - `go build ./cmd/gollmcore`

Run
- With config file:
  - `./gollmcore --config configs/example-config.json`
  - Configure host/port, data_dir, service toggles, models/voices, WebSocket, and Test UI in JSON.
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 9000,
    "data_dir": ""
  },
  "services": {
    "stt": {
      "enabled": true,
      "model": "base"
    },
    "embeddings": {
      "enabled": true,
      "model": "all-MiniLM-L6-v2"
    },
    "tts": {
      "enabled": true,
      "voice": "en_US-amy-medium"
    }
  },
  "websocket": {
    "enabled": true,
    "path_prefix": "/ws"
  },
  "test_ui": {
    "enabled": true
  }
}
```

Ephemeral Port
- Use `--port 0` to bind an available ephemeral port; the server log prints the actual address, e.g., `HTTP server listening on 127.0.0.1:51243`.

Key Flag
- `--config` (string): Path to JSON config

Health Check
- `GET /healthz` -> `ok`

### APIs
See per-service docs:
  - [STT (Whisper)](https://github.com/pmbstyle/gllmc/blob/main/docs/STT_API.md)
  - [TTS (Piper)](https://github.com/pmbstyle/gllmc/blob/main/docs/TTS_API.md)
  - [Embeddings](https://github.com/pmbstyle/gllmc/blob/main/docs/Embeddings_API.md)

### Downloads and Caching
- Whisper binaries are downloaded per-platform into `<data-dir>/bin` with required libs.
- Whisper models are downloaded into `<data-dir>/models/whisper`.
- Embedding models are cached under `<data-dir>/models/embeddings`.
- Piper binary is installed under `<data-dir>/bin`; voice models under `<data-dir>/models/tts/<voice>`.

### Tests
- Run: `go test ./...`
- Tests cover health and embeddings (hash backend) and verify STT is disabled when not registered.

### WebSocket Endpoints
- Enable in config: `"websocket": { "enabled": true, "path_prefix": "/ws" }`
- Embeddings: `ws://<host>:<port>/ws/embeddings`
- STT: `ws://<host>:<port>/ws/stt`
- TTS: `ws://<host>:<port>/ws/tts`

### Test UI
- Enable in config: `"test_ui": { "enabled": true }`
- Access at: `http://<host>:<port>/test/`
