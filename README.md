Go LLM Core — Local, Modular LLM Backend

Overview
- A single Go binary that runs local LLM-related APIs on Windows, macOS, and Linux.
- Modular services enabled by flags:
  - STT (Whisper) with on-demand binary and model downloads
  - Embeddings (fastembed via Python) with on-demand model downloads, or a hash fallback

Prerequisites
- Go 1.21+
- Internet access on first run to download binaries/models

Build
- From repo root:
  - `go build ./cmd/gollmcore`

Run
- With config file:
  - `./gollmcore --config configs/example-config.json`
  - Configure host/port, data_dir, service toggles, models/voices, WebSocket, and Test UI in JSON.

Ephemeral Port
- Use `--port 0` to bind an available ephemeral port; the server log prints the actual address, e.g., `HTTP server listening on 127.0.0.1:51243`.

Key Flag
- `--config` (string): Path to JSON config

Health Check
- `GET /healthz` -> `ok`

APIs
- See detailed per-service docs in `docs/`:
  - STT (Whisper): `docs/STT_API.md`
  - Embeddings: `docs/Embeddings_API.md`
  - TTS (Piper): `docs/TTS_API.md`

Downloads and Caching
- Whisper binaries are downloaded per-platform into `<data-dir>/bin` with required libs.
- Whisper models are downloaded into `<data-dir>/models/whisper`.
- Embedding models are cached under `<data-dir>/models/embeddings`.
- Piper binary is installed under `<data-dir>/bin`; voice models under `<data-dir>/models/tts/<voice>`.

Troubleshooting
- Embeddings backend fastembed requires Python 3.9+ in PATH. If missing, start with `--embedding-backend hash`.
- First requests may be slow due to model/binary downloads.
- On macOS/Linux, library paths for Whisper are set automatically at runtime. If you have custom security settings (e.g., macOS Gatekeeper), you may need to allow binaries manually.

Tests
- Run: `go test ./...`
- Tests cover health and embeddings (hash backend) and verify STT is disabled when not registered.

WebSocket Endpoints
- Enable in config: `"websocket": { "enabled": true, "path_prefix": "/ws" }`
- Embeddings: `ws://<host>:<port>/ws/embeddings`
- STT: `ws://<host>:<port>/ws/stt`
- TTS: `ws://<host>:<port>/ws/tts`

Test UI
- Enable in config: `"test_ui": { "enabled": true }`
- Access at: `http://<host>:<port>/test/`
