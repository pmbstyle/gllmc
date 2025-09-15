Embeddings API

Overview
- Produces dense vector embeddings for text.
- Default backend uses a local deterministic hash embedding for tests/dev; config can enable a real model backend and cache.

REST Endpoint
- POST `/v1/embeddings`
  - Request JSON:
    - `{ "input": "hello world" }` or `{ "input": ["hello", "world"] }`
  - Response JSON:
    - `{ "model": "<name>", "embeddings": [[...], ...] }`

WebSocket
- `ws://<host>:<port>/<prefix>/embeddings`
  - Send: `{ "input": "hello" }` or `{ "input": ["one","two"] }`
  - Receive: `{ "ok": true, "model": "...", "embeddings": [[...], ...] }`

Notes
- Model name and backend configured in the server config file.
- Vectors are L2-normalized by the current implementations.

