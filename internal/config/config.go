package config

import (
    "encoding/json"
    "fmt"
    "os"
)

type Server struct {
    Host    string `json:"host"`
    Port    int    `json:"port"`
    DataDir string `json:"data_dir"`
}

type STT struct {
    Enabled bool   `json:"enabled"`
    Model   string `json:"model"`
}

type Embeddings struct {
    Enabled bool   `json:"enabled"`
    Backend string `json:"backend"`   // fastembed|hash
    Model   string `json:"model"`
}

type TTS struct {
    Enabled bool   `json:"enabled"`
    Voice   string `json:"voice"` // e.g., en_US-amy-medium
}

type WebSocket struct {
    Enabled    bool   `json:"enabled"`
    PathPrefix string `json:"path_prefix"`
}

type Services struct {
    STT        STT        `json:"stt"`
    Embeddings Embeddings `json:"embeddings"`
    TTS        TTS        `json:"tts"`
}

type Config struct {
    Server    Server    `json:"server"`
    Services  Services  `json:"services"`
    WebSocket WebSocket `json:"websocket"`
    TestUI    TestUI    `json:"test_ui"`
}

func Load(path string) (Config, error) {
    var c Config
    b, err := os.ReadFile(path)
    if err != nil { return c, fmt.Errorf("read config: %w", err) }
    if err := json.Unmarshal(b, &c); err != nil { return c, fmt.Errorf("parse config: %w", err) }
    if c.Server.Host == "" { c.Server.Host = "127.0.0.1" }
    if c.Server.Port == 0 { c.Server.Port = 8080 }
    if c.WebSocket.PathPrefix == "" { c.WebSocket.PathPrefix = "/ws" }
    if c.Services.STT.Model == "" { c.Services.STT.Model = "base" }
    if c.Services.Embeddings.Backend == "" { c.Services.Embeddings.Backend = "fastembed" }
    if c.Services.Embeddings.Model == "" { c.Services.Embeddings.Model = "sentence-transformers/all-MiniLM-L6-v2" }
    if c.Services.TTS.Voice == "" { c.Services.TTS.Voice = "en_US-amy-medium" }
    return c, nil
}

type TestUI struct {
    Enabled bool `json:"enabled"`
}
