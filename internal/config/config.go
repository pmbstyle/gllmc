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
    LLM        LLM        `json:"llm"`
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
    if c.Services.Embeddings.Model == "" { c.Services.Embeddings.Model = "all-MiniLM-L6-v2" }
    if c.Services.TTS.Voice == "" { c.Services.TTS.Voice = "en_US-amy-medium" }
    // LLM defaults when enabled
    if c.Services.LLM.Enabled {
        if c.Services.LLM.Backend == "" { c.Services.LLM.Backend = "qwen-onnx" }
        if c.Services.LLM.Model.Name == "" { c.Services.LLM.Model.Name = "Qwen2.5-3B-Instruct-Q4_K_M" }
        if c.Services.LLM.Model.Filename == "" { c.Services.LLM.Model.Filename = "qwen2.5-3b-instruct-q4_k_m.gguf" }
        if c.Services.LLM.Model.URL == "" { c.Services.LLM.Model.URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" }
        if c.Services.LLM.Options.CtxLen == 0 { c.Services.LLM.Options.CtxLen = 4096 }
        // threads=0 means auto; gpu_layers default 0 for CPU
    }
    return c, nil
}

type TestUI struct {
    Enabled bool `json:"enabled"`
}

type LLM struct {
    Enabled    bool        `json:"enabled"`
    Backend    string      `json:"backend"`
    BinaryURL  string      `json:"binary_url"`   // optional; if empty, expect binary in PATH or preinstalled
    Model      LLMModel    `json:"model"`
    Options    LLMOptions  `json:"options"`
}

type LLMModel struct {
    Name     string `json:"name"`
    URL      string `json:"url"`
    Filename string `json:"filename"`
    OnnxURL  string `json:"onnx_url"`
    TokenizerURL string `json:"tokenizer_url"`
}

type LLMOptions struct {
    Threads   int `json:"threads"`
    CtxLen    int `json:"ctx_len"`
    GPULayers int `json:"gpu_layers"`
}
