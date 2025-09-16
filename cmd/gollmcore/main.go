package main

import (
    "context"
    "flag"
    "log"
    "net/http"
    "net"
    "os"
    "os/signal"
    "path/filepath"
    "syscall"
    "time"

    "gollmcore/internal/config"
    "gollmcore/internal/server"
    "gollmcore/internal/services/embeddings"
    ttsvc "gollmcore/internal/services/tts"
    llmsvc "gollmcore/internal/services/llm"
    "gollmcore/internal/services/stt"
    "strings"
)

func main() {
    var cfgPath string
    flag.StringVar(&cfgPath, "config", "config.json", "Path to config file")
    flag.Parse()

    c, err := config.Load(cfgPath)
    if err != nil {
        log.Fatalf("failed to load config: %v", err)
    }

    dataDir := c.Server.DataDir
    if dataDir == "" { dataDir = defaultDataDir() }
    if err := os.MkdirAll(dataDir, 0o755); err != nil {
        log.Fatalf("failed creating data dir: %v", err)
    }

    ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
    defer cancel()

    // Initialize services as requested
    var sttSvc *stt.STTService
    var embSvc embeddings.Service
    var ttsSvc *ttsvc.Service
    var llmSvc *llmsvc.Service

    if c.Services.STT.Enabled {
        sttSvc = stt.New(filepath.Join(dataDir, "bin"), filepath.Join(dataDir, "models", "whisper"))
        // Lazy downloads happen on first request.
        log.Printf("STT service enabled with model: %s", c.Services.STT.Model)
    }

    if c.Services.Embeddings.Enabled {
        // Use real MiniLM ONNX-backed embeddings
        modelDir := filepath.Join(dataDir, "models", "embeddings", "all-MiniLM-L6-v2")
        svc, err := embeddings.NewMiniLM(modelDir)
        if err != nil {
            log.Fatalf("failed to init embeddings (MiniLM ONNX): %v", err)
        }
        embSvc = svc
        log.Printf("Embeddings service enabled with model: %s", "all-MiniLM-L6-v2")
    }

    if c.Services.TTS.Enabled {
        ttsSvc = ttsvc.New(filepath.Join(dataDir, "bin"), filepath.Join(dataDir, "models", "tts"), filepath.Join(dataDir, "tts"))
        log.Printf("TTS service enabled with voice: %s", c.Services.TTS.Voice)
    }

    if c.Services.LLM.Enabled {
        if strings.EqualFold(c.Services.LLM.Backend, "qwen-onnx") || c.Services.LLM.Backend == "" {
            // Use ONNX in-process backend for Qwen3-0.6B
            modelDir := filepath.Join(dataDir, "models", "llm", "qwen3-0.6b")
            // Prefer FP32 first, then FP16 fallback
            onnxURL := c.Services.LLM.Model.OnnxURL
            if onnxURL == "" {
                onnxURL = "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model_fp32.onnx"
            }
            tokURL := c.Services.LLM.Model.TokenizerURL
            if tokURL == "" {
                tokURL = "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/tokenizer.json"
            }
            qwen, err := llmsvc.NewQwenONNX(modelDir, onnxURL, tokURL)
            if err != nil { log.Fatalf("failed to init Qwen ONNX backend: %v", err) }
            // Wrap QwenONNX into proxy-like handlers
            llmSvc = llmsvc.New(
                filepath.Join(dataDir, "bin"),
                filepath.Join(dataDir, "models", "llm"),
                filepath.Join(dataDir, "llm"),
                "", "", "", 0, 0, 0,
            )
            // Replace proxy handlers to call qwen.Generate
            // We piggyback the existing proxy endpoints by adding local handlers below.
            // For simplicity, we reuse server routes:
            // We'll assign llmSvc with custom Proxy methods via closure in server.RegisterRoutes
            // Done below via Dependencies: LLM remains llmSvc; server will call its methods.
            // We inject function pointers using package-level variables (hacky, but minimal change).
            llmsvc.LlmsvcSetLocal(qwen)
            log.Printf("LLM service enabled with Qwen3-0.6B-ONNX backend")
        } else {
            // Fallback to llama.cpp proxy if configured
            llmBinDir := filepath.Join(dataDir, "bin")
            llmModelDir := filepath.Join(dataDir, "models", "llm")
            llmWorkDir := filepath.Join(dataDir, "llm")
            llmSvc = llmsvc.New(
                llmBinDir,
                llmModelDir,
                llmWorkDir,
                c.Services.LLM.Model.URL,
                c.Services.LLM.Model.Filename,
                c.Services.LLM.BinaryURL,
                c.Services.LLM.Options.Threads,
                c.Services.LLM.Options.CtxLen,
                c.Services.LLM.Options.GPULayers,
            )
            if err := llmSvc.EnsureReady(ctx); err != nil { log.Fatalf("failed to start LLM service: %v", err) }
            log.Printf("LLM service enabled (llama proxy) with model: %s", c.Services.LLM.Model.Name)
        }
    }

    // Start HTTP server
    mux := http.NewServeMux()
    server.RegisterRoutes(mux, server.Dependencies{
        STT:             sttSvc,
        STTDefaultModel: c.Services.STT.Model,
        Embeddings:      embSvc,
        TTS:             ttsSvc,
        LLM:             llmSvc,
    })

    // Optional WebSocket endpoints
    server.RegisterWSRoutes(mux, server.Dependencies{
        STT:             sttSvc,
        STTDefaultModel: c.Services.STT.Model,
        Embeddings:      embSvc,
        TTS:             ttsSvc,
    }, server.WSOptions{Enable: c.WebSocket.Enabled, PathPrefix: c.WebSocket.PathPrefix})

    // Optional Test UI
    if c.TestUI.Enabled {
        server.RegisterTestUI(mux)
    }

    // Bind explicitly so we can support port=0 and log the actual port
    ln, err := net.Listen("tcp", c.Server.Host+":"+itoa(c.Server.Port))
    if err != nil { log.Fatalf("listen error: %v", err) }
    srv := &http.Server{Handler: mux}

    // Startup summary log
    sttStatus := "disabled"
    if sttSvc != nil {
        sttStatus = "enabled (model=" + c.Services.STT.Model + ")"
    }
    embStatus := "disabled"
    if embSvc != nil {
        embStatus = "enabled (model=all-MiniLM-L6-v2)"
    }
    wsStatus := "disabled"
    if c.WebSocket.Enabled { wsStatus = "enabled (prefix=" + c.WebSocket.PathPrefix + ")" }
    ttsStatus := "disabled"
    if ttsSvc != nil {
        ttsStatus = "enabled (voice=" + c.Services.TTS.Voice + ")"
    }
    log.Printf("Startup summary:\n  Address: %s\n  DataDir: %s\n  STT: %s\n  Embeddings: %s\n  TTS: %s\n  WebSocket: %s", ln.Addr().String(), dataDir, sttStatus, embStatus, ttsStatus, wsStatus)

    go func() {
        if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
            log.Fatalf("server error: %v", err)
        }
    }()

    <-ctx.Done()
    log.Printf("shutting down...")

    shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancelShutdown()
    _ = srv.Shutdown(shutdownCtx)
}

func defaultDataDir() string {
    if dir, err := os.UserConfigDir(); err == nil {
        return filepath.Join(dir, "gollmcore")
    }
    return filepath.Join(".", ".gollmcore")
}

func itoa(n int) string { return fmtInt(n) }

// tiny helper to avoid importing strconv across files
func fmtInt(n int) string {
    const digits = "0123456789"
    if n == 0 { return "0" }
    neg := false
    if n < 0 { neg = true; n = -n }
    buf := [20]byte{}
    i := len(buf)
    for n > 0 { i--; buf[i] = digits[n%10]; n /= 10 }
    if neg { i--; buf[i] = '-' }
    return string(buf[i:])
}
