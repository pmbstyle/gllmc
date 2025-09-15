package server

import (
    "bufio"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "strings"

    "gollmcore/internal/services/embeddings"
    "gollmcore/internal/services/stt"
)

type Dependencies struct {
    STT             *stt.STTService
    STTDefaultModel string
    Embeddings      embeddings.Service
    TTS             TTSService
}

func RegisterRoutes(mux *http.ServeMux, d Dependencies) {
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        _, _ = w.Write([]byte("ok"))
    })

    if d.STT != nil {
        mux.HandleFunc("/v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
            if r.Method != http.MethodPost {
                http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
                return
            }
            handleSTTTranscribe(w, r, d)
        })
        mux.HandleFunc("/v1/audio/transcriptions/stream", func(w http.ResponseWriter, r *http.Request) {
            if r.Method != http.MethodPost {
                http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
                return
            }
            handleSTTTranscribeStream(w, r, d)
        })
    }

    if d.Embeddings != nil {
        mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
            if r.Method != http.MethodPost {
                http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
                return
            }
            handleEmbeddings(w, r, d)
        })
    }

    if d.TTS != nil {
        mux.HandleFunc("/v1/tts", func(w http.ResponseWriter, r *http.Request) {
            if r.Method != http.MethodPost { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
            handleTTS(w, r, d)
        })
    }
}

// -------- STT Handlers --------

func handleSTTTranscribe(w http.ResponseWriter, r *http.Request, d Dependencies) {
    model := r.URL.Query().Get("model")
    if model == "" { model = d.STTDefaultModel }

    file, hdr, err := r.FormFile("file")
    if err != nil {
        // try alternative field name
        file, hdr, err = r.FormFile("audio")
    }
    if err != nil {
        http.Error(w, "missing form file 'file' or 'audio'", http.StatusBadRequest)
        return
    }
    defer file.Close()

    tmpDir := os.TempDir()
    tmpPath := filepath.Join(tmpDir, "stt-"+sanitizeName(hdr.Filename))
    out, err := os.Create(tmpPath)
    if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }
    defer func(){ out.Close(); os.Remove(tmpPath) }()
    if _, err := io.Copy(out, file); err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }

    text, err := d.STT.TranscribeFile(r.Context(), tmpPath, model)
    if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }

    resp := map[string]any{"text": text, "model": model}
    w.Header().Set("Content-Type", "application/json")
    _ = json.NewEncoder(w).Encode(resp)
}

func handleSTTTranscribeStream(w http.ResponseWriter, r *http.Request, d Dependencies) {
    model := r.URL.Query().Get("model")
    if model == "" { model = d.STTDefaultModel }

    reader, hdr, err := r.FormFile("file")
    if err != nil { reader, hdr, err = r.FormFile("audio") }
    if err != nil {
        http.Error(w, "missing form file 'file' or 'audio'", http.StatusBadRequest)
        return
    }
    defer reader.Close()

    tmpDir := os.TempDir()
    tmpPath := filepath.Join(tmpDir, "stt-"+sanitizeName(hdr.Filename))
    out, err := os.Create(tmpPath)
    if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }
    if _, err := io.Copy(out, reader); err != nil { out.Close(); http.Error(w, err.Error(), http.StatusInternalServerError); return }
    out.Close()
    defer os.Remove(tmpPath)

    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "streaming unsupported", http.StatusInternalServerError)
        return
    }

    linesCh, errCh := d.STT.TranscribeFileStream(r.Context(), tmpPath, model)
    enc := func(s string) string { return strings.ReplaceAll(s, "\n", " ") }
    for {
        select {
        case line, ok := <-linesCh:
            if !ok {
                fmt.Fprintf(w, "event: done\n")
                fmt.Fprintf(w, "data: %s\n\n", "")
                flusher.Flush()
                return
            }
            fmt.Fprintf(w, "data: %s\n\n", enc(line))
            flusher.Flush()
        case err := <-errCh:
            if err != nil {
                log.Printf("stream error: %v", err)
            }
            return
        case <-r.Context().Done():
            return
        }
    }
}

func sanitizeName(name string) string {
    name = filepath.Base(name)
    name = strings.ReplaceAll(name, " ", "-")
    return name
}

// -------- Embeddings Handler --------

type embeddingsRequest struct {
    Input any `json:"input"` // string or []string
}

type embeddingsResponse struct {
    Model      string        `json:"model"`
    Embeddings [][]float32   `json:"embeddings"`
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request, d Dependencies) {
    var req embeddingsRequest
    if err := json.NewDecoder(bufio.NewReader(r.Body)).Decode(&req); err != nil {
        http.Error(w, "invalid json", http.StatusBadRequest)
        return
    }
    var inputs []string
    switch v := req.Input.(type) {
    case string:
        inputs = []string{v}
    case []any:
        for _, it := range v {
            if s, ok := it.(string); ok { inputs = append(inputs, s) }
        }
    default:
        http.Error(w, "input must be string or array of strings", http.StatusBadRequest)
        return
    }
    if len(inputs) == 0 {
        http.Error(w, "no input provided", http.StatusBadRequest)
        return
    }
    vecs, model, err := d.Embeddings.Embed(r.Context(), inputs)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    w.Header().Set("Content-Type", "application/json")
    _ = json.NewEncoder(w).Encode(embeddingsResponse{Model: model, Embeddings: vecs})
}

// -------- TTS Handler --------

type ttsRequest struct {
    Text  string `json:"text"`
    Voice string `json:"voice"`
}

func handleTTS(w http.ResponseWriter, r *http.Request, d Dependencies) {
    var req ttsRequest
    if err := json.NewDecoder(bufio.NewReader(r.Body)).Decode(&req); err != nil { http.Error(w, "invalid json", http.StatusBadRequest); return }
    if req.Text == "" { http.Error(w, "missing text", http.StatusBadRequest); return }
    audio, err := d.TTS.Synthesize(r.Context(), req.Text, req.Voice)
    if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }
    w.Header().Set("Content-Type", "audio/wav")
    w.Header().Set("Content-Disposition", "inline; filename=tts.wav")
    w.WriteHeader(http.StatusOK)
    _, _ = w.Write(audio)
}
