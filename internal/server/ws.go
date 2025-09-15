package server

import (
    "context"
    "encoding/base64"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "time"

    "github.com/gorilla/websocket"
)

type WSOptions struct {
    Enable     bool
    PathPrefix string
}

var upgrader = websocket.Upgrader{ CheckOrigin: func(r *http.Request) bool { return true } }

func RegisterWSRoutes(mux *http.ServeMux, d Dependencies, o WSOptions) {
    if !o.Enable { return }
    prefix := o.PathPrefix
    if prefix == "" { prefix = "/ws" }

    if d.Embeddings != nil {
        mux.HandleFunc(prefix+"/embeddings", func(w http.ResponseWriter, r *http.Request) {
            conn, err := upgrader.Upgrade(w, r, nil)
            if err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }
            defer conn.Close()
            for {
                var req struct{ Input any `json:"input"` }
                if err := conn.ReadJSON(&req); err != nil { return }
                inputs := coerceInputsWS(req.Input)
                if len(inputs) == 0 {
                    _ = conn.WriteJSON(map[string]any{"error":"no input"})
                    continue
                }
                ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
                vecs, model, err := d.Embeddings.Embed(ctx, inputs)
                cancel()
                if err != nil { _ = conn.WriteJSON(map[string]any{"error":err.Error()}); continue }
                _ = conn.WriteJSON(map[string]any{"ok":true, "model": model, "embeddings": vecs})
            }
        })
    }

    if d.STT != nil {
        mux.HandleFunc(prefix+"/stt", func(w http.ResponseWriter, r *http.Request) {
            conn, err := upgrader.Upgrade(w, r, nil)
            if err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }
            defer conn.Close()
            for {
                var req struct{
                    Filename   string `json:"filename"`
                    Model      string `json:"model"`
                    AudioB64   string `json:"audio_base64"`
                    Stream     bool   `json:"stream"`
                }
                if err := conn.ReadJSON(&req); err != nil { return }
                model := req.Model
                if model == "" { model = d.STTDefaultModel }
                // Write audio to temp file
                b, err := base64.StdEncoding.DecodeString(req.AudioB64)
                if err != nil { _ = conn.WriteJSON(map[string]any{"error":"invalid base64"}); continue }
                tmp := filepath.Join(os.TempDir(), "ws-audio-"+sanitizeName(req.Filename))
                if err := os.WriteFile(tmp, b, 0o644); err != nil { _ = conn.WriteJSON(map[string]any{"error":err.Error()}); continue }
                defer os.Remove(tmp)
                if req.Stream {
                    _ = conn.WriteJSON(map[string]any{"event":"status", "message":"starting transcription"})
                    lines, errs := d.STT.TranscribeFileStream(r.Context(), tmp, model)
                    for {
                        select {
                        case l, ok := <-lines:
                            if !ok { _ = conn.WriteJSON(map[string]any{"event":"done"}); goto done }
                            _ = conn.WriteJSON(map[string]any{"event":"data", "text": l})
                        case e := <-errs:
                            if e != nil { _ = conn.WriteJSON(map[string]any{"error": e.Error()}) }
                            goto done
                        case <-r.Context().Done():
                            goto done
                        }
                    }
                done:
                    continue
                }
                text, err := d.STT.TranscribeFile(r.Context(), tmp, model)
                if err != nil { _ = conn.WriteJSON(map[string]any{"error": err.Error()}); continue }
                _ = conn.WriteJSON(map[string]any{"ok":true, "text": text, "model": model})
            }
        })
    }
    if d.TTS != nil {
        mux.HandleFunc(prefix+"/tts", func(w http.ResponseWriter, r *http.Request) {
            conn, err := upgrader.Upgrade(w, r, nil)
            if err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }
            defer conn.Close()
            for {
                var req struct{ Text, Voice string }
                if err := conn.ReadJSON(&req); err != nil { return }
                if req.Text == "" { _ = conn.WriteJSON(map[string]any{"error":"missing text"}); continue }
                audio, err := d.TTS.Synthesize(r.Context(), req.Text, req.Voice)
                if err != nil { _ = conn.WriteJSON(map[string]any{"error": err.Error()}); continue }
                // Return as base64 to keep it simple for browser
                _ = conn.WriteJSON(map[string]any{"ok": true, "mime": "audio/wav", "audio_base64": base64.StdEncoding.EncodeToString(audio)})
            }
        })
    }
    log.Printf("WebSocket endpoints enabled at %s/{embeddings,stt,tts}", prefix)
}

func coerceInputsWS(in any) []string {
    switch v := in.(type) {
    case string:
        return []string{v}
    case []any:
        out := make([]string, 0, len(v))
        for _, it := range v { if s, ok := it.(string); ok { out = append(out, s) } }
        return out
    default:
        return nil
    }
}
