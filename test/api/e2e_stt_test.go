package api_test

import (
    "bytes"
    "io"
    "mime/multipart"
    "net/http"
    "net/http/httptest"
    "os"
    "path/filepath"
    "testing"

    "gollmcore/internal/server"
    "gollmcore/internal/services/stt"
)

// This test verifies STT end-to-end via HTTP with a provided audio file.
// It is skipped by default and runs only when E2E_STT=1 and E2E_STT_AUDIO are set.
func TestE2E_STT_Transcription(t *testing.T) {
    if os.Getenv("E2E_STT") != "1" {
        t.Skip("skipping stt e2e, set E2E_STT=1 to enable")
    }
    audio := os.Getenv("E2E_STT_AUDIO")
    if audio == "" {
        t.Skip("set E2E_STT_AUDIO=/path/to/sample.wav")
    }
    if _, err := os.Stat(audio); err != nil { t.Fatalf("audio missing: %v", err) }

    dataDir := t.TempDir()
    svc := stt.New(filepath.Join(dataDir, "bin"), filepath.Join(dataDir, "models", "whisper"))
    mux := http.NewServeMux()
    server.RegisterRoutes(mux, server.Dependencies{ STT: svc, STTDefaultModel: "tiny" })
    ts := httptest.NewServer(mux)
    defer ts.Close()

    // Build multipart body
    var buf bytes.Buffer
    mw := multipart.NewWriter(&buf)
    fw, err := mw.CreateFormFile("file", filepath.Base(audio))
    if err != nil { t.Fatalf("form create failed: %v", err) }
    f, err := os.Open(audio)
    if err != nil { t.Fatalf("open audio failed: %v", err) }
    _, _ = io.Copy(fw, f)
    f.Close()
    mw.Close()

    req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/audio/transcriptions?model=tiny", &buf)
    req.Header.Set("Content-Type", mw.FormDataContentType())
    resp, err := http.DefaultClient.Do(req)
    if err != nil { t.Fatalf("request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        t.Fatalf("expected 200, got %d", resp.StatusCode)
    }
}

