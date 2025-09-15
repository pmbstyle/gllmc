package api_test

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "os"
    "testing"

    "gollmcore/internal/server"
    "gollmcore/internal/services/embeddings"
)

// This test verifies the fastembed backend end-to-end via HTTP.
// It is skipped by default and runs only when E2E_EMB=1 is set.
func TestE2E_Embeddings_FastEmbed(t *testing.T) {
    if os.Getenv("E2E_EMB") != "1" {
        t.Skip("skipping fastembed e2e, set E2E_EMB=1 to enable")
    }
    model := os.Getenv("E2E_EMB_MODEL")
    if model == "" { model = "sentence-transformers/all-MiniLM-L6-v2" }

    dataDir := t.TempDir()
    emb, err := embeddings.NewWithBackend(embeddings.Config{
        Backend:   "fastembed",
        ModelName: model,
        ModelDir:  dataDir,
        WorkDir:   dataDir,
    })
    if err != nil { t.Skipf("fastembed init failed: %v", err) }

    mux := http.NewServeMux()
    server.RegisterRoutes(mux, server.Dependencies{ Embeddings: emb })
    ts := httptest.NewServer(mux)
    defer ts.Close()

    in := map[string]any{"input": []string{"hello", "world"}}
    body, _ := json.Marshal(in)
    resp, err := http.Post(ts.URL+"/v1/embeddings", "application/json", bytes.NewReader(body))
    if err != nil { t.Fatalf("request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK { t.Fatalf("expected 200, got %d", resp.StatusCode) }
    var out struct{ Model string; Embeddings [][]float32 }
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil { t.Fatalf("decode failed: %v", err) }
    if len(out.Embeddings) != 2 { t.Fatalf("expected 2 embeddings, got %d", len(out.Embeddings)) }
    if len(out.Embeddings[0]) == 0 { t.Fatalf("got empty vector") }
}
