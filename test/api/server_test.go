package api_test

import (
    "bytes"
    "encoding/json"
    "io"
    "mime/multipart"
    "net/http"
    "net/http/httptest"
    "testing"

    "gollmcore/internal/server"
    "gollmcore/internal/services/embeddings"
)

func newTestServer(t *testing.T, emb embeddings.Service) *httptest.Server {
    t.Helper()
    mux := http.NewServeMux()
    server.RegisterRoutes(mux, server.Dependencies{
        STT:             nil, // disabled for tests
        STTDefaultModel: "base",
        Embeddings:      emb,
    })
    return httptest.NewServer(mux)
}

func TestHealthz(t *testing.T) {
    ts := newTestServer(t, nil)
    defer ts.Close()

    resp, err := http.Get(ts.URL + "/healthz")
    if err != nil { t.Fatalf("healthz request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        t.Fatalf("expected 200, got %d", resp.StatusCode)
    }
}

func TestBasicE2E_HealthAndEmbed(t *testing.T) {
    emb := embeddings.New(embeddings.Config{ModelName: "all-MiniLM-L6-v2"})
    ts := newTestServer(t, emb)
    defer ts.Close()

    // health
    if resp, err := http.Get(ts.URL + "/healthz"); err != nil {
        t.Fatalf("health failed: %v", err)
    } else if resp.StatusCode != http.StatusOK {
        t.Fatalf("health expected 200, got %d", resp.StatusCode)
    } else { resp.Body.Close() }

    // embeddings
    body := map[string]any{"input": "ping"}
    buf, _ := json.Marshal(body)
    resp, err := http.Post(ts.URL+"/v1/embeddings", "application/json", bytes.NewReader(buf))
    if err != nil { t.Fatalf("embed failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK { t.Fatalf("embed expected 200, got %d", resp.StatusCode) }
    var out struct{ Model string; Embeddings [][]float32 }
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil { t.Fatalf("decode failed: %v", err) }
    if len(out.Embeddings) != 1 || len(out.Embeddings[0]) == 0 { t.Fatalf("bad embeddings shape") }
}

func TestEmbeddings_SingleAndBatch(t *testing.T) {
    emb := embeddings.New(embeddings.Config{ModelName: "all-MiniLM-L6-v2"})
    ts := newTestServer(t, emb)
    defer ts.Close()

    // single input
    reqBody := map[string]any{"input": "hello world"}
    buf, _ := json.Marshal(reqBody)
    resp, err := http.Post(ts.URL+"/v1/embeddings", "application/json", bytes.NewReader(buf))
    if err != nil { t.Fatalf("emb request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK { t.Fatalf("expected 200, got %d", resp.StatusCode) }
    var out struct{
        Model string `json:"model"`
        Embeddings [][]float32 `json:"embeddings"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil { t.Fatalf("decode failed: %v", err) }
    if len(out.Embeddings) != 1 { t.Fatalf("expected 1 embedding, got %d", len(out.Embeddings)) }
    if got := len(out.Embeddings[0]); got != 384 { t.Fatalf("expected dim 384, got %d", got) }

    // batch input
    reqBody2 := map[string]any{"input": []string{"one", "two", "three"}}
    buf2, _ := json.Marshal(reqBody2)
    resp2, err := http.Post(ts.URL+"/v1/embeddings", "application/json", bytes.NewReader(buf2))
    if err != nil { t.Fatalf("emb request failed: %v", err) }
    defer resp2.Body.Close()
    if resp2.StatusCode != http.StatusOK { t.Fatalf("expected 200, got %d", resp2.StatusCode) }
    out = struct{ Model string `json:"model"`; Embeddings [][]float32 `json:"embeddings"` }{}
    if err := json.NewDecoder(resp2.Body).Decode(&out); err != nil { t.Fatalf("decode failed: %v", err) }
    if len(out.Embeddings) != 3 { t.Fatalf("expected 3 embeddings, got %d", len(out.Embeddings)) }
    if got := len(out.Embeddings[0]); got != 384 { t.Fatalf("expected dim 384, got %d", got) }
}

func TestEmbeddings_BadInput(t *testing.T) {
    emb := embeddings.New(embeddings.Config{ModelName: "all-MiniLM-L6-v2"})
    ts := newTestServer(t, emb)
    defer ts.Close()

    reqBody := map[string]any{"input": map[string]any{"not": "allowed"}}
    buf, _ := json.Marshal(reqBody)
    resp, err := http.Post(ts.URL+"/v1/embeddings", "application/json", bytes.NewReader(buf))
    if err != nil { t.Fatalf("request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusBadRequest {
        t.Fatalf("expected 400, got %d", resp.StatusCode)
    }
}

func TestSTTRoutesDisabled(t *testing.T) {
    ts := newTestServer(t, nil)
    defer ts.Close()

    body := &bytes.Buffer{}
    mw := multipart.NewWriter(body)
    w, _ := mw.CreateFormFile("file", "dummy.wav")
    _, _ = io.Copy(w, bytes.NewReader([]byte("fake")))
    mw.Close()
    req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/audio/transcriptions", body)
    req.Header.Set("Content-Type", mw.FormDataContentType())
    resp, err := http.DefaultClient.Do(req)
    if err != nil { t.Fatalf("request failed: %v", err) }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusNotFound {
        t.Fatalf("expected 404 when STT disabled, got %d", resp.StatusCode)
    }
}
