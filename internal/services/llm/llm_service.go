package llm

import (
    "bufio"
    "context"
    "errors"
    "fmt"
    "encoding/json"
    "io"
    "archive/zip"
    "compress/gzip"
    "log"
    "net"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "strings"
    "time"
    "archive/tar"
)

// Local Qwen ONNX backend bridge (simplified). We set function pointers to call local gen.
var localQwen *QwenONNX

func LlmsvcSetLocal(q *QwenONNX) { localQwen = q }

type Service struct {
    binDir   string
    modelDir string
    workDir  string

    modelURL  string
    modelFile string
    binaryURL string

    threads   int
    ctxLen    int
    gpuLayers int

    srvCmd *exec.Cmd
    addr   string
}

func New(binDir, modelDir, workDir, modelURL, modelFile, binaryURL string, threads, ctxLen, gpuLayers int) *Service {
    return &Service{binDir: binDir, modelDir: modelDir, workDir: workDir, modelURL: modelURL, modelFile: modelFile, binaryURL: binaryURL, threads: threads, ctxLen: ctxLen, gpuLayers: gpuLayers}
}

func (s *Service) EnsureReady(ctx context.Context) error {
    if err := os.MkdirAll(s.binDir, 0o755); err != nil { return err }
    if err := os.MkdirAll(s.modelDir, 0o755); err != nil { return err }
    // model
    mp := filepath.Join(s.modelDir, s.modelFile)
    if _, err := os.Stat(mp); err != nil {
        if s.modelURL == "" { return fmt.Errorf("LLM model URL not set and model missing: %s", mp) }
        if err := downloadFile(s.modelURL, mp, 0); err != nil { return fmt.Errorf("download model: %w", err) }
    }
    // binary
    bin := s.findServerBinary()
    if bin == "" {
        if s.binaryURL != "" {
            if err := s.fetchAndInstallBinary(s.binaryURL); err != nil { return fmt.Errorf("download llama server: %w", err) }
        } else {
            // Attempt platform-default URLs
            if err := s.downloadServerBinaryDefault(); err != nil {
                return fmt.Errorf("llama.cpp server (llama-server) not found. Place it in PATH or in %s, or set services.llm.binary_url to a downloadable binary. Last error: %w", s.binDir, err)
            }
        }
        bin = s.findServerBinary()
    }
    if bin == "" { return fmt.Errorf("failed to locate llama-server binary in %s", s.binDir) }
    // start server if not running
    if s.addr == "" {
        if err := s.startServer(ctx, bin, mp); err != nil { return err }
    }
    return nil
}

func (s *Service) findServerBinary() string {
    names := []string{"llama-server", "server", "llama-server.exe", "server.exe"}
    for _, n := range names {
        p := filepath.Join(s.binDir, n)
        if fi, err := os.Stat(p); err == nil && !fi.IsDir() { return p }
    }
    // Try PATH
    pathNames := []string{"llama-server", "server"}
    if runtime.GOOS == "windows" {
        pathNames = append(pathNames, "llama-server.exe", "server.exe")
    }
    for _, n := range pathNames {
        if p, err := exec.LookPath(n); err == nil {
            return p
        }
    }
    return ""
}

func (s *Service) startServer(ctx context.Context, bin, modelPath string) error {
    ln, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil { return err }
    addr := ln.Addr().String()
    port := strings.Split(addr, ":")[1]
    ln.Close()

    args := []string{"--model", modelPath, "--host", "127.0.0.1", "--port", port}
    if s.threads > 0 { args = append(args, "--threads", itoa(s.threads)) }
    if s.ctxLen > 0 { args = append(args, "--ctx-size", itoa(s.ctxLen)) }
    if s.gpuLayers > 0 { args = append(args, "--gpu-layers", itoa(s.gpuLayers)) }

    cmd := exec.CommandContext(ctx, bin, args...)
    cmd.Dir = s.binDir
    stderr, _ := cmd.StderrPipe()
    stdout, _ := cmd.StdoutPipe()
    if err := cmd.Start(); err != nil { return err }
    s.srvCmd = cmd
    s.addr = "http://127.0.0.1:" + port
    go func() {
        scanner := bufio.NewScanner(io.MultiReader(stdout, stderr))
        for scanner.Scan() { log.Printf("llama-server: %s", scanner.Text()) }
    }()
    // wait until health ready
    deadline := time.Now().Add(20 * time.Second)
    for time.Now().Before(deadline) {
        if s.health() == nil { return nil }
        time.Sleep(500 * time.Millisecond)
    }
    return fmt.Errorf("llama-server failed to become ready at %s", s.addr)
}

func (s *Service) health() error {
    if s.addr == "" { return errors.New("not started") }
    req, _ := http.NewRequest(http.MethodGet, s.addr+"/health", nil)
    c := &http.Client{ Timeout: 1500 * time.Millisecond }
    resp, err := c.Do(req)
    if err != nil { return err }
    resp.Body.Close()
    if resp.StatusCode != 200 { return fmt.Errorf("status %d", resp.StatusCode) }
    return nil
}

// ProxyChatCompletions forwards an OpenAI-style chat completion request to llama-server
func (s *Service) ProxyChatCompletions(w http.ResponseWriter, r *http.Request) {
    if localQwen != nil {
        // Minimal handler: read prompt from messages and generate
        var req struct{ Model string `json:"model"`; Messages []struct{ Role, Content string } `json:"messages"`; MaxTokens int `json:"max_tokens"` }
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil { http.Error(w, "bad json", 400); return }
        var prompt string
        for _, m := range req.Messages { if strings.ToLower(m.Role) == "user" { prompt = prompt + m.Content + "\n" } }
        // Streaming or non-stream based on request; here: always non-stream for now
        stream := strings.Contains(strings.ToLower(r.Header.Get("Accept")), "text/event-stream")
        ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second); defer cancel()
        if stream {
            w.Header().Set("Content-Type", "text/event-stream")
            w.Header().Set("Cache-Control", "no-cache")
            flusher, ok := w.(http.Flusher)
            if !ok { http.Error(w, "stream unsupported", 500); return }
            partial, _ := localQwen.GenerateWithCallback(ctx, prompt, req.MaxTokens, func(s string) {
                fmt.Fprintf(w, "data: %s\n\n", s)
                flusher.Flush()
            })
            fmt.Fprintf(w, "event: done\n")
            fmt.Fprintf(w, "data: %s\n\n", partial)
            flusher.Flush()
            return
        } else {
            text, err := localQwen.Generate(ctx, prompt, req.MaxTokens)
            if err != nil { http.Error(w, err.Error(), 500); return }
            resp := map[string]any{
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "choices": []any{map[string]any{"index":0, "message": map[string]string{"role":"assistant","content": text}, "finish_reason":"stop"}},
            }
            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(resp)
        }
        return
    }
    if s.addr == "" { http.Error(w, "llm not ready", 503); return }
    url := s.addr + "/v1/chat/completions"
    s.proxyJSON(w, r, url)
}

// ProxyCompletions forwards an OpenAI-style completion request to llama-server
func (s *Service) ProxyCompletions(w http.ResponseWriter, r *http.Request) {
    if localQwen != nil { http.Error(w, "use chat.completions", 400); return }
    if s.addr == "" { http.Error(w, "llm not ready", 503); return }
    url := s.addr + "/v1/completions"
    s.proxyJSON(w, r, url)
}

func (s *Service) proxyJSON(w http.ResponseWriter, r *http.Request, url string) {
    body, err := io.ReadAll(r.Body)
    if err != nil { http.Error(w, err.Error(), 400); return }
    req, _ := http.NewRequest(r.Method, url, strings.NewReader(string(body)))
    req.Header.Set("Content-Type", "application/json")
    // stream or non-stream based on client; just forward
    resp, err := http.DefaultClient.Do(req)
    if err != nil { http.Error(w, err.Error(), 502); return }
    defer resp.Body.Close()
    for k, vv := range resp.Header { for _, v := range vv { w.Header().Add(k, v) } }
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
}

// helpers
func downloadFile(url, dst string, timeout time.Duration) error {
    req, err := http.NewRequest(http.MethodGet, url, nil)
    if err != nil { return err }
    if timeout == 0 { timeout = 300 * time.Second }
    c := &http.Client{ Timeout: timeout }
    resp, err := c.Do(req)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode < 200 || resp.StatusCode >= 300 { return fmt.Errorf("bad status: %s", resp.Status) }
    tmp := dst + ".part"
    f, err := os.Create(tmp)
    if err != nil { return err }
    if _, err := io.Copy(f, resp.Body); err != nil { f.Close(); return err }
    f.Close()
    return os.Rename(tmp, dst)
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

func (s *Service) fetchAndInstallBinary(url string) error {
    if err := os.MkdirAll(s.binDir, 0o755); err != nil { return err }
    tmp := filepath.Join(s.binDir, filepath.Base(url))
    if err := downloadFile(url, tmp, 0); err != nil { return err }
    lower := strings.ToLower(tmp)
    if strings.HasSuffix(lower, ".zip") {
        if err := extractZipSelect(tmp, s.binDir, []string{"llama-server", "llama-server.exe"}); err != nil { return err }
        _ = os.Remove(tmp)
    } else if strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz") {
        if err := extractTgzSelect(tmp, s.binDir, []string{"llama-server"}); err != nil { return err }
        _ = os.Remove(tmp)
    } else {
        // assume it's directly the binary
        dest := filepath.Join(s.binDir, "llama-server")
        if runtime.GOOS == "windows" { dest += ".exe" }
        if err := os.Rename(tmp, dest); err != nil { return err }
        if runtime.GOOS != "windows" { _ = os.Chmod(dest, 0o755) }
    }
    return nil
}

func (s *Service) downloadServerBinaryDefault() error {
    // Note: Replace these URLs with your hosted, trusted binaries.
    // These are placeholders illustrating the pattern, similar to Whisper/Piper handling.
    var url string
    switch runtime.GOOS {
    case "windows":
        url = "https://aliceai.ca/app_assets/llama/llama-server-windows-amd64.zip"
    case "darwin":
        if runtime.GOARCH == "arm64" {
            url = "https://aliceai.ca/app_assets/llama/llama-server-macos-arm64.tar.gz"
        } else {
            url = "https://aliceai.ca/app_assets/llama/llama-server-macos-x64.tar.gz"
        }
    case "linux":
        url = "https://aliceai.ca/app_assets/llama/llama-server-linux-x64.tar.gz"
    default:
        return fmt.Errorf("unsupported platform: %s/%s", runtime.GOOS, runtime.GOARCH)
    }
    return s.fetchAndInstallBinary(url)
}

func extractZipSelect(zipPath, dstDir string, names []string) error {
    set := make(map[string]bool)
    for _, n := range names { set[strings.ToLower(n)] = true }
    r, err := zip.OpenReader(zipPath)
    if err != nil { return err }
    defer r.Close()
    for _, f := range r.File {
        base := strings.ToLower(filepath.Base(f.Name))
        if !set[base] { continue }
        rc, err := f.Open(); if err != nil { return err }
        defer rc.Close()
        out := filepath.Join(dstDir, filepath.Base(f.Name))
        of, err := os.Create(out); if err != nil { return err }
        if _, err := io.Copy(of, rc); err != nil { of.Close(); return err }
        of.Close()
        if runtime.GOOS != "windows" { _ = os.Chmod(out, 0o755) }
    }
    return nil
}

func extractTgzSelect(tgzPath, dstDir string, names []string) error {
    set := make(map[string]bool)
    for _, n := range names { set[n] = true }
    f, err := os.Open(tgzPath)
    if err != nil { return err }
    defer f.Close()
    gz, err := gzip.NewReader(f)
    if err != nil { return err }
    defer gz.Close()
    tr := tar.NewReader(gz)
    for {
        hdr, err := tr.Next(); if err == io.EOF { break }; if err != nil { return err }
        base := filepath.Base(hdr.Name)
        if !set[base] || hdr.FileInfo().IsDir() { continue }
        out := filepath.Join(dstDir, base)
        of, err := os.Create(out); if err != nil { return err }
        if _, err := io.Copy(of, tr); err != nil { of.Close(); return err }
        of.Close()
        if runtime.GOOS != "windows" { _ = os.Chmod(out, 0o755) }
    }
    return nil
}
