package tts

import (
    "archive/tar"
    "archive/zip"
    "bytes"
    "compress/gzip"
    "context"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "strings"
    "time"
)

type Service struct {
    binDir   string
    modelDir string
    workDir  string // unused now; reserved
}

func New(binDir, modelDir, workDir string) *Service {
    return &Service{binDir: binDir, modelDir: modelDir, workDir: workDir}
}

func (s *Service) Synthesize(ctx context.Context, text, voice string) ([]byte, error) {
    if text == "" { return nil, fmt.Errorf("empty text") }
    if voice == "" { voice = "en_US-amy-medium" }
    if err := s.ensurePiperInstalled(ctx); err != nil { return nil, err }
    modelPath, err := s.ensureVoiceModel(ctx, voice)
    if err != nil { return nil, err }
    piper := s.piperBinaryPath()
    if piper == "" { return nil, fmt.Errorf("piper binary not found") }

    outPath := filepath.Join(os.TempDir(), fmt.Sprintf("piper_out_%d.wav", time.Now().UnixNano()))
    defer os.Remove(outPath)
    cmd, err := s.piperExecCommand(ctx, modelPath, outPath, text)
    if err != nil { return nil, err }
    var stderr bytes.Buffer
    cmd.Stderr = &stderr
    if err := cmd.Run(); err != nil {
        return nil, fmt.Errorf("piper failed: %v: %s", err, stderr.String())
    }
    data, err := os.ReadFile(outPath)
    if err != nil { return nil, err }
    return data, nil
}

func (s *Service) ensurePiperInstalled(ctx context.Context) error {
    if err := os.MkdirAll(s.binDir, 0o755); err != nil { return err }
    // Prefer Python package path
    if s.piperBinaryPath() != "" { return nil }
    urls, file := piperDownloadURLs()
    if len(urls) == 0 { return fmt.Errorf("unsupported platform for piper: %s/%s", runtime.GOOS, runtime.GOARCH) }
    if err := os.MkdirAll(s.binDir, 0o755); err != nil { return err }
    downloadPath := filepath.Join(s.binDir, file)
    var last error
    for i, u := range urls {
        log.Printf("TTS: attempting to download Piper binary %d/%d: %s", i+1, len(urls), u)
        if err := downloadFileWithRetry(u, downloadPath, 2, 180*time.Second); err != nil {
            last = err
            continue
        }
        // Handle different file types
        lower := strings.ToLower(file)
        if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && file == "piper-macos-arm64" {
            target := filepath.Join(s.binDir, "piper")
            if err := os.Rename(downloadPath, target); err != nil { last = err; continue }
            _ = os.Chmod(target, 0o755)
        } else if strings.HasSuffix(lower, ".zip") {
            if err := extractZip(downloadPath, s.binDir); err != nil { last = err; continue }
            _ = os.Remove(downloadPath)
        } else if strings.HasSuffix(lower, ".tar.gz") {
            if err := extractTarGz(downloadPath, s.binDir); err != nil { last = err; continue }
            _ = os.Remove(downloadPath)
        }
        if s.piperBinaryPath() != "" { return nil }
        last = fmt.Errorf("piper binary not found after extraction")
    }
    if last != nil { return last }
    return fmt.Errorf("failed to install Piper binary")
}

func (s *Service) piperExecCommand(ctx context.Context, modelPath, outPath, text string) (*exec.Cmd, error) {
    // Always use platform binary
    bin := s.piperBinaryPath()
    if bin == "" { return nil, fmt.Errorf("piper binary not found") }
    args := []string{"--model", modelPath, "--output-file", outPath}
    cmd := exec.CommandContext(ctx, bin, args...)
    binDir := filepath.Dir(bin)
    cmd.Dir = binDir
    env := os.Environ()
    env = append(env, "PATH="+binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
    espeak := filepath.Join(binDir, "espeak-ng-data")
    env = append(env, "ESPEAK_DATA_PATH="+espeak)
    cmd.Env = env
    cmd.Stdin = bytes.NewBufferString(text)
    return cmd, nil
}

func (s *Service) ensureVoiceModel(ctx context.Context, voice string) (string, error) {
    if err := os.MkdirAll(s.modelDir, 0o755); err != nil { return "", err }
    vdir := filepath.Join(s.modelDir, voice)
    if err := os.MkdirAll(vdir, 0o755); err != nil { return "", err }
    relBase, onnxFileName, jsonFileName := voiceRelativePaths(voice)
    if relBase == "" { return "", fmt.Errorf("unsupported voice: %s", voice) }
    onnxPath := filepath.Join(vdir, onnxFileName)
    jsonPath := filepath.Join(vdir, jsonFileName)

    if !fileExists(onnxPath) {
        if err := s.fetchVoiceAsset(vdir, relBase+"/"+onnxFileName, onnxPath, true); err != nil {
            return "", fmt.Errorf("failed to download voice model .onnx: %w", err)
        }
    }
    if !fileExists(jsonPath) {
        if err := s.fetchVoiceAsset(vdir, relBase+"/"+jsonFileName, jsonPath, false); err != nil {
            return "", fmt.Errorf("failed to download voice config .json: %w", err)
        }
    }
    return onnxPath, nil
}

// old piperBinaryPath replaced with recursive version at bottom

func piperDownloadURLs() ([]string, string) {
    switch runtime.GOOS {
    case "windows":
        return []string{
            "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip",
        }, "piper_windows_amd64.zip"
    case "darwin":
        if runtime.GOARCH == "arm64" {
            return []string{
                "https://raw.githubusercontent.com/pmbstyle/Alice/main/assets/binaries/piper-macos-arm64",
                "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_aarch64.tar.gz",
            }, "piper-macos-arm64"
        }
        return []string{
            "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_x64.tar.gz",
        }, "piper_macos_x64.tar.gz"
    case "linux":
        if runtime.GOARCH == "arm64" {
            return []string{
                "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz",
            }, "piper_linux_aarch64.tar.gz"
        }
        if runtime.GOARCH == "arm" {
            return []string{
                "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_armv7l.tar.gz",
            }, "piper_linux_armv7l.tar.gz"
        }
        return []string{
            "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz",
        }, "piper_linux_x86_64.tar.gz"
    default:
        return nil, ""
    }
}

// removed urlExists; direct URLs per platform/version

func voiceRelativePaths(voice string) (relBase, onnxFile, jsonFile string) {
    parts := strings.Split(voice, "-")
    if len(parts) < 3 { return "", "", "" }
    locale := parts[0]
    quality := parts[len(parts)-1]
    voiceName := strings.Join(parts[1:len(parts)-1], "-")
    if len(locale) < 2 { return "", "", "" }
    lang := strings.ToLower(locale[0:2])
    base := filepath.ToSlash(filepath.Join(lang, locale, voiceName, quality))
    return base, voice + ".onnx", voice + ".onnx.json"
}

func (s *Service) fetchVoiceAsset(vdir, relPath, dstPath string, allowGzip bool) error {
    bases := []string{
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/",
        "https://huggingface.co/rhasspy/piper-voices/raw/main/",
    }
    for _, b := range bases {
        u := b + relPath
        log.Printf("TTS: attempting %s", u)
        if err := downloadFileWithRetry(u, dstPath, 2, 120*time.Second); err == nil { return nil }
    }
    if allowGzip && strings.HasSuffix(strings.ToLower(dstPath), ".onnx") {
        tmp := dstPath + ".gz.part"
        for _, b := range bases {
            u := b + relPath + ".gz"
            log.Printf("TTS: attempting %s", u)
            if err := downloadFileWithRetry(u, tmp, 2, 180*time.Second); err == nil {
                if err := gunzipFile(tmp, dstPath); err == nil { _ = os.Remove(tmp); return nil }
                _ = os.Remove(tmp)
            }
        }
        _ = os.Remove(tmp)
    }
    return fmt.Errorf("asset not found for %s", relPath)
}

func extractZip(zipPath, outDir string) error {
    zr, err := zip.OpenReader(zipPath)
    if err != nil { return err }
    defer zr.Close()
    for _, f := range zr.File {
        if f.FileInfo().IsDir() { continue }
        rc, err := f.Open(); if err != nil { return err }
        defer rc.Close()
        fp := filepath.Join(outDir, f.Name)
        if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil { return err }
        out, err := os.Create(fp); if err != nil { return err }
        if _, err := io.Copy(out, rc); err != nil { out.Close(); return err }
        out.Close()
        if runtime.GOOS != "windows" { _ = os.Chmod(fp, 0o755) }
    }
    return nil
}

func extractTarGz(archivePath, outDir string) error {
    f, err := os.Open(archivePath)
    if err != nil { return err }
    defer f.Close()
    gz, err := gzip.NewReader(f)
    if err != nil { return err }
    defer gz.Close()
    tr := tar.NewReader(gz)
    for {
        hdr, err := tr.Next()
        if err == io.EOF { break }
        if err != nil { return err }
        if hdr.FileInfo().IsDir() { continue }
        target := filepath.Join(outDir, hdr.Name)
        if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil { return err }
        out, err := os.Create(target)
        if err != nil { return err }
        if _, err := io.Copy(out, tr); err != nil { out.Close(); return err }
        out.Close()
    }
    return nil
}

func gunzipFile(src, dst string) error {
    in, err := os.Open(src)
    if err != nil { return err }
    defer in.Close()
    gz, err := gzip.NewReader(in)
    if err != nil { return err }
    defer gz.Close()
    if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil { return err }
    out, err := os.Create(dst)
    if err != nil { return err }
    if _, err := io.Copy(out, gz); err != nil { out.Close(); return err }
    out.Close()
    return nil
}

func downloadFileWithRetry(url, dst string, retries int, timeout time.Duration) error {
    var last error
    for i := 0; i <= retries; i++ {
        if i > 0 { time.Sleep(time.Duration(i*i) * 500 * time.Millisecond) }
        if err := downloadFile(url, dst, timeout); err != nil { last = err; log.Printf("download failed (%s): %v", url, err); continue }
        return nil
    }
    return last
}

func downloadFile(url, dst string, timeout time.Duration) error {
    req, err := http.NewRequest(http.MethodGet, url, nil)
    if err != nil { return err }
    req.Header.Set("User-Agent", "GoLLMCore/1.0")
    req.Header.Set("Accept", "application/octet-stream")
    client := &http.Client{ Timeout: timeout }
    resp, err := client.Do(req)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode < 200 || resp.StatusCode >= 300 { return fmt.Errorf("bad status: %s", resp.Status) }
    tmp := dst + ".part"
    out, err := os.Create(tmp); if err != nil { return err }
    if _, err := io.Copy(out, resp.Body); err != nil { out.Close(); return err }
    out.Close()
    return os.Rename(tmp, dst)
}

// libEnv no longer used; env built per binary dir

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// Removed Python helpers; binary-only implementation

// Find piper binary recursively under binDir to handle archives with nested folders
func (s *Service) piperBinaryPath() string {
    names := map[string]bool{"piper": true, "piper.exe": true}
    var found string
    filepath.WalkDir(s.binDir, func(path string, d os.DirEntry, err error) error {
        if err != nil || d.IsDir() { return nil }
        base := filepath.Base(path)
        if names[base] {
            found = path
            return io.EOF // stop walk
        }
        return nil
    })
    return found
}
