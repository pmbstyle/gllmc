package stt

import (
    "archive/zip"
    "bufio"
    "context"
    "errors"
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

type STTService struct {
    binDir    string
    modelDir  string
}

func New(binDir, modelDir string) *STTService {
    return &STTService{binDir: binDir, modelDir: modelDir}
}

// TranscribeFile performs a non-streaming transcription and returns the final text.
func (s *STTService) TranscribeFile(ctx context.Context, audioPath, modelSize string) (string, error) {
    if err := s.ensureWhisperInstalled(ctx); err != nil { return "", err }
    modelPath, err := s.ensureWhisperModel(ctx, modelSize)
    if err != nil { return "", err }

    bin, err := s.pickWhisperBinary()
    if err != nil { return "", err }

    outPrefix := filepath.Join(os.TempDir(), fmt.Sprintf("whisper_out_%d", time.Now().UnixNano()))
    args := []string{"-m", modelPath, "-f", audioPath, "-otxt", "-of", outPrefix, "-nt"}
    cmd := exec.CommandContext(ctx, bin, args...)
    cmd.Dir = s.binDir
    cmd.Env = append(os.Environ(), s.libEnv()...)
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    if err := cmd.Run(); err != nil {
        return "", fmt.Errorf("whisper execution failed: %w", err)
    }
    txtPath := outPrefix + ".txt"
    data, err := os.ReadFile(txtPath)
    if err != nil { return "", fmt.Errorf("reading transcript: %w", err) }
    _ = os.Remove(txtPath)
    return string(data), nil
}

// TranscribeFileStream runs whisper and streams its stdout lines.
func (s *STTService) TranscribeFileStream(ctx context.Context, audioPath, modelSize string) (<-chan string, <-chan error) {
    lines := make(chan string)
    errs := make(chan error, 1)
    go func() {
        defer close(lines)
        defer close(errs)
        if err := s.ensureWhisperInstalled(ctx); err != nil { errs <- err; return }
        modelPath, err := s.ensureWhisperModel(ctx, modelSize)
        if err != nil { errs <- err; return }
        bin, err := s.pickWhisperBinary()
        if err != nil { errs <- err; return }

        args := []string{"-m", modelPath, "-f", audioPath, "-nt"}
        cmd := exec.CommandContext(ctx, bin, args...)
        cmd.Dir = s.binDir
        cmd.Env = append(os.Environ(), s.libEnv()...)
        stdout, _ := cmd.StdoutPipe()
        stderr, _ := cmd.StderrPipe()
        if err := cmd.Start(); err != nil { errs <- err; return }

        scan := bufio.NewScanner(io.MultiReader(stdout, stderr))
        for scan.Scan() {
            select {
            case <-ctx.Done():
                _ = cmd.Process.Kill()
                return
            default:
            }
            line := strings.TrimSpace(scan.Text())
            if line == "" { continue }
            lines <- line
        }
        if err := scan.Err(); err != nil { errs <- err }
        _ = cmd.Wait()
    }()
    return lines, errs
}

// ----- Installation helpers -----

func (s *STTService) ensureWhisperInstalled(ctx context.Context) error {
    if err := os.MkdirAll(s.binDir, 0o755); err != nil { return err }
    // If any known binary exists, return
    if _, err := s.pickWhisperBinary(); err == nil { return nil }
    // Download and extract
    return s.downloadWhisperBinary(ctx)
}

func (s *STTService) pickWhisperBinary() (string, error) {
    candidates := []string{"whisper", "whisper-cli", "whisper-command", "main", "whisper.exe", "main.exe", "whisper-cli.exe", "whisper-command.exe"}
    for _, name := range candidates {
        p := filepath.Join(s.binDir, name)
        if info, err := os.Stat(p); err == nil && !info.IsDir() {
            return p, nil
        }
    }
    return "", errors.New("whisper binary not found")
}

func (s *STTService) ensureWhisperModel(ctx context.Context, size string) (string, error) {
    if err := os.MkdirAll(s.modelDir, 0o755); err != nil { return "", err }
    size = strings.ToLower(size)
    urls, file := whisperModelURLs(size)
    if len(urls) == 0 {
        return "", fmt.Errorf("unsupported whisper model size: %s", size)
    }
    dst := filepath.Join(s.modelDir, file)
    if _, err := os.Stat(dst); err == nil { return dst, nil }
    log.Printf("Downloading Whisper model %s...", size)
    var last error
    for i, u := range urls {
        log.Printf("Attempt %d/%d: %s", i+1, len(urls), u)
        if err := downloadFileWithRetry(u, dst, 2, 60*time.Second); err != nil {
            last = err
            continue
        }
        last = nil
        break
    }
    if last != nil { return "", last }
    return dst, nil
}

func whisperModelURLs(size string) ([]string, string) {
    // Try reliable public sources (Hugging Face mirrors). Order matters.
    // Primary: ggerganov/whisper.cpp repo model files in main branch.
    // Example: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
    file := ""
    switch size {
    case "tiny": file = "ggml-tiny.bin"
    case "base": file = "ggml-base.bin"
    case "small": file = "ggml-small.bin"
    case "medium": file = "ggml-medium.bin"
    case "large", "large-v2": file = "ggml-large-v2.bin"
    case "large-v3": file = "ggml-large-v3.bin"
    }
    if file == "" { return nil, "" }
    return []string{
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + file,
        // Secondary mirror (some environments mirror HF under different hostnames, leave as fallback template)
        // Add more mirrors here if needed.
    }, file
}

// downloadWhisperBinary downloads the whisper.cpp binary for the current platform
func (s *STTService) downloadWhisperBinary(ctx context.Context) error {
    var downloadURLs []string
    var fileName string

    switch runtime.GOOS {
    case "windows":
        if runtime.GOARCH == "amd64" || runtime.GOARCH == "x86_64" {
            downloadURLs = []string{
                "https://aliceai.ca/app_assets/whisper/whisper-windows.zip",
            }
            fileName = "whisper-windows.zip"
        } else {
            return fmt.Errorf("unsupported Windows architecture: %s", runtime.GOARCH)
        }
    case "darwin":
        if runtime.GOARCH == "arm64" {
            downloadURLs = []string{
                "https://aliceai.ca/app_assets/whisper/whisper-macos-arm64.zip",
            }
            fileName = "whisper-macos-arm64.zip"
        } else {
            downloadURLs = []string{
                "https://aliceai.ca/app_assets/whisper/whisper-macos-x64.zip",
            }
            fileName = "whisper-macos-x64.zip"
        }
    case "linux":
        if runtime.GOARCH == "amd64" || runtime.GOARCH == "x86_64" {
            downloadURLs = []string{
                "https://aliceai.ca/app_assets/whisper/whisper-linux-x64.zip",
            }
            fileName = "whisper-linux-x64.zip"
        } else {
            return fmt.Errorf("unsupported Linux architecture: %s", runtime.GOARCH)
        }
    default:
        return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
    }

    log.Printf("Downloading Whisper binary for %s/%s", runtime.GOOS, runtime.GOARCH)

    if err := os.MkdirAll(s.binDir, 0o755); err != nil {
        return fmt.Errorf("failed to create bin directory: %w", err)
    }

    downloadPath := filepath.Join(s.binDir, fileName)
    var lastErr error

    for i, downloadURL := range downloadURLs {
        log.Printf("Attempting binary download from source %d/%d: %s", i+1, len(downloadURLs), downloadURL)
        if err := downloadFileWithRetry(downloadURL, downloadPath, 2, 30*time.Second); err != nil {
            lastErr = err
            log.Printf("Binary download source %d failed: %v", i+1, err)
            continue
        }
        log.Printf("Binary download successful from source %d", i+1)
        break
    }
    if _, err := os.Stat(downloadPath); err != nil {
        return fmt.Errorf("failed to download whisper binary from any source: %w", lastErr)
    }

    defer os.Remove(downloadPath)
    if err := s.extractWhisperBinary(downloadPath); err != nil {
        return fmt.Errorf("failed to extract whisper binary: %w", err)
    }

    log.Printf("Whisper binary installed successfully")
    return nil
}

// extractWhisperBinary extracts the whisper binary from the downloaded zip
func (s *STTService) extractWhisperBinary(zipPath string) error {
    reader, err := zip.OpenReader(zipPath)
    if err != nil { return err }
    defer reader.Close()

    log.Printf("Extracting whisper binary from: %s", zipPath)

    extractedCount := 0
    whisperBinaries := []string{"whisper-cli.exe", "whisper-command.exe", "main.exe", "whisper.exe"}
    requiredDLLs := []string{"ggml-base.dll", "ggml-cpu.dll", "ggml.dll", "whisper.dll", "SDL2.dll"}
    requiredDylibs := []string{}

    if runtime.GOOS != "windows" {
        whisperBinaries = []string{"whisper-cli", "whisper-command", "main", "whisper"}
        requiredDLLs = []string{}
        if runtime.GOOS == "darwin" {
            requiredDylibs = []string{"libggml.dylib", "libggml-base.dylib", "libggml-blas.dylib",
                "libggml-cpu.dylib", "libggml-metal.dylib", "libwhisper.dylib",
                "libwhisper.1.dylib", "libwhisper.1.7.6.dylib"}
        } else if runtime.GOOS == "linux" {
            requiredDLLs = []string{"libggml.so", "libggml-base.so", "libggml-cpu.so",
                "libwhisper.so", "libwhisper.so.1", "libwhisper.so.1.7.6"}
        }
    }

    for _, f := range reader.File {
        if f.FileInfo().IsDir() { continue }
        lower := strings.ToLower(filepath.Base(f.Name))

        for _, wanted := range whisperBinaries {
            if lower == strings.ToLower(wanted) {
                outputPath := filepath.Join(s.binDir, wanted)
                if err := extractSingleFile(f, outputPath); err != nil {
                    log.Printf("Failed to extract %s: %v", wanted, err)
                    continue
                }
                if runtime.GOOS != "windows" {
                    _ = os.Chmod(outputPath, 0o755)
                }
                extractedCount++
                break
            }
        }

        for _, wanted := range requiredDLLs {
            if lower == strings.ToLower(wanted) {
                outputPath := filepath.Join(s.binDir, wanted)
                if err := extractSingleFile(f, outputPath); err != nil {
                    log.Printf("Failed to extract lib %s: %v", wanted, err)
                    continue
                }
                extractedCount++
                break
            }
        }

        for _, wanted := range requiredDylibs {
            if lower == strings.ToLower(wanted) {
                if err := os.MkdirAll(filepath.Join(s.binDir, "libinternal"), 0o755); err != nil {
                    log.Printf("Failed to create libinternal directory: %v", err)
                    continue
                }
                outputPath := filepath.Join(s.binDir, "libinternal", wanted)
                if err := extractSingleFile(f, outputPath); err != nil {
                    log.Printf("Failed to extract dylib %s: %v", wanted, err)
                    continue
                }
                extractedCount++
                break
            }
        }
    }

    if extractedCount == 0 {
        return fmt.Errorf("no suitable whisper binary found in archive")
    }

    log.Printf("Successfully extracted %d whisper files", extractedCount)
    return nil
}

func extractSingleFile(f *zip.File, outputPath string) error {
    rc, err := f.Open(); if err != nil { return err }
    defer rc.Close()
    if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil { return err }
    out, err := os.Create(outputPath); if err != nil { return err }
    defer out.Close()
    _, err = io.Copy(out, rc)
    return err
}

func (s *STTService) libEnv() []string {
    // Ensure local libs and binaries are discoverable by the OS loader
    libDir := filepath.Join(s.binDir, "libinternal")
    switch runtime.GOOS {
    case "darwin":
        // DYLD_LIBRARY_PATH for macOS dynamic libs
        return []string{
            "DYLD_LIBRARY_PATH=" + s.binDir + string(os.PathListSeparator) + libDir,
            "PATH=" + s.binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
        }
    case "linux":
        return []string{
            "LD_LIBRARY_PATH=" + s.binDir + string(os.PathListSeparator) + libDir,
            "PATH=" + s.binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
        }
    case "windows":
        return []string{
            "PATH=" + s.binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
        }
    default:
        return nil
    }
}

// download with retry and timeout
func downloadFileWithRetry(url, dst string, retries int, timeout time.Duration) error {
    var last error
    for i := 0; i <= retries; i++ {
        if i > 0 {
            backoff := time.Duration(i*i) * 500 * time.Millisecond
            time.Sleep(backoff)
        }
        if err := downloadFile(url, dst, timeout); err != nil {
            last = err
            log.Printf("download failed (attempt %d/%d): %v", i+1, retries+1, err)
            continue
        }
        return nil
    }
    return last
}

func downloadFile(url, dst string, timeout time.Duration) error {
    req, err := http.NewRequest(http.MethodGet, url, nil)
    if err != nil { return err }
    req.Header.Set("User-Agent", "GoLLMCore/1.0 (+https://localhost)")
    req.Header.Set("Accept", "application/octet-stream")
    client := &http.Client{ Timeout: timeout }
    resp, err := client.Do(req)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode < 200 || resp.StatusCode >= 300 {
        return fmt.Errorf("bad status: %s", resp.Status)
    }
    tmp := dst + ".part"
    out, err := os.Create(tmp)
    if err != nil { return err }
    if _, err := io.Copy(out, resp.Body); err != nil { out.Close(); return err }
    out.Close()
    return os.Rename(tmp, dst)
}
