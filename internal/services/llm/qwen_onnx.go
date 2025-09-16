package llm

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "net/http"
    "os"
    "path/filepath"
    "runtime"
    "strings"
    "time"

    ort "github.com/yalue/onnxruntime_go"
    "archive/zip"
    "compress/gzip"
    "archive/tar"
)

// QwenONNX provides in-process generation using an ONNX model.
type QwenONNX struct {
    modelDir string
    modelURL string
    tokURL   string

    modelPath string
    tokPath   string

    session *ort.DynamicAdvancedSession
    tok     *bpeTokenizer
    maxLen  int
}

func NewQwenONNX(modelDir, modelURL, tokURL string) (*QwenONNX, error) {
    q := &QwenONNX{modelDir: modelDir, modelURL: modelURL, tokURL: tokURL, maxLen: 512}
    if err := os.MkdirAll(modelDir, 0o755); err != nil { return nil, err }
    // Ensure ORT runtime
    libPath, err := ensureORTSharedLib()
    if err != nil { return nil, fmt.Errorf("ort lib: %w", err) }
    if !ort.IsInitialized() {
        ort.SetSharedLibraryPath(libPath)
        if err := ort.InitializeEnvironment(); err != nil { return nil, err }
    }
    // Download model + tokenizer
    q.modelPath = filepath.Join(modelDir, "model.onnx")
    if _, err := os.Stat(q.modelPath); err != nil {
        // Try FP32 first (as provided), then FP16 fallback
        if err := download(q.modelURL, q.modelPath, 10*time.Minute); err != nil {
            // fallback to fp16 path if fp32 failed and modelURL looks like fp32
            fp16 := strings.ReplaceAll(q.modelURL, "model_fp32.onnx", "model_fp16.onnx")
            if fp16 == q.modelURL { fp16 = strings.ReplaceAll(q.modelURL, "model.onnx", "onnx/model_fp16.onnx") }
            _ = download(fp16, q.modelPath, 10*time.Minute)
        }
        if _, e2 := os.Stat(q.modelPath); e2 != nil {
            return nil, fmt.Errorf("failed to download ONNX model (tried %s and fp16): %w", q.modelURL, e2)
        }
    }
    q.tokPath = filepath.Join(modelDir, "tokenizer.json")
    if _, err := os.Stat(q.tokPath); err != nil {
        if err := download(q.tokURL, q.tokPath, 2*time.Minute); err != nil { return nil, err }
    }
    // Load tokenizer
    tok, err := loadBPETokenizer(q.tokPath)
    if err != nil { return nil, fmt.Errorf("load tokenizer: %w", err) }
    q.tok = tok
    // Create session (environment may already be initialized by others)
    in := []string{"input_ids", "attention_mask", "position_ids"}
    out := []string{"logits"}
    sess, err := ort.NewDynamicAdvancedSession(q.modelPath, in, out, nil)
    if err != nil { return nil, err }
    q.session = sess
    return q, nil
}

// Generate greedy for now; returns the full text.
func (q *QwenONNX) Generate(ctx context.Context, prompt string, maxTokens int) (string, error) {
    if maxTokens <= 0 { maxTokens = 64 }
    ids := q.tok.Encode(prompt)
    if len(ids) > q.maxLen { ids = ids[len(ids)-q.maxLen:] }
    for t := 0; t < maxTokens; t++ {
        // Build attention mask
        mask := make([]int64, len(ids))
        for i := range ids { mask[i] = 1 }
        // Build position_ids: 0..len(ids)-1
        pos := make([]int64, len(ids))
        for i := range pos { pos[i] = int64(i) }
        // Create tensors
        inIDs, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), ids)
        if err != nil { return "", err }
        inMask, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), mask)
        if err != nil { return "", err }
        inPos, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), pos)
        if err != nil { return "", err }
        inputs := []ort.Value{inIDs, inMask, inPos}
        outputs := make([]ort.Value, 1) // logits auto-alloc
        if err := q.session.Run(inputs, outputs); err != nil { return "", err }
        // Read logits
        logitsVal := outputs[0]
        tens, ok := logitsVal.(*ort.Tensor[float32])
        if !ok { return "", errors.New("unexpected logits type") }
        data := tens.GetData()
        shape := tens.GetShape()
        if len(shape) != 3 { return "", fmt.Errorf("unexpected logits shape: %v", shape) }
        vocab := int(shape[2])
        // last token distribution at position len(ids)-1
        start := vocab * (int(shape[1]) - 1)
        nextID := argmax(data[start : start+vocab])
        // Append
        ids = append(ids, int64(nextID))
        if len(ids) > q.maxLen { ids = ids[1:] }
        // Stop if EOS known
        if q.tok.IsEOS(nextID) { break }
    }
    return q.tok.Decode(ids), nil
}

// GenerateWithCallback generates tokens greedily and calls cb with the partial decoded text at each step.
func (q *QwenONNX) GenerateWithCallback(ctx context.Context, prompt string, maxTokens int, cb func(s string)) (string, error) {
    if maxTokens <= 0 { maxTokens = 64 }
    ids := q.tok.Encode(prompt)
    if len(ids) > q.maxLen { ids = ids[len(ids)-q.maxLen:] }
    var outText string
    for t := 0; t < maxTokens; t++ {
        select { case <-ctx.Done(): return outText, ctx.Err(); default: }
        mask := make([]int64, len(ids))
        for i := range mask { mask[i] = 1 }
        pos := make([]int64, len(ids))
        for i := range pos { pos[i] = int64(i) }
        inIDs, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), ids)
        if err != nil { return outText, err }
        inMask, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), mask)
        if err != nil { return outText, err }
        inPos, err := ort.NewTensor[int64](ort.NewShape(1, int64(len(ids))), pos)
        if err != nil { return outText, err }
        outputs := make([]ort.Value, 1)
        if err := q.session.Run([]ort.Value{inIDs, inMask, inPos}, outputs); err != nil { return outText, err }
        tens, ok := outputs[0].(*ort.Tensor[float32])
        if !ok { return outText, errors.New("unexpected logits type") }
        data := tens.GetData()
        shape := tens.GetShape()
        if len(shape) != 3 { return outText, fmt.Errorf("unexpected logits shape: %v", shape) }
        vocab := int(shape[2])
        start := vocab * (int(shape[1]) - 1)
        nextID := argmax(data[start : start+vocab])
        ids = append(ids, int64(nextID))
        if q.tok.IsEOS(nextID) { break }
        outText = q.tok.Decode(ids)
        if cb != nil { cb(outText) }
    }
    return outText, nil
}

// Helpers

func download(u, dst string, timeout time.Duration) error {
    req, _ := http.NewRequest(http.MethodGet, u, nil)
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

func argmax(v []float32) int {
    max := float32(-1e30)
    idx := 0
    for i, x := range v {
        if x > max { max = x; idx = i }
    }
    return idx
}

// Minimal BPE tokenizer for tokenizer.json (supports Encode/Decode for demo)
type bpeTokenizer struct {
    vocab       map[string]int
    id2token    []string
    eosTokenID  int
}

func loadBPETokenizer(path string) (*bpeTokenizer, error) {
    b, err := os.ReadFile(path)
    if err != nil { return nil, err }
    var t struct {
        Model struct {
            Type  string            `json:"type"`
            Vocab map[string]int    `json:"vocab"`
        } `json:"model"`
        AddedTokens []struct {
            ID    int    `json:"id"`
            Content string `json:"content"`
            Special bool   `json:"special"`
        } `json:"added_tokens"`
    }
    if err := json.Unmarshal(b, &t); err != nil { return nil, err }
    id2 := make([]string, len(t.Model.Vocab))
    for tok, id := range t.Model.Vocab {
        if id >= 0 && id < len(id2) { id2[id] = tok }
    }
    eos := -1
    for _, at := range t.AddedTokens {
        if strings.Contains(strings.ToLower(at.Content), "eos") || at.Content == "<|endoftext|>" {
            eos = at.ID
            break
        }
    }
    return &bpeTokenizer{vocab: t.Model.Vocab, id2token: id2, eosTokenID: eos}, nil
}

func (t *bpeTokenizer) Encode(s string) []int64 {
    // Extremely naive whitespace-based encoding using vocab; real BPE is TODO.
    parts := strings.Fields(s)
    var ids []int64
    for _, p := range parts {
        if id, ok := t.vocab[p]; ok { ids = append(ids, int64(id)) } else { ids = append(ids, 0) }
    }
    if len(ids) == 0 { ids = []int64{0} }
    return ids
}

func (t *bpeTokenizer) Decode(ids []int64) string {
    var parts []string
    for _, id := range ids {
        i := int(id)
        if i >= 0 && i < len(t.id2token) && t.id2token[i] != "" { parts = append(parts, t.id2token[i]) }
    }
    return strings.Join(parts, " ")
}

func (t *bpeTokenizer) IsEOS(id int) bool {
    return t.eosTokenID >= 0 && id == t.eosTokenID
}

// --- ORT shared library downloader (duplicated minimal variant) ---

func ensureORTSharedLib() (string, error) {
    baseDir := filepath.Join(os.TempDir(), "onnxruntime")
    ortVersion := "v1.22.0"
    versionDir := filepath.Join(baseDir, ortVersion)
    if err := os.MkdirAll(versionDir, 0o755); err != nil { return "", err }
    switch runtime.GOOS {
    case "windows":
        dll := filepath.Join(versionDir, "onnxruntime.dll")
        if fileExists(dll) { return dll, nil }
        url := "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-win-x64-"+strings.TrimPrefix(ortVersion, "v")+".zip"
        zipPath := filepath.Join(versionDir, "ort.zip")
        if err := download(url, zipPath, 4*time.Minute); err != nil { return "", err }
        if err := unzipSelect(zipPath, versionDir, []string{"onnxruntime.dll"}); err != nil { return "", err }
        return dll, nil
    case "darwin":
        dylib := filepath.Join(versionDir, "libonnxruntime.dylib")
        if fileExists(dylib) { return dylib, nil }
        urls := []string{
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-universal2-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-arm64-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-x64-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
        }
        tgz := filepath.Join(versionDir, "ort.tgz")
        for _, u := range urls {
            if err := download(u, tgz, 4*time.Minute); err == nil { break }
        }
        if err := untarSelect(tgz, versionDir, []string{"libonnxruntime.dylib"}); err != nil { return "", err }
        return dylib, nil
    case "linux":
        so := filepath.Join(versionDir, "libonnxruntime.so")
        if fileExists(so) { return so, nil }
        url := "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-linux-x64-"+strings.TrimPrefix(ortVersion, "v")+".tgz"
        tgz := filepath.Join(versionDir, "ort.tgz")
        if err := download(url, tgz, 4*time.Minute); err != nil { return "", err }
        if err := untarSelect(tgz, versionDir, []string{"libonnxruntime.so"}); err != nil { return "", err }
        return so, nil
    default:
        return "", fmt.Errorf("unsupported platform: %s", runtime.GOOS)
    }
}

func unzipSelect(zipPath, dst string, names []string) error {
    set := map[string]bool{}
    for _, n := range names { set[n] = true }
    r, err := zip.OpenReader(zipPath)
    if err != nil { return err }
    defer r.Close()
    for _, f := range r.File {
        base := filepath.Base(f.Name)
        if !set[base] { continue }
        rc, err := f.Open(); if err != nil { return err }
        defer rc.Close()
        out := filepath.Join(dst, base)
        fo, err := os.Create(out); if err != nil { return err }
        if _, err := io.Copy(fo, rc); err != nil { fo.Close(); return err }
        fo.Close()
        if runtime.GOOS != "windows" { _ = os.Chmod(out, 0o755) }
    }
    return nil
}

func untarSelect(tgzPath, dstDir string, names []string) error {
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
        delete(set, base)
        if len(set) == 0 { break }
    }
    if len(set) > 0 { return fmt.Errorf("missing files: %v", names) }
    return nil
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }
