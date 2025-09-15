package embeddings

import (
    "archive/tar"
    "archive/zip"
    "compress/gzip"
    "context"
    "errors"
    "fmt"
    "io"
    "log"
    "math"
    "net/http"
    "os"
    "path/filepath"
    "runtime"
    "sort"
    "strings"
    "unicode"
    "time"

    ort "github.com/yalue/onnxruntime_go"
)

// Real MiniLM L6-v2 ONNX-backed embedder using onnxruntime_go (no Python).
// Downloads model/vocab and ONNX Runtime shared lib on demand.

type miniLMOnnx struct {
    modelDir   string
    modelPath  string
    vocabPath  string
    session    *ort.DynamicAdvancedSession
    tokenizer  *wordPiece
    maxLen     int
}

// NewMiniLM returns a real ONNX-backed embeddings service.
func NewMiniLM(modelDir string) (Service, error) {
    m := &miniLMOnnx{modelDir: modelDir, maxLen: 128}
    if err := m.ensureRuntimeAndModel(); err != nil { return nil, err }
    if err := m.initSession(); err != nil { return nil, err }
    return m, nil
}

func (m *miniLMOnnx) Embed(ctx context.Context, inputs []string) ([][]float32, string, error) {
    if len(inputs) == 0 { return nil, "all-MiniLM-L6-v2", nil }
    // Tokenize
    ids, masks := m.batchTokenize(inputs, m.maxLen)
    // Create tensors
    bsz := len(inputs)
    seq := m.maxLen
    inputIDs := make([]int64, bsz*seq)
    attMask := make([]int64, bsz*seq)
    for i := 0; i < bsz; i++ {
        copy(inputIDs[i*seq:(i+1)*seq], ids[i])
        copy(attMask[i*seq:(i+1)*seq], masks[i])
    }
    in1, err := ort.NewTensor[int64](ort.NewShape(int64(bsz), int64(seq)), inputIDs)
    if err != nil { return nil, "all-MiniLM-L6-v2", err }
    in2, err := ort.NewTensor[int64](ort.NewShape(int64(bsz), int64(seq)), attMask)
    if err != nil { return nil, "all-MiniLM-L6-v2", err }

    // Try common input names
    // Build inputs slice in the order of input names
    // token_type_ids (all zeros)
    ttiData := make([]int64, bsz*seq)
    tti, err := ort.NewTensor[int64](ort.NewShape(int64(bsz), int64(seq)), ttiData)
    if err != nil { return nil, "all-MiniLM-L6-v2", err }
    inputsVals := []ort.Value{in1, in2, tti}
    // Prepare outputs slice matching output names (auto-alloc by leaving nil)
    outputsVals := make([]ort.Value, 1)
    if err := m.session.Run(inputsVals, outputsVals); err != nil { return nil, "all-MiniLM-L6-v2", err }
    // Expect single output last_hidden_state
    out0 := outputsVals[0]
    t, ok := out0.(*ort.Tensor[float32])
    if !ok { return nil, "all-MiniLM-L6-v2", errors.New("unexpected output type") }
    dataF := t.GetData()
    shape := t.GetShape()
    if len(shape) != 3 { return nil, "all-MiniLM-L6-v2", fmt.Errorf("unexpected output shape: %v", shape) }
    s := int(shape[1])
    h := int(shape[2])
    // mean pooling with attention mask
    out := make([][]float32, bsz)
    for i := 0; i < bsz; i++ {
        start := i * s * h
        vec := make([]float32, h)
        var count float32
        for j := 0; j < s; j++ {
            if attMask[i*seq+j] == 0 { continue }
            base := start + j*h
            for d := 0; d < h; d++ { vec[d] += dataF[base+d] }
            count += 1
        }
        if count > 0 {
            inv := 1.0 / count
            var norm float64
            for d := 0; d < h; d++ { vec[d] *= float32(inv); norm += float64(vec[d]*vec[d]) }
            if norm > 0 {
                invn := float32(1.0 / (math.Sqrt(norm)))
                for d := 0; d < h; d++ { vec[d] *= invn }
            }
        }
        out[i] = vec
    }
    return out, "all-MiniLM-L6-v2", nil
}

// -------- Session/model/runtime management --------

func (m *miniLMOnnx) ensureRuntimeAndModel() error {
    // Ensure directories
    if err := os.MkdirAll(m.modelDir, 0o755); err != nil { return err }
    // Download ORT shared library
    libPath, err := ensureORTSharedLib()
    if err != nil { return fmt.Errorf("onnxruntime lib: %w", err) }
    // Point onnxruntime_go to the shared library
    ort.SetSharedLibraryPath(libPath)

    // Download model, tokenizer and vocab
    m.modelPath, m.vocabPath, err = ensureMiniLMModel(m.modelDir)
    if err != nil { return err }
    // Load vocab-based WordPiece tokenizer (uncased)
    tk, err := loadWordPiece(m.vocabPath)
    if err != nil { return err }
    m.tokenizer = tk
    return nil
}

func (m *miniLMOnnx) initSession() error {
    if err := ort.InitializeEnvironment(); err != nil { return err }
    // Input and output names we expect
    inNames := []string{"input_ids", "attention_mask", "token_type_ids"}
    outNames := []string{"last_hidden_state"}
    sess, err := ort.NewDynamicAdvancedSession(m.modelPath, inNames, outNames, nil)
    if err != nil { return err }
    m.session = sess
    return nil
}

// -------- Tokenization (minimal WordPiece, uncased) --------

func (m *miniLMOnnx) batchTokenize(texts []string, maxLen int) ([][]int64, [][]int64) {
    ids := make([][]int64, len(texts))
    masks := make([][]int64, len(texts))
    for i, t := range texts {
        ii, mm := m.encode(t, maxLen)
        ids[i], masks[i] = ii, mm
    }
    return ids, masks
}

func (m *miniLMOnnx) encode(text string, maxLen int) ([]int64, []int64) {
    toks := basicTokens(text)
    var pieces []int
    for _, w := range toks {
        pieces = append(pieces, m.tokenizer.tokenizeWord(w)...)
    }
    seq := []int{m.tokenizer.clsID}
    seq = append(seq, pieces...)
    seq = append(seq, m.tokenizer.sepID)
    if len(seq) > maxLen { seq = seq[:maxLen] }
    ids := make([]int64, maxLen)
    mask := make([]int64, maxLen)
    for i, v := range seq { ids[i] = int64(v); mask[i] = 1 }
    for i := len(seq); i < maxLen; i++ { ids[i] = 0 }
    return ids, mask
}

// -------- Downloads --------

func ensureMiniLMModel(dir string) (modelPath, vocabPath string, err error) {
    modelPath = filepath.Join(dir, "model.onnx")
    vocabPath = filepath.Join(dir, "vocab.txt")
    if _, e := os.Stat(modelPath); e != nil {
        urls := []string{
            // ONNX export of MiniLM (Transformers.js format)
            "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            // Alternate path (some mirrors place model at root)
            "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/model.onnx",
            // Community ONNX mirrors
            "https://huggingface.co/onnx-community/all-MiniLM-L6-v2/resolve/main/model.onnx",
        }
        if err = tryDownload(urls, modelPath, 3, 180*time.Second); err != nil { return "", "", err }
    }
    if _, e := os.Stat(vocabPath); e != nil {
        urls := []string{
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
        }
        if err = tryDownload(urls, vocabPath, 3, 60*time.Second); err != nil { return "", "", err }
    }
    return modelPath, vocabPath, nil
}

func ensureORTSharedLib() (string, error) {
    baseDir := filepath.Join(os.TempDir(), "onnxruntime")
    ortVersion := "v1.22.0"
    versionDir := filepath.Join(baseDir, ortVersion)
    if err := os.MkdirAll(versionDir, 0o755); err != nil { return "", err }
    switch runtime.GOOS {
    case "windows":
        dll := filepath.Join(versionDir, "onnxruntime.dll")
        if fileExists(dll) { return dll, nil }
        urls := []string{
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-win-x64-"+strings.TrimPrefix(ortVersion, "v")+".zip",
        }
        zipPath := filepath.Join(versionDir, "ort.zip")
        if err := tryDownload(urls, zipPath, 3, 240*time.Second); err != nil { return "", err }
        if err := unzipOne(zipPath, versionDir, "onnxruntime.dll"); err != nil { return "", err }
        return dll, nil
    case "darwin":
        dylib := filepath.Join(versionDir, "libonnxruntime.dylib")
        if fileExists(dylib) { return dylib, nil }
        // arm64 vs x64 both extract libonnxruntime.dylib
        urls := []string{
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-universal2-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-arm64-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-osx-x64-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
        }
        tgz := filepath.Join(versionDir, "ort.tgz")
        if err := tryDownload(urls, tgz, 3, 240*time.Second); err != nil { return "", err }
        if err := untarSelect(tgz, versionDir, []string{"libonnxruntime.dylib"}); err != nil { return "", err }
        return dylib, nil
    case "linux":
        so := filepath.Join(versionDir, "libonnxruntime.so")
        if fileExists(so) { return so, nil }
        urls := []string{
            "https://github.com/microsoft/onnxruntime/releases/download/"+ortVersion+"/onnxruntime-linux-x64-"+strings.TrimPrefix(ortVersion, "v")+".tgz",
        }
        tgz := filepath.Join(versionDir, "ort.tgz")
        if err := tryDownload(urls, tgz, 3, 240*time.Second); err != nil { return "", err }
        if err := untarSelect(tgz, versionDir, []string{"libonnxruntime.so"}); err != nil { return "", err }
        return so, nil
    default:
        return "", fmt.Errorf("unsupported platform for ORT: %s", runtime.GOOS)
    }
}

func tryDownload(urls []string, dst string, retries int, timeout time.Duration) error {
    var last error
    for i, u := range urls {
        log.Printf("Downloading: %s (%d/%d)", u, i+1, len(urls))
        if err := downloadFile(u, dst, timeout); err != nil {
            last = err
            continue
        }
        return nil
    }
    return last
}

func downloadFile(url, dst string, timeout time.Duration) error {
    req, err := http.NewRequest(http.MethodGet, url, nil)
    if err != nil { return err }
    req.Header.Set("User-Agent", "GoLLMCore/1.0")
    client := &http.Client{ Timeout: timeout }
    resp, err := client.Do(req)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode < 200 || resp.StatusCode >= 300 { return fmt.Errorf("bad status: %s", resp.Status) }
    tmp := dst + ".part"
    out, err := os.Create(tmp)
    if err != nil { return err }
    if _, err := io.Copy(out, resp.Body); err != nil { out.Close(); return err }
    out.Close()
    return os.Rename(tmp, dst)
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// unzipOne extracts a specific file from a zip archive to dstDir
func unzipOne(zipPath, dstDir, wanted string) error {
    r, err := zip.OpenReader(zipPath)
    if err != nil { return err }
    defer r.Close()
    for _, f := range r.File {
        if filepath.Base(f.Name) == wanted {
            rc, err := f.Open(); if err != nil { return err }
            defer rc.Close()
            out := filepath.Join(dstDir, wanted)
            fo, err := os.Create(out); if err != nil { return err }
            if _, err := io.Copy(fo, rc); err != nil { fo.Close(); return err }
            fo.Close()
            if runtime.GOOS != "windows" { _ = os.Chmod(out, 0o755) }
            return nil
        }
    }
    return fmt.Errorf("file %s not found in zip", wanted)
}

// untarSelect extracts specific files from a .tgz into dstDir
func untarSelect(tgzPath, dstDir string, names []string) error {
    set := make(map[string]bool)
    for _, n := range names { set[n] = true }
    f, err := os.Open(tgzPath); if err != nil { return err }
    defer f.Close()
    gz, err := gzip.NewReader(f); if err != nil { return err }
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
    if len(set) > 0 { return fmt.Errorf("missing files: %v", keys(set)) }
    return nil
}

func keys(m map[string]bool) []string { ks := make([]string, 0, len(m)); for k := range m { ks = append(ks, k) }; sort.Strings(ks); return ks }

// -------- WordPiece tokenizer (uncased) --------

type wordPiece struct {
    vocab map[string]int
    unkID int
    clsID int
    sepID int
    padID int
}

func loadWordPiece(path string) (*wordPiece, error) {
    b, err := os.ReadFile(path)
    if err != nil { return nil, err }
    lines := strings.Split(string(b), "\n")
    vp := make(map[string]int, len(lines))
    for i, line := range lines {
        tok := strings.TrimSpace(line)
        if tok == "" { continue }
        if _, ok := vp[tok]; !ok { vp[tok] = i }
    }
    get := func(tok string, def int) int { if id, ok := vp[tok]; ok { return id }; return def }
    return &wordPiece{
        vocab: vp,
        unkID: get("[UNK]", 100),
        clsID: get("[CLS]", 101),
        sepID: get("[SEP]", 102),
        padID: get("[PAD]", 0),
    }, nil
}

func basicTokens(s string) []string {
    s = strings.ToLower(s)
    var out []string
    var b strings.Builder
    flush := func() { if b.Len() > 0 { out = append(out, b.String()); b.Reset() } }
    for _, r := range s {
        if unicode.IsLetter(r) || unicode.IsDigit(r) {
            b.WriteRune(r)
        } else {
            flush()
        }
    }
    flush()
    return out
}

func (w *wordPiece) tokenizeWord(tok string) []int {
    if tok == "" { return nil }
    var out []int
    for len(tok) > 0 {
        end := len(tok)
        var cur string
        var id int
        found := false
        for end > 0 {
            sub := tok[:end]
            candidate := sub
            if len(out) > 0 { candidate = "##" + sub }
            if vid, ok := w.vocab[candidate]; ok {
                cur = candidate; id = vid; found = true; break
            }
            end--
        }
        if !found {
            out = append(out, w.unkID)
            break
        }
        out = append(out, id)
        if strings.HasPrefix(cur, "##") { cur = cur[2:] }
        tok = tok[len(cur):]
    }
    return out
}
