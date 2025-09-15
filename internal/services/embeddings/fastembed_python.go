package embeddings

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "strings"
)

type fastEmbedPy struct {
    modelName string
    modelDir  string
    workDir   string
    venvDir   string
    pyExe     string
}

func newFastEmbedPy(cfg Config) (Service, error) {
    fe := &fastEmbedPy{
        modelName: cfg.ModelName,
        modelDir:  cfg.ModelDir,
        workDir:   cfg.WorkDir,
        venvDir:   filepath.Join(cfg.WorkDir, "venv"),
    }
    if fe.modelName == "" {
        fe.modelName = "sentence-transformers/all-MiniLM-L6-v2"
    }
    if err := os.MkdirAll(fe.modelDir, 0o755); err != nil { return nil, err }
    if err := os.MkdirAll(fe.workDir, 0o755); err != nil { return nil, err }
    if err := fe.ensurePython(); err != nil { return nil, err }
    if err := fe.ensureVenv(); err != nil { return nil, err }
    if err := fe.ensureFastEmbedInstalled(); err != nil { return nil, err }
    return fe, nil
}

func (f *fastEmbedPy) Embed(ctx context.Context, inputs []string) ([][]float32, string, error) {
    if len(inputs) == 0 { return nil, f.modelName, nil }
    // Construct a Python script which reads JSON list from stdin and prints embeddings as JSON
    script := strings.TrimSpace(`
import sys, json
from fastembed import TextEmbedding
model_name = sys.argv[1]
cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
te = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
docs = json.load(sys.stdin)
vecs = []
for v in te.embed(docs, batch_size=256):
    vecs.append(v.tolist())
print(json.dumps({"embeddings": vecs, "model": model_name}))
`)
    py := f.pythonExec()
    if py == "" { return nil, f.modelName, errors.New("python not available") }
    args := []string{"-c", script, f.modelName, f.modelDir}
    cmd := exec.CommandContext(ctx, py, args...)
    cmd.Env = f.envForVenv()
    cmd.Dir = f.workDir
    // supply inputs via stdin
    b, _ := json.Marshal(inputs)
    cmd.Stdin = bytes.NewReader(b)
    var out bytes.Buffer
    var errb bytes.Buffer
    cmd.Stdout = &out
    cmd.Stderr = &errb
    if err := cmd.Run(); err != nil {
        return nil, f.modelName, fmt.Errorf("fastembed python failed: %v: %s", err, errb.String())
    }
    var resp struct {
        Embeddings [][]float32 `json:"embeddings"`
        Model      string      `json:"model"`
    }
    if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
        return nil, f.modelName, fmt.Errorf("parse fastembed output: %w; raw=%s", err, out.String())
    }
    return resp.Embeddings, resp.Model, nil
}

func (f *fastEmbedPy) ensurePython() error {
    // Prefer venv python if present or any discoverable python
    if exe := f.pythonExec(); exe != "" {
        f.pyExe = exe
        return nil
    }
    // Probe system Python
    cands := []string{"python3", "python"}
    if runtime.GOOS == "windows" {
        cands = append([]string{"py"}, cands...)
    }
    for _, c := range cands {
        if exe, err := exec.LookPath(c); err == nil {
            f.pyExe = exe
            return nil
        }
    }
    return errors.New("python is required for fastembed backend; please install Python 3.9+ and retry")
}

func (f *fastEmbedPy) ensureVenv() error {
    // Use existing venv if python inside exists
    if _, err := os.Stat(f.venvPython()); err == nil {
        return nil
    }
    if f.pyExe == "" {
        if err := f.ensurePython(); err != nil { return err }
    }
    if err := os.MkdirAll(f.workDir, 0o755); err != nil { return err }
    // Create venv
    cmd := exec.Command(f.pyExe, "-m", "venv", f.venvDir)
    var out bytes.Buffer
    cmd.Stdout = &out
    cmd.Stderr = &out
    if err := cmd.Run(); err != nil {
        return fmt.Errorf("creating venv failed: %v: %s", err, out.String())
    }
    return nil
}

func (f *fastEmbedPy) ensureFastEmbedInstalled() error {
    py := f.pythonExec()
    // Upgrade pip and install fastembed
    cmds := [][]string{{py, "-m", "pip", "install", "--upgrade", "pip"}, {py, "-m", "pip", "install", "--upgrade", "fastembed"}}
    for _, a := range cmds {
        cmd := exec.Command(a[0], a[1:]...)
        cmd.Env = f.envForVenv()
        cmd.Dir = f.workDir
        var out bytes.Buffer
        cmd.Stdout = &out
        cmd.Stderr = &out
        if err := cmd.Run(); err != nil {
            return fmt.Errorf("pip failed: %v: %s", err, out.String())
        }
    }
    return nil
}

func (f *fastEmbedPy) pythonExec() string {
    if p := f.venvPython(); p != "" {
        if _, err := os.Stat(p); err == nil { return p }
    }
    if f.pyExe != "" { return f.pyExe }
    // fallback to PATH
    if exe, err := exec.LookPath("python3"); err == nil { return exe }
    if exe, err := exec.LookPath("python"); err == nil { return exe }
    if runtime.GOOS == "windows" {
        if exe, err := exec.LookPath("py"); err == nil { return exe }
    }
    return ""
}

func (f *fastEmbedPy) venvPython() string {
    if runtime.GOOS == "windows" {
        return filepath.Join(f.venvDir, "Scripts", "python.exe")
    }
    return filepath.Join(f.venvDir, "bin", "python3")
}

func (f *fastEmbedPy) envForVenv() []string {
    env := os.Environ()
    // Prepend venv bin to PATH
    var pathKey string = "PATH"
    if runtime.GOOS == "windows" {
        pathKey = "Path"
    }
    pathVal := f.venvBin() + string(os.PathListSeparator) + os.Getenv(pathKey)
    // Some Windows setups need both PATH and Path
    return append(env,
        "PYTHONNOUSERSITE=1",
        pathKey+"="+pathVal,
    )
}

func (f *fastEmbedPy) venvBin() string {
    if runtime.GOOS == "windows" {
        return filepath.Join(f.venvDir, "Scripts")
    }
    return filepath.Join(f.venvDir, "bin")
}
