package embeddings

import (
    "context"
    "hash/fnv"
    "math"
    "fmt"
    "regexp"
    "strings"
)

// Service is an interface to produce embeddings.
type Service interface {
    Embed(ctx context.Context, inputs []string) ([][]float32, string, error)
}

type Config struct {
    ModelName string
    ModelDir  string
    Backend   string // fastembed|hash
    WorkDir   string // for venv/cache
}

// Default lightweight, deterministic embedding (portable, no deps).
// Provides a drop-in local embedding useful for dev and fallback.
type hashEmbedder struct {
    modelName string
    dim       int
}

func New(cfg Config) Service {
    if cfg.ModelName == "" { cfg.ModelName = "hash-embeddings-384" }
    return &hashEmbedder{modelName: cfg.ModelName, dim: 384}
}

// NewWithBackend returns the requested backend when available, otherwise an error.
func NewWithBackend(cfg Config) (Service, error) {
    b := strings.ToLower(cfg.Backend)
    switch b {
    case "", "hash":
        return New(cfg), nil
    case "fastembed":
        return newFastEmbedPy(cfg)
    default:
        return nil, fmt.Errorf("unknown embedding backend: %s", cfg.Backend)
    }
}

func (h *hashEmbedder) Embed(_ context.Context, inputs []string) ([][]float32, string, error) {
    out := make([][]float32, len(inputs))
    for i, s := range inputs {
        out[i] = h.embedOne(s)
    }
    return out, h.modelName, nil
}

var wordRE = regexp.MustCompile(`\pL+|\pN+`)

func (h *hashEmbedder) embedOne(text string) []float32 {
    vec := make([]float32, h.dim)
    tokens := wordRE.FindAllString(strings.ToLower(text), -1)
    if len(tokens) == 0 { return vec }
    for _, tok := range tokens {
        idx := h.hash(tok) % uint32(h.dim)
        // signed bucket hashing for some balance
        sign := float32(1)
        if (h.hash(tok+"_alt") & 1) == 1 { sign = -1 }
        vec[idx] += sign
    }
    // l2 normalize
    var norm float64
    for _, v := range vec { norm += float64(v*v) }
    norm = math.Sqrt(norm)
    if norm > 0 {
        inv := float32(1.0/ norm)
        for i := range vec { vec[i] *= inv }
    }
    return vec
}

func (h *hashEmbedder) hash(s string) uint32 {
    hsh := fnv.New32a()
    _, _ = hsh.Write([]byte(s))
    return hsh.Sum32()
}
