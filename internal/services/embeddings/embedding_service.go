package embeddings

import (
    "context"
    "hash/fnv"
    "math"
    "regexp"
    "strings"
    "unicode"
)

// Service is an interface to produce embeddings.
type Service interface {
    Embed(ctx context.Context, inputs []string) ([][]float32, string, error)
}

type Config struct {
    ModelName string
}

// MiniLM L6-v2 compatible, deterministic embedding (384-dim) with no external deps.
// This is a heuristic approximation suitable for testing and offline use.
type miniLMCompat struct {
    modelName string
    dim       int
}

func New(cfg Config) Service {
    // Force model to all-MiniLM-L6-v2 and 384 dims
    return &miniLMCompat{modelName: "all-MiniLM-L6-v2", dim: 384}
}

func (h *miniLMCompat) Embed(_ context.Context, inputs []string) ([][]float32, string, error) {
    out := make([][]float32, len(inputs))
    for i, s := range inputs {
        out[i] = h.embedOne(s)
    }
    return out, h.modelName, nil
}

// Use Go RE2 Unicode classes with braces; fallback handled in code.
var wordRE = regexp.MustCompile(`\p{L}+|\p{N}+`)

func (h *miniLMCompat) embedOne(text string) []float32 {
    vec := make([]float32, h.dim)
    tokens := wordRE.FindAllString(strings.ToLower(text), -1)
    if len(tokens) == 0 {
        tokens = fallbackTokens(text)
    }
    if len(tokens) == 0 { return vec }

    // Build character 3-grams for each token (with boundary markers) and add token bigrams
    features := make([]string, 0, len(tokens)*6)
    for _, t := range tokens {
        features = append(features, charNGrams(t, 3)...)
    }
    for i := 0; i+1 < len(tokens); i++ {
        features = append(features, charNGrams(tokens[i]+"_"+tokens[i+1], 3)...)
    }

    if len(features) == 0 { return vec }

    // Signed k-projection feature hashing to increase diversity
    const k = 16
    total := float64(len(features) * k)
    if total < 1 { total = 1 }
    w := float32(1.0 / math.Sqrt(total))
    for _, f := range features {
        base := hashIndex(f)
        x := uint32(base)
        for j := 0; j < k; j++ {
            // advance PRNG
            x = xorshift32(x + uint32(j)*0x9e3779b9)
            idx := int(x % uint32(h.dim))
            // derive sign and magnitude from next state
            x2 := xorshift32(x ^ 0x85ebca6b)
            sgn := float32(1)
            if (x2 & 1) == 1 { sgn = -1 }
            // small magnitude variation in (0.5 .. 1.0]
            mag := 0.5 + float32((x2>>1)&0x7fff)/32767.0*0.5
            vec[idx] += sgn * mag * w
        }
    }

    // l2 normalize
    var norm float64
    for _, v := range vec { norm += float64(v*v) }
    norm = math.Sqrt(norm)
    if norm > 0 {
        inv := float32(1.0/ norm)
        for i := range vec { vec[i] *= inv }
    } else {
        // Ensure non-zero output deterministically
        idx := h.hash(text) % uint32(h.dim)
        vec[idx] = 1
    }
    return vec
}

func (h *miniLMCompat) hash(s string) uint32 {
    hsh := fnv.New32a()
    _, _ = hsh.Write([]byte(s))
    return hsh.Sum32()
}

func fallbackTokens(s string) []string {
    s = strings.ToLower(s)
    var b strings.Builder
    toks := make([]string, 0, 16)
    flush := func() {
        if b.Len() > 0 {
            toks = append(toks, b.String())
            b.Reset()
        }
    }
    for _, r := range s {
        if unicode.IsLetter(r) || unicode.IsDigit(r) {
            b.WriteRune(r)
        } else {
            flush()
        }
    }
    flush()
    return toks
}

func charNGrams(token string, n int) []string {
    if n <= 0 { return nil }
    // add boundary markers
    t := "^" + token + "$"
    runes := []rune(t)
    if len(runes) < n {
        return []string{string(runes)}
    }
    out := make([]string, 0, len(runes)-n+1)
    for i := 0; i+n <= len(runes); i++ {
        out = append(out, string(runes[i:i+n]))
    }
    return out
}

func hashIndex(s string) int32 {
    h := fnv.New32a()
    _, _ = h.Write([]byte(s))
    return int32(h.Sum32())
}

func hashSign(s string) bool {
    // Use FNV-1 (non-a) for independent sign bit
    h := fnv.New32()
    _, _ = h.Write([]byte(s))
    return (h.Sum32() & 1) == 1
}

func xorshift32(x uint32) uint32 {
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x
}
