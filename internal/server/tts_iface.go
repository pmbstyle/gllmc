package server

import (
    "context"
    "net/http"
)

type TTSService interface {
    Synthesize(ctx context.Context, text, voice string) ([]byte, error)
}

type LLMService interface {
    ProxyChatCompletions(w http.ResponseWriter, r *http.Request)
    ProxyCompletions(w http.ResponseWriter, r *http.Request)
}
