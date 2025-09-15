package server

import "context"

type TTSService interface {
    Synthesize(ctx context.Context, text, voice string) ([]byte, error)
}

