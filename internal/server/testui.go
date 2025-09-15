package server

import (
    "embed"
    "io/fs"
    "log"
    "net/http"
)

//go:embed web/test/*
var testFS embed.FS

// RegisterTestUI mounts the simple test frontend at /test/.
func RegisterTestUI(mux *http.ServeMux) {
    sub, err := fs.Sub(testFS, "web/test")
    if err != nil {
        log.Printf("test UI fs error: %v", err)
        sub = testFS
    }
    mux.Handle("/test/", http.StripPrefix("/test/", http.FileServer(http.FS(sub))))
}
