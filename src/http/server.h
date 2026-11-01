#pragma once

#include <memory>

#include "api/handlers.h"
#include "config.h"

namespace lgc {

// ops-2 item 6: in an ARCINT_TLS build with --tls-cert/--tls-key set, load the
// certificate and key once at boot (the same OpenSSL calls httplib's SSLServer
// makes, no listener) and return false + `error` when they do not load. Called
// before the backend so a bad TLS material fails without a model load. A build
// without ARCINT_TLS, or with no TLS flags, returns true.
bool tls_material_loads(const Config& cfg, std::string& error);

// Thin wrapper over cpp-httplib. Pimpl'd so the 700 KB vendored header is
// compiled exactly once.
class HttpServer {
public:
    HttpServer(const Config& cfg, api::Context& ctx);
    ~HttpServer();

    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Blocks until stop() is called. False if the socket could not be bound.
    bool listen();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lgc
