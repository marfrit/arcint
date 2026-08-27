#pragma once

#include <memory>

#include "api/handlers.h"
#include "config.h"

namespace lgc {

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
