#include "core/turnstile.h"

#include <chrono>

namespace lgc {

Turnstile::Turn& Turnstile::Turn::operator=(Turn&& other) noexcept {
    if (this != &other) {
        if (gate_ != nullptr) gate_->release();
        gate_        = other.gate_;
        waited_      = other.waited_;
        other.gate_  = nullptr;
        other.waited_ = 0.0;
    }
    return *this;
}

Turnstile::Turn::~Turn() {
    if (gate_ != nullptr) gate_->release();
}

Turnstile::Turn Turnstile::take() {
    const auto               t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t           mine = next_ticket_++;
    // The uncontended path never touches the clock twice and never sleeps.
    if (mine == now_serving_) return Turn(this, 0.0);
    cv_.wait(lock, [&] { return now_serving_ == mine; });
    return Turn(this, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

void Turnstile::release() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ++now_serving_;
    }
    cv_.notify_all();
}

uint64_t Turnstile::served() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return now_serving_;
}

uint64_t Turnstile::issued() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return next_ticket_;
}

}  // namespace lgc
