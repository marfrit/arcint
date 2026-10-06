#pragma once

// nvme-direct-expert-tier (0.5.3 LISBON), design note D2/D3 -- the load-time
// pinned fill's schedule, kept pure and device-free so it builds and is tested
// on any host. `docs/design-nvme-direct-expert-tier.md` §3 fixes the shape:
//
//   * membership is the static partition's pinned set, computed ONCE from
//     configuration at bind() and never re-derived from live traffic
//     (patch 0018; patch 0046). This header does not compute membership --
//     the caller hands it the batches -- so it cannot grow a traffic input.
//   * one batch per layer (ratio 86 -> 71 requests, ratio 75 -> 128; both
//     under AW_BATCH_MAX = 256).
//   * FOUR batches in flight over AW_IOC_SUBMIT_BATCH / AW_IOC_BATCH_WAIT,
//     collecting the oldest (the note's measured shape: four batches overlap
//     and all collect).
//   * on collect, every landed expert is marked filled -- the caller's
//     callback is the cache's set_filled(slot), so the existing first-use
//     fill branch never fires on the decode path.
//   * NO fetch on the decode path. The transport interface deliberately has
//     no synchronous read primitive; `require_no_sync_read` is the guard that
//     refuses a would-be synchronous AW_IOC_READ_BLOCKS once serving has
//     begun (the campaign's Out).
//
// D3, the load barrier: a pinned expert whose fetch has not landed is a LOAD
// FAILURE -- retried at the barrier, then refused. It is never silently
// demoted to the host tier: host and device are different arithmetic, so a
// demotion would make this boot's residency differ from a clean boot and break
// the byte-identity the gate measures across two cold boots. The scheduler has
// exactly one retry per batch and only ever reports success for a fully
// collected batch; a shortfall is a refusal, never a partial fill.
//
// The transport is injected. The production implementation, WHEN SUPPLIED,
// will drive arcwell's batch ioctls (`~/src/arcwell`, `stub/include/aw_uapi.h`);
// it is not part of this header, so the schedule is testable with a fake and
// does not link against the module. No production transport is wired yet: the
// static partition's pinned slots are host-mapped and arcwell needs a dma-buf
// from an xe VRAM BO, so the destination is OWED (campaign
// nvme-direct-expert-tier, D2/D3). `Transport::begin` is the whole-configuration
// setup: a failure there (MAP_BUFFER refuses, out_flags lacks
// AW_MAP_F_REQUIRE_P2P, the BO cannot be registered, the batch ioctl is
// unavailable) is a load failure for the configuration -- there is no
// host-bounce fallback path.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lgc::nvme_fill {

// One pinned expert to fetch. `lba`/`bytes` are the file's absolute NVMe block
// range, translated once at open (FIEMAP + partition start); `slot` is the
// fixed device slot the static partition assigned at bind(); `layer` and
// `expert` identify the member for accounting and refusal messages.
struct FillRequest {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint32_t slot = 0;
    uint64_t lba = 0;
    uint64_t bytes = 0;
};

// One batch = one layer's pinned set, submitted as a single
// AW_IOC_SUBMIT_BATCH. The note's ratio-86 set is 71 requests; AW_BATCH_MAX is
// 256, so one batch per layer always fits.
struct FillBatch {
    uint32_t layer = 0;
    std::vector<FillRequest> requests;
};

// Whether the fill is still at load (the only phase a fetch may run in),
// serving (the decode path, where it must not), or terminally refused (a load
// failure; nothing else may run).
enum class Phase { kCreated, kLoading, kServing, kRefused };

struct CollectOutcome {
    bool        ok = false;
    std::size_t completed = 0;
    std::size_t requested = 0;
    std::string error;
};

// The arcwell transport the scheduler drives. It exposes ONLY the batch
// surface (submit + collect) plus setup/teardown: there is no synchronous
// read, so a losing configuration cannot be reached through this interface.
// A single collector per batch id is the module's own contract (a second
// concurrent wait returns -EBUSY); this scheduler collects each id once.
class Transport {
public:
    virtual ~Transport() = default;

    // Whole-configuration setup. Returns false (with `reason`) when arcwell
    // cannot be set up at all -- a load failure for the configuration.
    virtual bool begin(std::string* reason) = 0;

    // Queue one batch. Returns false only for a submission-time failure;
    // transfer errors surface at collect.
    virtual bool submit(const FillBatch& batch, std::uint64_t* batch_id,
                        std::string* reason) = 0;

    // Collect a batch, blocking until it is done. `outcome.completed` is the
    // number of requests that landed; `outcome.requested` the number
    // submitted. A shortfall is reported, not hidden.
    virtual bool collect(std::uint64_t batch_id, CollectOutcome* outcome,
                         std::string* reason) = 0;

    virtual void end() {}
};

// The load-time schedule. Constructed at bind() with the transport and the
// note's depth (4); fed one batch per layer; run once at the load barrier.
class Scheduler {
public:
    using FilledCallback = std::function<void(const FillRequest&)>;

    explicit Scheduler(Transport* transport, std::size_t depth = 4)
        : transport_(transport), depth_(depth == 0 ? 1 : depth) {}

    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    Phase phase() const { return phase_; }

    // Set the transport up. A false return is a load failure; nothing else in
    // the schedule may run.
    bool begin(std::string* reason) {
        if (phase_ != Phase::kCreated) {
            if (reason) *reason = "pinned fill begin() called twice";
            return false;
        }
        if (transport_ == nullptr) {
            if (reason) *reason = "pinned fill has no transport";
            return false;
        }
        if (!transport_->begin(reason)) return false;
        phase_ = Phase::kLoading;
        return true;
    }

    // One batch per layer. May only be called before the barrier.
    bool enqueue(FillBatch batch, std::string* reason) {
        if (phase_ != Phase::kLoading) {
            if (reason) *reason = "pinned fill enqueue() outside the load phase";
            return false;
        }
        pending_.push_back(std::move(batch));
        return true;
    }

    // The load barrier. Drives the batches four at a time, collects the oldest,
    // marks every landed expert filled via `on_filled`, retries a short batch
    // once, and REFUSES -- with `reason` -- if any pinned expert is still not
    // landed. A refusal is TERMINAL (phase kRefused): a second call refuses too,
    // so a partially filled configuration can never resume into serving. On
    // success the phase becomes kServing. No fetch runs after this.
    bool barrier(const FilledCallback& on_filled, std::string* reason) {
        if (phase_ == Phase::kCreated) {
            if (reason) *reason = "pinned fill barrier() before begin()";
            return false;
        }
        if (phase_ == Phase::kRefused) {
            if (reason) *reason = "pinned fill already refused the load";
            return false;
        }
        if (phase_ == Phase::kServing) {
            if (reason) *reason = "pinned fill barrier() already run";
            return false;
        }

        std::map<std::uint64_t, FillBatch> inflight;
        std::deque<std::uint64_t>         order;
        std::map<std::uint32_t, bool>     retried;  // one retry per layer, see below

        auto submit_one = [&](const FillBatch& batch) -> bool {
            std::uint64_t id = 0;
            if (!transport_->submit(batch, &id, reason)) return false;
            inflight.emplace(id, batch);
            order.push_back(id);
            return true;
        };

        while (!pending_.empty() || !inflight.empty()) {
            // Fill the in-flight window to `depth_` before collecting, so the
            // NVMe queue depth is actually used (the note's four-overlap
            // shape).
            while (inflight.size() < depth_ && !pending_.empty()) {
                FillBatch next = std::move(pending_.front());
                pending_.pop_front();
                if (!submit_one(next)) return false;
            }
            if (inflight.empty()) {
                if (reason) *reason = "pinned fill stalled with no batch in flight";
                return false;
            }

            const std::uint64_t oldest = order.front();
            order.pop_front();
            FillBatch batch = std::move(inflight.at(oldest));
            inflight.erase(oldest);

            CollectOutcome outcome;
            if (!transport_->collect(oldest, &outcome, reason)) return false;

            if (outcome.ok && outcome.completed == batch.requests.size()) {
                for (const auto& request : batch.requests) on_filled(request);
                continue;
            }

            // A short batch: retry it once, in full. Partial completion does
            // not name which requests landed, so re-issuing the whole batch is
            // the only way to keep "filled" honest -- and re-fetching an
            // already-landed expert is idempotent. One retry per layer; a
            // second shortfall is a refusal.
            if (retried.find(batch.layer) == retried.end()) {
                retried.emplace(batch.layer, true);
                if (!submit_one(batch)) return false;
                continue;
            }

            if (reason) {
                *reason = "load failed: pinned expert set for layer " +
                          std::to_string(batch.layer) + " did not land after a retry (" +
                          std::to_string(outcome.completed) + " of " +
                          std::to_string(batch.requests.size()) +
                          "); refusing the load rather than demoting to the host tier";
            }
            phase_ = Phase::kRefused;
            return false;
        }

        phase_ = Phase::kServing;
        return true;
    }

    // The losing configuration guard. Any code path that would issue a
    // synchronous AW_IOC_READ_BLOCKS must come through here; once the load
    // barrier has passed, the decode path is serving and the read is refused.
    void require_no_sync_read(const char* what) const {
        if (phase_ == Phase::kServing) {
            throw std::logic_error(
                std::string("synchronous arcwell read forbidden on the decode path: ") +
                (what != nullptr ? what : "AW_IOC_READ_BLOCKS"));
        }
    }

private:
    Transport*                 transport_ = nullptr;
    std::size_t                depth_     = 4;
    Phase                      phase_     = Phase::kCreated;
    std::deque<FillBatch>      pending_;
};

}  // namespace lgc::nvme_fill
