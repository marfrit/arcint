#include "api/responses_state.h"

namespace lgc::api {

ResponseStore::ResponseStore(Limits limits) : limits_(limits) {}
void ResponseStore::put(std::shared_ptr<StoredResponse>) {}
std::shared_ptr<StoredResponse> ResponseStore::get(const std::string&, int) { return nullptr; }
void ResponseStore::erase(const std::string&) {}
void ResponseStore::sweep(int64_t) {}
void ResponseStore::evict_lane_locked(int, int64_t) {}
int ResponseStore::entries() const { return 0; }
uint64_t ResponseStore::bytes() const { return 0; }
ResponseStore::Limits ResponseStore::limits() const { return limits_; }

std::string sha256_hex(std::string_view) { return std::string(64, '0'); }

void remember(const Context&, const PreparedResponse&, const nlohmann::json&, int) {}
void evict_continuation(const Context&, const std::string&) {}

}  // namespace lgc::api
