#include "exec/ngram_reader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "util/log.h"

namespace lgc::ngram {

namespace {

constexpr uint32_t kPage = 4096;

double now_us() {
    using namespace std::chrono;
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / 1000.0;
}

uint64_t mix(uint64_t r) {
    const uint64_t x = r * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
}

}  // namespace

void RowReader::RowCache::init(uint64_t rows, size_t row_bytes) {
    sets = rows / kWays;
    rb   = row_bytes;
    keys.assign(sets * kWays, kEmpty);
    data.assign(sets * kWays * rb, 0);
    next.assign(sets, 0);
}

const uint8_t* RowReader::RowCache::find(uint64_t row) const {
    if (sets == 0) return nullptr;
    const uint64_t s = mix(row) % sets;
    for (uint32_t w = 0; w < kWays; ++w)
        if (keys[s * kWays + w] == row) return &data[(s * kWays + w) * rb];
    return nullptr;
}

void RowReader::RowCache::insert(uint64_t row, const uint8_t* bytes) {
    if (sets == 0 || find(row) != nullptr) return;
    const uint64_t s = mix(row) % sets;
    const uint32_t w = next[s];
    next[s] = static_cast<uint8_t>((w + 1) % kWays);
    keys[s * kWays + w] = row;
    std::memcpy(&data[(s * kWays + w) * rb], bytes, rb);
}

RowReader::RowReader(const std::string& path, uint64_t table_offset, uint64_t n_rows, size_t row_bytes,
                     const ReaderOptions& opt)
    : table_offset_(table_offset), n_rows_(n_rows), row_bytes_(row_bytes), opt_(opt) {
    if (row_bytes_ == 0 || row_bytes_ > kPage)
        throw std::runtime_error(log::format("ngram reader: row size %zu is not 1..%u", row_bytes_, kPage));
    if (opt_.direct) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        direct_ = fd_ >= 0;
    }
    if (fd_ < 0) {
        // A filesystem that refuses O_DIRECT (tmpfs) still reads, through the
        // page cache: the rows are the same bytes.
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0)
            throw std::runtime_error(log::format("ngram reader: cannot open %s: %s", path.c_str(),
                                                 std::strerror(errno)));
        if (opt_.direct)
            log::warn("ngram", "%s refuses O_DIRECT; the n-gram rows are read through the page cache",
                      path.c_str());
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        throw std::runtime_error(log::format("ngram reader: cannot size %s", path.c_str()));
    }
    file_size_ = static_cast<uint64_t>(st.st_size);
    if (table_offset_ + n_rows_ * row_bytes_ > file_size_) {
        ::close(fd_);
        throw std::runtime_error(log::format("ngram reader: the table extends past the end of %s", path.c_str()));
    }
    cache_.init(opt_.cache_rows, row_bytes_);
    rng_ ^= static_cast<uint64_t>(now_us());
    const int n = std::clamp(opt_.io_threads, 1, 64);
    for (int i = 0; i < n; ++i) threads_.emplace_back([this] { worker(); });
    if (opt_.keepalive_ms > 0) keepalive_ = std::thread([this] { keepalive_loop(); });
}

RowReader::~RowReader() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        queue_.clear();
    }
    cv_work_.notify_all();
    cv_keep_.notify_all();
    for (auto& t : threads_) t.join();
    if (keepalive_.joinable()) keepalive_.join();
    const ReaderStats s = stats_;
    if (s.requests > 0) {
        log::info("ngram",
                  "reader: %llu rows in %llu forwards, %.1f %% from the row cache, %llu page reads "
                  "(%llu rows shared a page), collect waited %.3f ms a forward, %llu keep-alive reads, %s",
                  static_cast<unsigned long long>(s.requests), static_cast<unsigned long long>(s.tickets),
                  100.0 * static_cast<double>(s.cache_hits) / static_cast<double>(s.requests),
                  static_cast<unsigned long long>(s.reads), static_cast<unsigned long long>(s.dedup_rows),
                  s.tickets ? s.wait_us / 1000.0 / static_cast<double>(s.tickets) : 0.0,
                  static_cast<unsigned long long>(s.keepalive_reads), direct_ ? "O_DIRECT" : "buffered");
    }
    ::close(fd_);
}

uint64_t RowReader::issue(const std::vector<uint64_t>& rows, uint8_t* dst) {
    std::unique_lock<std::mutex> lk(m_);
    const uint64_t id = next_ticket_++;
    ++stats_.tickets;
    last_issue_us_ = now_us();
    std::unordered_map<uint64_t, size_t> by_page;  // aligned offset -> index in `jobs`
    std::vector<Job> jobs;
    for (size_t i = 0; i < rows.size(); ++i) {
        uint8_t* d = dst + i * row_bytes_;
        ++stats_.requests;
        if (const uint8_t* hit = cache_.find(rows[i])) {
            std::memcpy(d, hit, row_bytes_);
            ++stats_.cache_hits;
            continue;
        }
        const uint64_t at     = table_offset_ + rows[i] * row_bytes_;
        const uint64_t first  = at / kPage * kPage;
        const uint32_t length = static_cast<uint32_t>((at + row_bytes_ - 1) / kPage * kPage - first + kPage);
        const Use      use{static_cast<uint32_t>(at - first), d, rows[i]};
        auto f = by_page.find(first);
        if (f != by_page.end()) {
            Job& j   = jobs[f->second];
            j.length = std::max(j.length, length);
            j.uses.push_back(use);
            ++stats_.dedup_rows;
            continue;
        }
        by_page.emplace(first, jobs.size());
        Job j;
        j.offset = first;
        j.length = length;
        j.ticket = id;
        j.uses.push_back(use);
        jobs.push_back(std::move(j));
    }
    pending_[id] = static_cast<uint32_t>(jobs.size());
    for (auto& j : jobs) queue_.push_back(std::move(j));
    lk.unlock();
    if (!jobs.empty()) cv_work_.notify_all();
    return id;
}

void RowReader::collect(uint64_t ticket) {
    const double t0 = now_us();
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] {
        auto it = pending_.find(ticket);
        return it == pending_.end() || it->second == 0 || !error_.empty();
    });
    pending_.erase(ticket);
    stats_.wait_us += now_us() - t0;
    if (!error_.empty()) throw std::runtime_error("ngram reader: " + error_);
}

ReaderStats RowReader::stats() const {
    std::lock_guard<std::mutex> lk(m_);
    return stats_;
}

void RowReader::worker() {
    void* raw = nullptr;
    if (::posix_memalign(&raw, kPage, 2 * kPage) != 0) {
        std::lock_guard<std::mutex> lk(m_);
        error_ = "cannot allocate a read buffer";
        cv_done_.notify_all();
        return;
    }
    auto* buf = static_cast<uint8_t*>(raw);
    std::unique_lock<std::mutex> lk(m_);
    for (;;) {
        cv_work_.wait(lk, [&] { return stop_ || !queue_.empty(); });
        if (stop_) break;
        Job j = std::move(queue_.front());
        queue_.pop_front();
        last_read_us_ = now_us();
        lk.unlock();
        // The last page of the file may be short: a read is good when it
        // covers every row it carries.
        uint32_t got = 0;
        bool     ok  = true;
        while (got < j.length) {
            const ssize_t n = ::pread(fd_, buf + got, j.length - got, static_cast<off_t>(j.offset + got));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            got += static_cast<uint32_t>(n);
        }
        for (const Use& u : j.uses) {
            if (u.in_page + row_bytes_ > got) {
                ok = false;
                break;
            }
            std::memcpy(u.dst, buf + u.in_page, row_bytes_);
        }
        lk.lock();
        if (j.ticket == 0) {
            ++stats_.keepalive_reads;
            continue;
        }
        if (!ok) {
            error_ = log::format("a read of %u bytes at %llu returned %u", j.length,
                                 static_cast<unsigned long long>(j.offset), got);
        } else {
            ++stats_.reads;
            stats_.bytes += got;
            for (const Use& u : j.uses) cache_.insert(u.row, buf + u.in_page);
        }
        auto it = pending_.find(j.ticket);
        if (it != pending_.end() && it->second > 0) --it->second;
        cv_done_.notify_all();
    }
    lk.unlock();
    std::free(raw);
}

// Strata's keep-alive (PleReader::set_keepalive): some SSDs drop into a power
// state after ~250 ms without a command and stall the next reads 50-150 ms,
// which in decode happens after a few rounds served from the row cache. While
// rows were asked for within the window, a page of the table is read when no
// read went out for `keepalive_ms`.
void RowReader::keepalive_loop() {
    const auto period = std::chrono::microseconds(static_cast<int64_t>(opt_.keepalive_ms * 1000.0));
    std::unique_lock<std::mutex> lk(m_);
    while (!stop_) {
        cv_keep_.wait_for(lk, period, [&] { return stop_; });
        if (stop_) break;
        const double now = now_us();
        if (last_issue_us_ <= 0 || now - last_issue_us_ > opt_.keepalive_window_s * 1e6) continue;
        if (now - last_read_us_ < opt_.keepalive_ms * 1000.0) continue;
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 7;
        rng_ ^= rng_ << 17;
        const uint64_t first = table_offset_ / kPage, end = (table_offset_ + n_rows_ * row_bytes_) / kPage;
        Job j;
        j.offset = (first + (end > first ? rng_ % (end - first) : 0)) * kPage;
        j.length = kPage;
        j.ticket = 0;
        queue_.push_back(std::move(j));
        last_read_us_ = now;
        cv_work_.notify_one();
    }
}

}  // namespace lgc::ngram
