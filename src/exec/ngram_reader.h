#pragma once

// The n-gram table's per-forward rows read the way Strata reads them
// (`code`: ~/src/Strata-ref/include/strata/ngram/ple_reader.hpp,
// src/ngram/ple_reader.cpp, src/platform/direct_file.cpp): unbuffered
// (O_DIRECT) page reads on a pool of I/O threads -- the thread count is the
// queue depth, so a token's 16 rows on 16 pages are read at once instead of
// one `pread` after another -- with the rows of one request that share a page
// read once, a bounded row cache of rows this process has already fetched,
// and an SSD keep-alive. The split API is Strata's:
//
//     t = reader.issue(rows, dst);   // as soon as the forward's rows are known
//     ...                            // the rest of the forward's setup
//     reader.collect(t);             // before the forward reads them
//
// The bytes are the file's, so the staged rows are `pread_staging_rows`'s,
// byte for byte (`tests/test_ngram_staging.cpp`).
//
// Unbuffered reads keep the table's random 4 KiB pages out of the page cache,
// which on the served Flash-Next host holds the CPU tier's experts (Strata
// notes read-ahead on this pattern "is pure waste and would evict useful
// pages", include/strata/kernels/ngram.hpp:201-204).

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lgc::ngram {

struct ReaderOptions {
    int      io_threads         = 16;          // Strata: STRATA_IO_THREADS, default 16
    uint64_t cache_rows         = 1ull << 20;  // Strata: PleIoOptions::cache_rows (~95 MB of 90 B rows)
    double   keepalive_ms       = 100.0;       // Strata: STRATA_SSD_KEEPALIVE, default 100
    double   keepalive_window_s = 60.0;        // Strata: STRATA_SSD_KEEPALIVE_WINDOW, default 60
    bool     direct             = true;        // O_DIRECT; falls back to buffered where refused
};

struct ReaderStats {
    uint64_t requests        = 0;  // rows asked for
    uint64_t cache_hits      = 0;  // rows served from the row cache
    uint64_t dedup_rows      = 0;  // rows that shared a page already being read in the same request
    uint64_t reads           = 0;  // page reads issued (rows)
    uint64_t bytes           = 0;  // bytes read for rows
    uint64_t tickets         = 0;
    double   wait_us         = 0;  // time `collect` spent blocked
    uint64_t keepalive_reads = 0;
};

class RowReader {
public:
    // `table_offset`: the file offset of row 0; `n_rows` rows of `row_bytes`
    // (at most one page). Throws when the file cannot be opened.
    RowReader(const std::string& path, uint64_t table_offset, uint64_t n_rows, size_t row_bytes,
              const ReaderOptions& opt = {});
    ~RowReader();
    RowReader(const RowReader&) = delete;
    RowReader& operator=(const RowReader&) = delete;

    // Starts fetching `rows`; row i's bytes land at `dst + i * row_bytes`.
    // `dst` must stay valid until `collect` returns. Rows must be in range
    // (the caller's plan refuses others by name).
    uint64_t issue(const std::vector<uint64_t>& rows, uint8_t* dst);
    // Blocks until every row of the ticket is in place; throws on an I/O error.
    void collect(uint64_t ticket);

    ReaderStats stats() const;
    bool        direct() const { return direct_; }

private:
    struct Use {
        uint32_t in_page;  // byte offset of the row inside the read
        uint8_t* dst;
        uint64_t row;
    };
    struct Job {
        uint64_t         offset = 0;  // aligned file offset
        uint32_t         length = 0;  // one page, or two for a row across a page boundary
        uint64_t         ticket = 0;  // 0: a keep-alive read
        std::vector<Use> uses;
    };
    // Strata's RowCache: 8 ways per set, round-robin replacement in a set.
    struct RowCache {
        static constexpr uint32_t kWays  = 8;
        static constexpr uint64_t kEmpty = ~0ull;
        uint64_t              sets = 0;
        size_t                rb   = 0;
        std::vector<uint64_t> keys;
        std::vector<uint8_t>  data;
        std::vector<uint8_t>  next;
        void           init(uint64_t rows, size_t row_bytes);
        const uint8_t* find(uint64_t row) const;
        void           insert(uint64_t row, const uint8_t* bytes);
    };

    void worker();
    void keepalive_loop();

    int                       fd_ = -1;
    bool                      direct_ = false;
    uint64_t                  table_offset_;
    uint64_t                  n_rows_;
    size_t                    row_bytes_;
    uint64_t                  file_size_ = 0;
    ReaderOptions             opt_;

    mutable std::mutex        m_;
    std::condition_variable   cv_work_, cv_done_, cv_keep_;
    std::deque<Job>           queue_;
    std::unordered_map<uint64_t, uint32_t> pending_;  // ticket -> page reads not yet done
    uint64_t                  next_ticket_ = 1;
    RowCache                  cache_;
    ReaderStats               stats_;
    std::string               error_;
    bool                      stop_ = false;
    double                    last_issue_us_ = 0, last_read_us_ = 0;
    uint64_t                  rng_ = 0x9E3779B97F4A7C15ull;
    std::vector<std::thread>  threads_;
    std::thread               keepalive_;
};

}  // namespace lgc::ngram
