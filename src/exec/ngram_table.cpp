#include "exec/ngram_table.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <utility>

#include "core/ngram_header.h"
#include "exec/ngram_gather.h"
#include "util/log.h"

namespace lgc::ngram {

NGramLookup::~NGramLookup() {
    if (map_ != nullptr) {
        ::munmap(map_, map_size_);
        map_ = nullptr;
    }
}

NGramLookup::NGramLookup(NGramLookup&& o) noexcept { *this = std::move(o); }

NGramLookup& NGramLookup::operator=(NGramLookup&& o) noexcept {
    if (this != &o) {
        if (map_ != nullptr) ::munmap(map_, map_size_);
        map_              = o.map_;
        map_size_         = o.map_size_;
        owned_            = std::move(o.owned_);
        payload_          = o.payload_;
        ggml_type_        = o.ggml_type_;
        n_cols_           = o.n_cols_;
        n_rows_           = o.n_rows_;
        row_stride_bytes_ = o.row_stride_bytes_;
        layers_           = std::move(o.layers_);
        o.map_      = nullptr;
        o.map_size_ = 0;
        o.payload_  = nullptr;
    }
    return *this;
}

void NGramLookup::finish_construction() {
    if (n_cols_ % kBlockElements != 0)
        throw std::runtime_error(log::format(
            "ngram lookup: n_cols %zu is not a multiple of the %u-element block",
            n_cols_, kBlockElements));
    const std::size_t bpb = bytes_per_block(static_cast<uint32_t>(ggml_type_));
    if (bpb == 0)
        throw std::runtime_error(log::format(
            "ngram lookup: unknown ggml_type %d", ggml_type_));
    row_stride_bytes_ = bpb * (n_cols_ / kBlockElements);
    if (layers_.empty())
        throw std::runtime_error("ngram lookup: no PLE layer hash constants");
    for (const auto& p : layers_) p.validate();
}

NGramLookup NGramLookup::adopt_owned(std::vector<uint8_t> payload, int32_t ggml_type,
                                     size_t n_cols, uint32_t n_rows,
                                     std::vector<HashParams> layers) {
    NGramLookup t;
    t.owned_     = std::move(payload);
    t.payload_   = t.owned_.data();
    t.ggml_type_ = ggml_type;
    t.n_cols_    = n_cols;
    t.n_rows_    = n_rows;
    t.layers_    = std::move(layers);
    t.finish_construction();
    if (t.owned_.size() < static_cast<size_t>(n_rows) * t.row_stride_bytes_)
        throw std::runtime_error(log::format(
            "ngram lookup: owned payload %zu bytes is short of %zu rows x %zu bytes/row",
            t.owned_.size(), static_cast<size_t>(n_rows), t.row_stride_bytes_));
    return t;
}

NGramLookup NGramLookup::mmap_table(const std::string& path, std::vector<HashParams> layers) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error(log::format("ngram lookup: cannot open '%s'", path.c_str()));
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        throw std::runtime_error(log::format("ngram lookup: '%s' is not a regular file", path.c_str()));
    }
    const size_t size = static_cast<size_t>(st.st_size);
    if (size < kHeaderBytes) {
        ::close(fd);
        throw std::runtime_error(log::format(
            "ngram lookup: '%s' is shorter than the ARCINGRM header", path.c_str()));
    }
    void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map != MAP_FAILED) {
        // Not populated or pinned: rows are paged in on demand. The gather
        // touches only the rows a forward looks up, so hint random access
        // rather than reading the whole table ahead.
        ::madvise(map, size, MADV_RANDOM);
    }
    ::close(fd);
    if (map == MAP_FAILED)
        throw std::runtime_error(log::format("ngram lookup: mmap failed for '%s'", path.c_str()));

    // `t` owns the mapping from here: any throw below unwinds through its
    // destructor, which munmaps.
    NGramLookup t;
    t.map_      = map;
    t.map_size_ = size;
    const auto* base = static_cast<const uint8_t*>(map);
    Header      header;
    if (auto err = parse_header(base, kHeaderBytes, header); !err.empty())
        throw std::runtime_error(log::format("ngram lookup: '%s': %s", path.c_str(), err.c_str()));
    t.payload_   = base + kHeaderBytes;
    t.ggml_type_ = static_cast<int32_t>(header.ggml_type);
    t.n_cols_    = header.n_cols;
    t.n_rows_    = header.n_rows;
    t.layers_    = std::move(layers);
    t.finish_construction();
    const uint64_t need =
        kHeaderBytes + static_cast<uint64_t>(header.n_rows) * t.row_stride_bytes_;
    if (size < need)
        throw std::runtime_error(log::format(
            "ngram lookup: '%s' is %zu bytes, header implies %llu", path.c_str(), size,
            static_cast<unsigned long long>(need)));
    return t;
}

std::vector<float> NGramLookup::lookup(size_t layer_ordinal, const std::vector<int64_t>& context,
                                       const std::vector<int64_t>& tokens) const {
    if (payload_ == nullptr)
        throw std::runtime_error("ngram lookup: table not constructed");
    const HashParams& p    = layers_.at(layer_ordinal);
    const auto        ids  = row_ids(p, context, tokens);  // [T * num_heads] int64
    std::vector<uint32_t> indices(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= n_rows_)
            throw std::runtime_error(log::format(
                "ngram lookup: row id %lld is outside the %u-row table (PLE layer %zu); "
                "the table is too small for the hashed row space",
                static_cast<long long>(ids[i]), n_rows_, layer_ordinal));
        indices[i] = static_cast<uint32_t>(ids[i]);
    }
    std::vector<float> out(ids.size() * n_cols_);
    gather_dequant(ggml_type_, payload_, row_stride_bytes_, n_cols_, indices, out.data());
    return out;
}

std::optional<NGramLookup> load_ngram_lookup(const Artifact& artifact, const std::string& path,
                                             uint64_t host_ram_bytes, uint64_t other_resident_bytes,
                                             uint64_t margin_bytes, std::string& err,
                                             uint64_t& out_payload_bytes) {
    err.clear();
    out_payload_bytes = 0;
    if (path.empty()) return std::nullopt;  // flag not set: cold path

    // Link 2 admission: header, type, row width, hashed-row-space lower bound,
    // on-disk size, host-RAM fit. Returns the payload bytes on success.
    std::string admit_err =
        admit_ngram_table_from_disk(artifact, path, host_ram_bytes, other_resident_bytes,
                                    margin_bytes, out_payload_bytes);
    if (!admit_err.empty()) {
        err = std::move(admit_err);
        out_payload_bytes = 0;
        return std::nullopt;
    }

    try {
        const auto& nc = artifact.ngram_config;
        // Admission guarantees ple_layer_ids is non-empty (a declared table with
        // an empty ple_layer_ids is refused there).
        const int num_ple = static_cast<int>(nc.ple_layer_ids.size());
        std::vector<HashParams> layers;
        layers.reserve(static_cast<size_t>(num_ple));
        for (int k = 0; k < num_ple; ++k) {
            HashParams hp = derive_hash_constants(nc.vocab_size, nc.ngram_size, nc.heads_per_ngram,
                                                  nc.ngram_vocab_size_base, /*ple_layer_index=*/k);
            hp.eos_token_id = nc.ngram_boundary_token_id;
            hp.validate();
            layers.push_back(std::move(hp));
        }
        return NGramLookup::mmap_table(path, std::move(layers));
    } catch (const std::exception& e) {
        err = e.what();
        out_payload_bytes = 0;
        return std::nullopt;
    }
}

}  // namespace lgc::ngram
