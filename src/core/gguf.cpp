#include "core/gguf.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util/log.h"

// GGUF v3 layout (llama.cpp's gguf.cpp / ggml's own writer, and gguf-py's
// GGUFWriter/GGUFReader, which this was checked against byte-for-byte):
//
//   magic      "GGUF" (4 bytes, not a u32 -- read as bytes so endianness of
//              the reading host never enters into it)
//   version    u32, 3 for the files this reader accepts
//   n_tensors  u64
//   n_kv       u64
//   <n_kv key-value pairs>
//   <n_tensors tensor infos>
//   <padding to `general.alignment`, default 32>
//   <tensor data, each tensor's start padded to alignment>
//
// A key-value pair is a length-prefixed string key, a u32 type tag, and a
// value: fixed-width for the scalar types, length-prefixed bytes for
// STRING, or (u32 elem_type, u64 count, count values) for ARRAY. A tensor
// info is a length-prefixed name, u32 n_dims, n_dims x u64 dims (dims[0]
// fastest-varying), u32 ggml type, u64 offset (relative to the data
// section start).
namespace lgc::gguf {

namespace {

// Element counts/bytes for the types this reader can size. Absent entries
// (Q8_1, IQ1_S, IQ1_M, MXFP4 among GgmlType's named ids, and anything not
// named at all) are refused at open by numeric id -- the design note's
// "types not in this table are unknown."
struct TypeEntry {
    int32_t     type;
    size_t      block_size;
    size_t      type_size;
    const char* name;
};

constexpr TypeEntry kTypes[] = {
    {0, 1, 4, "F32"},        {1, 1, 2, "F16"},         {2, 32, 18, "Q4_0"},
    {3, 32, 20, "Q4_1"},     {6, 32, 22, "Q5_0"},      {7, 32, 24, "Q5_1"},
    {8, 32, 34, "Q8_0"},     {10, 256, 84, "Q2_K"},    {11, 256, 110, "Q3_K"},
    {12, 256, 144, "Q4_K"},  {13, 256, 176, "Q5_K"},   {14, 256, 210, "Q6_K"},
    {15, 256, 292, "Q8_K"},  {16, 256, 66, "IQ2_XXS"}, {17, 256, 74, "IQ2_XS"},
    {18, 256, 98, "IQ3_XXS"},{20, 32, 18, "IQ4_NL"},   {21, 256, 110, "IQ3_S"},
    {22, 256, 82, "IQ2_S"},  {23, 256, 136, "IQ4_XS"}, {24, 1, 1, "I8"},
    {25, 1, 2, "I16"},       {26, 1, 4, "I32"},        {27, 1, 8, "I64"},
    {28, 1, 8, "F64"},       {30, 1, 2, "BF16"},
};

const TypeEntry* find_type(int32_t type) {
    for (const TypeEntry& e : kTypes) {
        if (e.type == type) return &e;
    }
    return nullptr;
}

// A cursor over the mapped bytes. Every read checks bounds first; a short
// file surfaces as "gguf: truncated" rather than a read past the mapping.
class Cursor {
public:
    Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    void need(uint64_t n) const {
        if (n > size_ - pos_) {
            throw std::runtime_error(log::format(
                "gguf: truncated file (need %llu more bytes at offset %zu, have %zu)",
                static_cast<unsigned long long>(n), pos_, size_ - pos_));
        }
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T v;
        std::memcpy(&v, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        const uint64_t len = read<uint64_t>();
        need(len);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += len;
        return s;
    }

    size_t pos() const { return pos_; }
    void   skip(uint64_t n) { need(n); pos_ += n; }

private:
    const uint8_t* data_;
    size_t         size_;
    size_t         pos_ = 0;
};

// Reads one value of `vtype` (not an array -- read_array handles that
// separately since arrays nest one level of element type). Throws if
// vtype is ARRAY (nested arrays: "not needed", design note stage 0).
void read_scalar_into(Cursor& c, ValueType vtype, MetaValue& out) {
    switch (vtype) {
        case ValueType::UInt8:  out.ints.push_back(c.read<uint8_t>()); break;
        case ValueType::Int8:   out.ints.push_back(c.read<int8_t>()); break;
        case ValueType::UInt16: out.ints.push_back(c.read<uint16_t>()); break;
        case ValueType::Int16:  out.ints.push_back(c.read<int16_t>()); break;
        case ValueType::UInt32: out.ints.push_back(c.read<uint32_t>()); break;
        case ValueType::Int32:  out.ints.push_back(c.read<int32_t>()); break;
        case ValueType::UInt64: out.ints.push_back(static_cast<int64_t>(c.read<uint64_t>())); break;
        case ValueType::Int64:  out.ints.push_back(c.read<int64_t>()); break;
        case ValueType::Bool:   out.ints.push_back(c.read<uint8_t>() != 0 ? 1 : 0); break;
        case ValueType::Float32: out.floats.push_back(static_cast<double>(c.read<float>())); break;
        case ValueType::Float64: out.floats.push_back(c.read<double>()); break;
        case ValueType::String:  out.strings.push_back(c.read_string()); break;
        case ValueType::Array:
            throw std::runtime_error("gguf: nested arrays are not supported");
    }
}

MetaValue read_value(Cursor& c, ValueType vtype) {
    MetaValue v;
    v.type = vtype;
    if (vtype == ValueType::Array) {
        const auto elem_type_raw = c.read<uint32_t>();
        const uint64_t count = c.read<uint64_t>();
        v.elem_type = static_cast<ValueType>(elem_type_raw);
        if (v.elem_type == ValueType::Array) {
            throw std::runtime_error("gguf: nested arrays are not supported");
        }
        for (uint64_t i = 0; i < count; ++i) read_scalar_into(c, v.elem_type, v);
    } else {
        read_scalar_into(c, vtype, v);
    }
    return v;
}

}  // namespace

TypeInfo type_info(int32_t ggml_type) {
    const TypeEntry* e = find_type(ggml_type);
    if (e == nullptr) {
        throw std::runtime_error(
            log::format("gguf: unknown tensor type id %d", ggml_type));
    }
    return TypeInfo{e->block_size, e->type_size, e->name};
}

bool is_known_type(int32_t ggml_type) { return find_type(ggml_type) != nullptr; }

std::string type_name(int32_t ggml_type) {
    const TypeEntry* e = find_type(ggml_type);
    return e != nullptr ? e->name : ("unknown(" + std::to_string(ggml_type) + ")");
}

GgufFile& GgufFile::operator=(GgufFile&& other) noexcept {
    if (this != &other) {
        unmap();
        map_         = other.map_;
        map_size_    = other.map_size_;
        alignment_   = other.alignment_;
        data_offset_ = other.data_offset_;
        tensors_     = std::move(other.tensors_);
        meta_        = std::move(other.meta_);
        other.map_      = nullptr;
        other.map_size_ = 0;
    }
    return *this;
}

GgufFile::~GgufFile() { unmap(); }

void GgufFile::unmap() {
    if (map_ != nullptr) {
        ::munmap(map_, map_size_);
        map_ = nullptr;
    }
    map_size_ = 0;
}

GgufFile GgufFile::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(log::format("gguf: cannot open '%s'", path.c_str()));
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        throw std::runtime_error(log::format("gguf: '%s' is not a regular file", path.c_str()));
    }
    const size_t size = static_cast<size_t>(st.st_size);
    if (size < 4) {
        ::close(fd);
        throw std::runtime_error(log::format(
            "gguf: '%s' is truncated (%zu bytes, smaller than the magic)", path.c_str(), size));
    }

    // mmap(2) refuses a zero-length mapping; a well-formed GGUF is always
    // larger than that (header alone is 24 bytes), so this only guards an
    // input that already failed the size check above in a future edit.
    void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        throw std::runtime_error(log::format("gguf: mmap failed for '%s'", path.c_str()));
    }

    GgufFile file;
    try {
        file.map_      = map;
        file.map_size_ = size;

        const uint8_t* base = static_cast<const uint8_t*>(map);
        Cursor         c(base, size);

        char magic[4];
        c.need(4);
        std::memcpy(magic, base, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0) {
            throw std::runtime_error(log::format(
                "gguf: bad magic (got %02x%02x%02x%02x, want 'GGUF')",
                static_cast<unsigned>(magic[0]) & 0xFF, static_cast<unsigned>(magic[1]) & 0xFF,
                static_cast<unsigned>(magic[2]) & 0xFF, static_cast<unsigned>(magic[3]) & 0xFF));
        }
        c.skip(4);

        const uint32_t version = c.read<uint32_t>();
        if (version != 3) {
            throw std::runtime_error(
                log::format("gguf: unsupported version %u (this reader handles v3 only)", version));
        }

        const uint64_t n_tensors = c.read<uint64_t>();
        const uint64_t n_kv      = c.read<uint64_t>();

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.read_string();
            const auto  raw_type = c.read<uint32_t>();
            MetaValue   value    = read_value(c, static_cast<ValueType>(raw_type));
            file.meta_[std::move(key)] = std::move(value);
        }

        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name          = c.read_string();
            const uint32_t n_dims = c.read<uint32_t>();
            // Bounds-check before resize(): a garbage n_dims (a truncated or
            // corrupt file) must fail as "truncated", not as an attempted
            // multi-gigabyte allocation.
            c.need(static_cast<uint64_t>(n_dims) * sizeof(uint64_t));
            t.dims.resize(n_dims);
            uint64_t n_elements = 1;
            for (uint32_t d = 0; d < n_dims; ++d) {
                t.dims[d] = c.read<uint64_t>();
                n_elements *= t.dims[d];
            }
            t.n_elements = static_cast<size_t>(n_elements);
            t.ggml_type  = static_cast<int32_t>(c.read<uint32_t>());
            t.offset     = c.read<uint64_t>();

            if (!is_known_type(t.ggml_type)) {
                throw std::runtime_error(log::format(
                    "gguf: tensor '%s' has unknown type id %d", t.name.c_str(), t.ggml_type));
            }
            file.tensors_.push_back(std::move(t));
        }

        // general.alignment, if present, must be a UINT32 and a power of
        // two (gguf-py's GGUFReader applies the same check).
        uint32_t alignment = 32;
        const auto align_it = file.meta_.find("general.alignment");
        if (align_it != file.meta_.end()) {
            if (align_it->second.type != ValueType::UInt32 || align_it->second.ints.empty()) {
                throw std::runtime_error("gguf: general.alignment has the wrong type");
            }
            alignment = static_cast<uint32_t>(align_it->second.ints[0]);
        }
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error(
                log::format("gguf: misaligned -- general.alignment %u is not a power of two", alignment));
        }
        file.alignment_ = alignment;

        size_t data_offset = c.pos();
        const size_t rem = data_offset % alignment;
        if (rem != 0) data_offset += alignment - rem;
        file.data_offset_ = data_offset;

        // Every tensor's bytes must fit inside the mapping, and its start
        // must land on an alignment boundary (a GGUF invariant: each
        // tensor's padded size, and therefore every offset after it, is a
        // multiple of `alignment`).
        for (const TensorInfo& t : file.tensors_) {
            if (t.offset % alignment != 0) {
                throw std::runtime_error(log::format(
                    "gguf: misaligned -- tensor '%s' offset %llu is not a multiple of alignment %u",
                    t.name.c_str(), static_cast<unsigned long long>(t.offset), alignment));
            }
            const size_t nbytes = file.bytes(t);
            if (data_offset > size || t.offset > size - data_offset ||
                nbytes > size - data_offset - t.offset) {
                throw std::runtime_error(log::format(
                    "gguf: tensor '%s' runs past EOF (offset %llu, %zu bytes, file is %zu bytes)",
                    t.name.c_str(), static_cast<unsigned long long>(t.offset), nbytes, size));
            }
        }
    } catch (...) {
        file.unmap();
        throw;
    }

    return file;
}

namespace {

bool is_integer_type(ValueType t) {
    switch (t) {
        case ValueType::UInt8: case ValueType::Int8: case ValueType::UInt16: case ValueType::Int16:
        case ValueType::UInt32: case ValueType::Int32: case ValueType::UInt64: case ValueType::Int64:
            return true;
        default:
            return false;
    }
}

}  // namespace

std::optional<int64_t> GgufFile::get_int(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type == ValueType::Array) return std::nullopt;
    if (!is_integer_type(it->second.type) || it->second.ints.empty()) return std::nullopt;
    return it->second.ints[0];
}

std::optional<double> GgufFile::get_float(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type == ValueType::Array) return std::nullopt;
    if (it->second.type != ValueType::Float32 && it->second.type != ValueType::Float64) return std::nullopt;
    if (it->second.floats.empty()) return std::nullopt;
    return it->second.floats[0];
}

std::optional<bool> GgufFile::get_bool(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type != ValueType::Bool || it->second.ints.empty()) {
        return std::nullopt;
    }
    return it->second.ints[0] != 0;
}

std::optional<std::string> GgufFile::get_string(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type != ValueType::String || it->second.strings.empty()) {
        return std::nullopt;
    }
    return it->second.strings[0];
}

std::optional<std::vector<int64_t>> GgufFile::get_int_array(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type != ValueType::Array) return std::nullopt;
    if (!is_integer_type(it->second.elem_type)) return std::nullopt;
    return it->second.ints;
}

std::optional<std::vector<double>> GgufFile::get_float_array(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type != ValueType::Array) return std::nullopt;
    if (it->second.elem_type != ValueType::Float32 && it->second.elem_type != ValueType::Float64) {
        return std::nullopt;
    }
    return it->second.floats;
}

std::optional<std::vector<std::string>> GgufFile::get_string_array(std::string_view key) const {
    const auto it = meta_.find(key);
    if (it == meta_.end() || it->second.type != ValueType::Array) return std::nullopt;
    if (it->second.elem_type != ValueType::String) return std::nullopt;
    return it->second.strings;
}

const MetaValue* GgufFile::raw_meta(std::string_view key) const {
    const auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

const TensorInfo* GgufFile::tensor(std::string_view name) const {
    for (const TensorInfo& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

const uint8_t* GgufFile::data(const TensorInfo& t) const {
    return static_cast<const uint8_t*>(map_) + data_offset_ + t.offset;
}

size_t GgufFile::bytes(const TensorInfo& t) const {
    const TypeInfo& ti = type_info(t.ggml_type);
    return (t.n_elements / ti.block_size) * ti.type_size;
}

}  // namespace lgc::gguf
