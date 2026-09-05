#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// GGUF v3 reader (docs/design-gguf-native.md stage 0). Maps the file
// read-only and parses the header, key-value metadata and tensor table; it
// does not build a graph or interpret any architecture-specific meaning of a
// tensor name. Dequantization lives in gguf_dequant.h, one layer up.
//
// ggml dimension convention (carried through unchanged from the file):
// dims[0] is the fastest-varying axis, i.e. the contraction dimension of a
// `[out, in]` weight; a quantization block runs along dims[0] and never
// crosses a row boundary.
namespace lgc::gguf {

// The GGUF metadata value-type tag (spec: general.alignment and friends).
enum class ValueType : uint32_t {
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
};

// ggml tensor element types this reader recognises by numeric id (the ids
// themselves are the GGUF/ggml wire values, not renumbered here).
enum class GgmlType : int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    MXFP4   = 39,
};

// Block layout of a type this reader can compute byte sizes for. Not every
// id in GgmlType has an entry here -- see type_info()'s comment.
struct TypeInfo {
    size_t      block_size;  // elements per block
    size_t      type_size;   // bytes per block
    const char* name;
};

// Block size / bytes-per-block for a known ggml type id. Throws
// std::runtime_error naming the numeric id when `ggml_type` is not one of
// the types this reader has a byte layout for (design note stage 0: "Types
// not in this table are unknown, refused at open with the numeric id").
TypeInfo type_info(int32_t ggml_type);
bool            is_known_type(int32_t ggml_type);
// Human-readable name for a type id, known or not (for error messages).
std::string     type_name(int32_t ggml_type);

struct TensorInfo {
    std::string           name;
    std::vector<uint64_t> dims;        // ggml order: dims[0] fastest-varying
    int32_t               ggml_type = 0;
    uint64_t              offset    = 0;  // relative to the data section start
    size_t                n_elements = 0; // product of dims
};

// One key-value metadata entry. Scalars occupy index 0 of the matching
// vector; arrays occupy however many elements the file declared. Nested
// arrays are not represented (design note stage 0: "nested arrays not
// needed") -- open() throws if one is encountered.
struct MetaValue {
    ValueType type = ValueType::UInt8;       // Array for an array value
    ValueType elem_type = ValueType::UInt8;  // meaningful only when type == Array
    std::vector<int64_t>     ints;           // integer and bool scalars/arrays, widened
    std::vector<double>      floats;         // float scalars/arrays, widened
    std::vector<std::string> strings;        // string scalars/arrays
};

class GgufFile {
public:
    GgufFile() = default;
    GgufFile(const GgufFile&)            = delete;
    GgufFile& operator=(const GgufFile&) = delete;
    GgufFile(GgufFile&& other) noexcept { *this = std::move(other); }
    GgufFile& operator=(GgufFile&& other) noexcept;
    ~GgufFile();

    // Maps `path` and parses it. Throws std::runtime_error, naming what is
    // wrong, for: a missing file, a bad magic, an unsupported version, a
    // truncated header/metadata/tensor table, an unknown tensor type, a
    // tensor whose bytes run past EOF, or an invalid alignment.
    static GgufFile open(const std::string& path);

    // Typed metadata lookups. Each returns nullopt when the key is absent or
    // is not of the requested shape (a scalar getter on an array key, or
    // vice versa, is a nullopt, not a throw: the caller asked the wrong
    // question, which is not the same as the file being malformed).
    std::optional<int64_t>                  get_int(std::string_view key) const;
    std::optional<double>                   get_float(std::string_view key) const;
    std::optional<bool>                     get_bool(std::string_view key) const;
    std::optional<std::string>              get_string(std::string_view key) const;
    std::optional<std::vector<int64_t>>     get_int_array(std::string_view key) const;
    std::optional<std::vector<double>>      get_float_array(std::string_view key) const;
    std::optional<std::vector<std::string>> get_string_array(std::string_view key) const;
    const MetaValue*                        raw_meta(std::string_view key) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo*              tensor(std::string_view name) const;

    // A pointer into the mapped file at the tensor's data, and its total
    // byte size (block count for its type * bytes per block).
    const uint8_t* data(const TensorInfo& t) const;
    size_t         bytes(const TensorInfo& t) const;

    uint32_t alignment() const { return alignment_; }
    size_t   data_offset() const { return data_offset_; }
    size_t   file_size() const { return map_size_; }

private:
    void unmap();

    void*                                map_       = nullptr;  // mmap base, or nullptr
    size_t                               map_size_  = 0;
    uint32_t                             alignment_ = 32;
    size_t                               data_offset_ = 0;
    std::vector<TensorInfo>              tensors_;
    std::map<std::string, MetaValue, std::less<>> meta_;
};

}  // namespace lgc::gguf
