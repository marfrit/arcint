// The top-level patches/ directory mirrors the series the runtime recipe
// applies (contrib/packaging/marfrit-openvino/patches/); it fell five patches
// behind between 0.3.0 and 0.4.1 with nothing to say so. Every patch of the
// recipe's series must be present at the top level, byte for byte; the top
// level may carry more (the never-applied records 0001 and 0002).
#include "harness.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {
std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
}  // namespace

TEST(the_top_level_patches_directory_mirrors_the_recipes_series) {
    namespace fs = std::filesystem;
    const fs::path root   = std::string(ARCINT_SOURCE_DIR);
    const fs::path series = root / "contrib/packaging/marfrit-openvino/patches";
    const fs::path mirror = root / "patches";
    CHECK(fs::is_directory(series));
    CHECK(fs::is_directory(mirror));
    size_t patches = 0;
    for (const auto& entry : fs::directory_iterator(series)) {
        if (entry.path().extension() != ".patch") continue;
        ++patches;
        const fs::path twin = mirror / entry.path().filename();
        if (!fs::exists(twin)) {
            std::printf("missing at the top level: %s\n", entry.path().filename().c_str());
            CHECK(fs::exists(twin));
            continue;
        }
        if (slurp(entry.path()) != slurp(twin)) {
            std::printf("differs from the recipe's: %s\n", entry.path().filename().c_str());
            CHECK(slurp(entry.path()) == slurp(twin));
        }
    }
    CHECK(patches >= 21);  // 0003 .. 0023 as of 0.4.1
}
