// The MTP head file's LM head (src/exec/mtp_head_file.h, llama.cpp patch 0105): a qwen35moe head
// file without output.weight is refused unless the draft vocabulary is set; everything else
// is served as before.
//
// Planned red cases (not run, no build in the session that wrote this): with the refusal returning
// for every file without a vocabulary, the "with a head" and "other architectures" checks fail; with
// the refusal removed, "without output needs the vocabulary" fails.
#include "exec/mtp_head_file.h"
#include "harness.h"

#include <string>
#include <vector>

using namespace lgc;

namespace {

MtpHeadFile file(const std::vector<std::string>& names, const char* arch = "qwen35moe") {
    MtpHeadFile f = mtp_head_file_from_names(names);
    f.arch        = arch;
    return f;
}

const std::vector<std::string> kHead = { "token_embd.weight", "output_norm.weight", "blk.40.attn_q.weight",
                                         "blk.40.nextn.eh_proj.weight" };

}  // namespace

TEST(mtp_head_file_names) {
    CHECK(!file(kHead).has_output);
    CHECK(!file(kHead).has_shared_head);
    std::vector<std::string> with = kHead;
    with.push_back("output.weight");
    CHECK(file(with).has_output);
    std::vector<std::string> shared = kHead;
    shared.push_back("blk.40.nextn.shared_head_head.weight");
    CHECK(file(shared).has_shared_head);
    CHECK(!file(shared).has_output);
    std::vector<std::string> similar = kHead;
    similar.push_back("blk.40.output.weight");          // not the top-level LM head
    CHECK(!file(similar).has_output);
}

TEST(mtp_head_file_without_output_needs_the_vocabulary) {
    const std::string why = mtp_head_file_refusal(file(kHead), /*have_vocab=*/false);
    CHECK(!why.empty());
    CHECK(why.find("--llama-mtp-vocab") != std::string::npos);
    CHECK(mtp_head_file_refusal(file(kHead), /*have_vocab=*/true).empty());
}

TEST(mtp_head_file_with_a_head_is_served_as_before) {
    std::vector<std::string> with = kHead;
    with.push_back("output.weight");
    CHECK(mtp_head_file_refusal(file(with), false).empty());
    CHECK(mtp_head_file_refusal(file(with), true).empty());
    std::vector<std::string> shared = kHead;
    shared.push_back("blk.40.nextn.shared_head_head.weight");
    CHECK(mtp_head_file_refusal(file(shared), false).empty());
}

TEST(mtp_head_file_other_architectures_are_not_touched) {
    CHECK(mtp_head_file_refusal(file(kHead, "qwen4_exp"), false).empty());
    CHECK(mtp_head_file_refusal(file(kHead, ""), false).empty());
}
