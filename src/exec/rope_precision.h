#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/unsqueeze.hpp>

// M11 §2 follow-up (DESIGN §7.0.2ag's own discriminator, ARCINT_DRAFT_F32,
// fired: forcing both drafter graphs to f32 recovers acceptance at 76k). This
// is the production, per-node fix -- mark the rotary-position-embedding
// subgraph so the plugin's own ConvertPrecision pass leaves it in its
// original precision instead of folding it into the GPU default f16, the
// way DESIGN §7.0.2ag already reads the overflow: cos/sin(pos * inv_freq)
// with inv_freq[0] == 1.0, so the trig argument's magnitude equals the
// absolute position, and f16 tops out at 65,504 -- export_dflash.py:51-54
// already names this for its own residual stream (a separate, already-fixed
// overflow); export_mtp.py carries no equivalent scaling at all, and NEITHER
// exporter marks its rotary subgraph, so both hit this one. The base graph
// survives at the same depth because its own position chain is
// ShapeOf-derived, and the GPU plugin's own mixed-precision marking already
// keeps a ShapeOf-fed subgraph in f32 -- both drafters instead take
// positions as a plain Parameter, which nothing marks, so the plugin folds
// their rotary chain to f16 like everything else in the graph.
//
// The rt_info API: the name this fix was originally asked for,
// `ov::disable_fp16_compression`, is DEPRECATED and is not a linkable symbol
// in either OpenVINO build this repository touches -- measured with
// `nm -DC` against BOTH this repository's dev toolchain (the OpenVINO
// Python wheel arcint's build pulls headers from: OpenVINO 2026.3.1,
// `libopenvino.so.2631`) and the packaged `marfrit-openvino` runtime
// /usr/bin/arcint actually loads at start (OpenVINO 2026.4.0, build
// 2026.4.0-22849-71640275d29, `libopenvino.so.2640`): neither exports
// `ov::disable_fp16_compression` as a callable symbol (only the deprecated
// RTTI class `DisableFP16Compression` exists, as an internal rt_info
// payload type, not a function), but BOTH export its documented
// replacement, `ov::disable_conversion(node, element::Type)` -- the
// dependency was checked to hold across both versions, 2026.3.1 (dev) and
// 2026.4.0 (deployed), not just one of them. This is not a guess: it is
// literally the function OpenVINO's own upstream
// `MarkRopeInputsToKeepInMixedPrecision` pass calls today
// (src/common/transformations/src/transformations/common_optimizations/
// mark_rope_input_to_keep_in_mixed_precision.cpp, read off the side-loaded
// full OpenVINO source tree kept on this fleet for an unrelated
// plugin-patching campaign) -- the SAME mechanism the plugin's
// ConvertPrecision pass honours, under its current, non-deprecated name.
//
// Failure mode if a future OpenVINO version drops or renames this symbol:
// the real exposure is build-here, run-there, not a link-time-vs-runtime
// distinction within one build. arcint links against the DEV toolchain's
// libopenvino (2026.3.1, above) at build time; the binary that actually
// ships runs against the separately-built, separately-versioned packaged
// marfrit-openvino (2026.4.0, above) instead. Nothing in arcint's own
// build step re-links or re-checks against that production library, so a
// future marfrit-openvino nightly that drops or renames
// `ov::disable_conversion` would build and test clean here and only fail
// once deployed. When it does fail, the mechanism compounds the exposure:
// the reference is resolved via lazy PLT binding, not eagerly at link
// time, so even a from-scratch build against the changed library would
// succeed, and the failure would only surface at first call -- loading an
// MTP or DFlash drafter -- as a dynamic linker symbol-lookup error. That
// is not a C++ exception, so it is NOT caught by any
// `catch (const std::exception&)` handler in this file or its callers;
// the process aborts. This is a known, accepted risk of depending on an
// internal (non-public-API) OpenVINO symbol, not a gap in this file's own
// error handling; the two-version `nm -DC` check above is the mitigation
// this file can actually offer -- confirming the symbol on both ends of
// the skew this comment describes, not eliminating the skew itself.
//
// The header that declares it
// (transformations/rt_info/disable_precision_conversion.hpp) is internal to
// OpenVINO's own build and is not shipped with the public OpenVINO Runtime
// SDK this repository links against -- neither the dev toolchain nor the
// packaged marfrit-openvino carries it under their include/ trees, even
// though the symbol itself is exported from libopenvino.so in both. The two
// declarations below are vendored to match that already-linked ABI exactly
// (std::shared_ptr<ov::Node> / ov::element::Type, both ordinary public
// types), not a reimplementation of the mechanism -- a forward declaration
// of a function this repository does not define and does not own the
// header for. This is a real dependency on an internal OpenVINO symbol, not
// part of the versioned public API surface: a future OpenVINO upgrade that
// changes this symbol's signature (or stops exporting it) needs this file
// re-checked against `nm -DC` on the new library before anything here is
// trusted again.
namespace ov {
void disable_conversion(const std::shared_ptr<Node>& node, const element::Type& to);
bool is_conversion_disabled(const std::shared_ptr<const Node>& node, const element::Type& to);
}  // namespace ov

namespace lgc {

// Structural detection, not name-based (the exporters name ports
// differently and drift is exactly what this repository's own record
// shows -- DESIGN.md's export drift entries). Starting from every
// Parameter in the model: walk forward through Convert/Unsqueeze/Concat
// only (the "through Unsqueeze/Convert/Concat" chain the design names)
// looking for a Multiply CANDIDATE whose OTHER input is a Constant -- the
// inv_freq table. A candidate is confirmed as the actual RoPE angle
// multiply (pos * inv_freq) only once a Cos or Sin sits downstream of it,
// possibly through further passthrough ops (measured: the DFlash IR
// concatenates the angle with itself before Cos/Sin) -- "has a Constant
// input" alone is not enough on its own:
// an ordinary Parameter*Constant multiply elsewhere in the graph (a scale
// factor, say) is structurally identical to the real one by that test
// alone, and this file's own red-first test caught exactly that false
// positive before the Cos/Sin confirmation was added. A Parameter with no
// CONFIRMED angle multiply downstream is not the positions input and
// contributes nothing. From each confirmed angle multiply: the Cos/Sin
// nodes that directly consume it, then the "rotate-half" Multiply/Add
// nodes that go on to consume THOSE (the nodes that fold cos/sin into the
// query or key tensor) -- followed a bounded number of hops so an
// unrelated Multiply/Add elsewhere in the graph is never swept in.
//
// Returns every node on that path, each exactly once, in no particular
// order. An empty result means the pattern was not found (exporter drift,
// not a crash) -- the caller logs the count either way.
inline std::vector<std::shared_ptr<ov::Node>> find_rope_nodes(
    const std::shared_ptr<ov::Model>& model) {
    using ov::Node;

    std::vector<std::shared_ptr<Node>> found;
    if (!model) return found;

    auto contains = [](const std::vector<std::shared_ptr<Node>>& v, const std::shared_ptr<Node>& n) {
        return std::find(v.begin(), v.end(), n) != v.end();
    };
    auto mark = [&](const std::shared_ptr<Node>& n) {
        if (!contains(found, n)) found.push_back(n);
    };
    auto is_passthrough = [](const std::shared_ptr<Node>& n) {
        return ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v0::Unsqueeze>(n) ||
               ov::is_type<ov::op::v0::Concat>(n);
    };
    // A Constant feeding a Multiply directly, OR through exactly one Convert
    // (measured on the real MTP layer IR: inv_freq is stored at its own
    // width and cast with an explicit Convert before the angle Multiply --
    // `has_constant_input` must see through that one hop or it never
    // recognizes the real angle multiply at all).
    auto is_constant_ish = [](const std::shared_ptr<Node>& n) {
        if (ov::is_type<ov::op::v0::Constant>(n)) return true;
        if (ov::is_type<ov::op::v0::Convert>(n) && n->get_input_size() == 1) {
            return ov::is_type<ov::op::v0::Constant>(n->get_input_node_shared_ptr(0));
        }
        return false;
    };
    auto has_constant_input = [&](const std::shared_ptr<Node>& n) {
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            if (is_constant_ish(n->get_input_node_shared_ptr(i))) return true;
        }
        return false;
    };
    // Subtract joins Multiply/Add here: rotate_half's two components
    // combine with opposite signs (measured: Subtract_207 pairs with
    // Add_210 on the real MTP layer IR, each combining a Cos-branch and a
    // Sin-branch Multiply into one rotated half) -- the design's own
    // "Multiply/Add" wording names the shape of the pattern, not an
    // exhaustive op list, and Subtract is structurally the same
    // combination step.
    auto is_combine = [](const std::shared_ptr<Node>& n) {
        return ov::is_type<ov::op::v1::Multiply>(n) || ov::is_type<ov::op::v1::Add>(n) ||
               ov::is_type<ov::op::v1::Subtract>(n);
    };
    // The walk from Cos/Sin to the nodes that actually apply them may cross
    // a passthrough op first (measured: Cos/Sin each feed an Unsqueeze --
    // reshaping the trig table's broadcast shape -- before the Multiply
    // that combines it with the query/key data), so this walk's own hop
    // predicate is passthrough-OR-combine, not combine alone.
    auto is_passthrough_or_combine = [&](const std::shared_ptr<Node>& n) {
        return is_passthrough(n) || is_combine(n);
    };

    for (const auto& param : model->get_parameters()) {
        // Forward BFS from this one Parameter, through Convert/Unsqueeze/
        // Concat only, collecting every Multiply-with-a-Constant reached
        // that way as a CANDIDATE -- confirmed as the angle multiply only
        // once we know it feeds a Cos or Sin directly (below). A
        // Parameter*Constant Multiply that goes nowhere near a Cos/Sin
        // (e.g. an ordinary scale multiply elsewhere in the graph) is
        // structurally indistinguishable from the real angle multiply by
        // "has a Constant input" alone -- this repository's own red-first
        // test for this file caught exactly that false positive (a decoy
        // Multiply(param, const) with no Cos/Sin downstream) before this
        // extra check was added.
        std::vector<std::shared_ptr<Node>> passthrough_chain;
        std::vector<std::shared_ptr<Node>> angle_candidates;
        std::vector<std::shared_ptr<Node>> seen{param};
        std::vector<std::shared_ptr<Node>> queue{param};

        while (!queue.empty()) {
            const auto n = queue.back();
            queue.pop_back();
            for (auto& out : n->outputs()) {
                for (auto& in : out.get_target_inputs()) {
                    const auto consumer = in.get_node()->shared_from_this();
                    if (contains(seen, consumer)) continue;
                    if (is_passthrough(consumer)) {
                        seen.push_back(consumer);
                        passthrough_chain.push_back(consumer);
                        queue.push_back(consumer);
                    } else if (ov::is_type<ov::op::v1::Multiply>(consumer) &&
                               has_constant_input(consumer)) {
                        seen.push_back(consumer);
                        angle_candidates.push_back(consumer);
                    }
                }
            }
        }

        // Confirm each candidate: does a Cos or Sin sit downstream, through
        // passthrough ops only (measured: the DFlash IR concatenates the
        // angle Multiply with itself -- Concat -- before Cos/Sin, so
        // "feeds Cos/Sin directly" is too strict; the MTP layer IR feeds
        // them directly, a 0-hop case of the same walk). Only a confirmed
        // candidate's own trig nodes AND the passthrough nodes crossed to
        // reach them are collected -- an unconfirmed candidate contributes
        // nothing, including none of the front passthrough chain (a
        // Parameter with ONLY an unrelated candidate downstream is not the
        // positions input at all).
        std::vector<std::shared_ptr<Node>> angle_multiplies;
        std::vector<std::vector<std::shared_ptr<Node>>> angle_trig;
        std::vector<std::vector<std::shared_ptr<Node>>> angle_bridge;
        for (const auto& candidate : angle_candidates) {
            std::vector<std::shared_ptr<Node>> trig;
            std::vector<std::shared_ptr<Node>> bridge;
            std::vector<std::shared_ptr<Node>> c_seen{candidate};
            std::vector<std::shared_ptr<Node>> c_queue{candidate};
            while (!c_queue.empty()) {
                const auto n = c_queue.back();
                c_queue.pop_back();
                for (auto& out : n->outputs()) {
                    for (auto& in : out.get_target_inputs()) {
                        const auto consumer = in.get_node()->shared_from_this();
                        if (contains(c_seen, consumer)) continue;
                        c_seen.push_back(consumer);
                        if (ov::is_type<ov::op::v0::Cos>(consumer) || ov::is_type<ov::op::v0::Sin>(consumer)) {
                            trig.push_back(consumer);
                        } else if (is_passthrough(consumer)) {
                            bridge.push_back(consumer);
                            c_queue.push_back(consumer);
                        }
                    }
                }
            }
            if (!trig.empty()) {
                angle_multiplies.push_back(candidate);
                angle_trig.push_back(trig);
                angle_bridge.push_back(bridge);
            }
        }

        if (angle_multiplies.empty()) continue;  // not the positions parameter

        for (const auto& n : passthrough_chain) mark(n);

        for (size_t ai = 0; ai < angle_multiplies.size(); ++ai) {
            const auto& angle = angle_multiplies[ai];
            mark(angle);

            for (const auto& b : angle_bridge[ai]) mark(b);
            const auto& trig = angle_trig[ai];
            for (const auto& t : trig) mark(t);

            // The rotate-half application: a bounded forward walk from each
            // Cos/Sin, following passthrough-or-combine nodes (measured on
            // the real MTP layer IR: Cos/Sin -> Unsqueeze [passthrough,
            // reshaping the broadcast] -> Multiply [combine, with the
            // query/key data] -> Subtract/Add [combine, joining the two
            // rotated halves]). Four hops covers that whole shape -- deep
            // enough to reach the Subtract/Add the design names, not so
            // deep it wanders into the general output reshaping
            // (Unsqueeze/Concat/Reshape) that follows every combine node
            // on the real IR and is unrelated to RoPE specifically. Only
            // combine nodes are MARKED here; a passthrough node crossed on
            // the way is not (Cos/Sin's own reshape is not itself part of
            // "the rotate-half Multiply/Add", even though the walk has to
            // pass through it to reach them).
            std::vector<std::shared_ptr<Node>> frontier = trig;
            std::vector<std::shared_ptr<Node>> combine_seen = trig;
            for (int hop = 0; hop < 4 && !frontier.empty(); ++hop) {
                std::vector<std::shared_ptr<Node>> next;
                for (const auto& n : frontier) {
                    for (auto& out : n->outputs()) {
                        for (auto& in : out.get_target_inputs()) {
                            const auto consumer = in.get_node()->shared_from_this();
                            if (contains(combine_seen, consumer)) continue;
                            if (is_passthrough_or_combine(consumer)) {
                                if (is_combine(consumer)) mark(consumer);
                                combine_seen.push_back(consumer);
                                next.push_back(consumer);
                            }
                        }
                    }
                }
                frontier = next;
            }
        }
    }
    return found;
}

// Marks every node find_rope_nodes finds with ov::disable_conversion(node,
// ov::element::f16) -- keeping the rotary subgraph in its original (f32)
// precision through the plugin's own ConvertPrecision pass. Returns how
// many nodes were marked (0 if the pattern was not found).
inline size_t mark_rope_no_fp16_compression(const std::shared_ptr<ov::Model>& model) {
    const auto nodes = find_rope_nodes(model);
    for (const auto& n : nodes) ov::disable_conversion(n, ov::element::f16);
    return nodes.size();
}

}  // namespace lgc
