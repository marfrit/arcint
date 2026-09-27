"""STRUCTURAL: every rebuilt paged-lane request gets its inputs re-bound, and the
n-gram staging window is per lane.

WHY (DESIGN 7.0.2cz, 2026-09-27, `code`).
  * `release_kv_pools()` and `reset_lane_request()` rebuild a lane's
    `InferRequest` from the compiled paged model. A fresh request holds none of
    the tensors the load bound to the old one. Both paths re-bound the GDN state
    rows (and the reset path the KV pools) but neither the n-gram table ports:
    a second reservation pass, a failed probe or a failed pre-warm would have
    served every later forward with unbound table ports. No load reached those
    paths on 2026-09-27; the defect was read in the code.
  * The staging window was ONE tensor shared by every lane, and
    `feed_ngram_ports` fills it before the lane's turn: with two lanes, one
    lane's rows could overwrite another's before its infer. Read in the code
    by review; never run with two lanes.

The fix routes every rebuild through `rebind_lane_request(lane, with_kv_pools)`
(state rows, optionally the KV pools, the n-gram tables) and gives each lane its
own staging window (`Lane::ngram_staging`). This file pins that shape
device-free. RED on the tree before the fix: the rebuild cells, the rebind cell
and the per-lane cell fail.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "exec" / "backend_ov.cpp"
CREATE_CALL = "paged_model_.create_infer_request("
REBIND_CALL = re.compile(r"rebind_lane_request\(\s*\*?\w+\s*,\s*/\*with_kv_pools=\*/(true|false)\s*\)\s*;")


def _source():
    """The source with string/char literals emptied and `//` and `/* */`
    comments removed, so braces and quoted code in either cannot mislead the
    scans below (a comment quotes one of the assignments verbatim)."""
    text = SRC.read_text()
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            # keep the keyword comment REBIND_CALL reads: /*with_kv_pools=*/
            chunk = text[i:j + 2]
            out.append(chunk if chunk == "/*with_kv_pools=*/" else " ")
            i = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _block_after(text, pos):
    """The rest of the brace block that encloses `pos`."""
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[pos:i]
            depth -= 1
    return text[pos:]


def _function_named(text, name):
    m = re.search(r"\n\s*(?:void|size_t|bool)\s+" + re.escape(name) + r"\s*\([^)]*\)\s*(?:const\s*)?\{", text)
    assert m, f"no definition of {name} in {SRC.name}"
    return _block_after(text, m.end())


def _create_sites(text):
    return [m.start() for m in re.finditer(re.escape(CREATE_CALL), text)]


def test_paged_requests_are_created_at_exactly_three_sites():
    sites = _create_sites(_source())
    assert len(sites) == 3, (
        f"{CREATE_CALL}) appears {len(sites)} time(s). A new site must re-bind the "
        "request's inputs through rebind_lane_request and be added to this file on purpose.")


def test_the_load_site_precedes_the_first_table_binding():
    text = _source()
    first_site = _create_sites(text)[0]
    assert "lanes_.back()->req" in text[first_site - 40:first_site], "the load-time site moved"
    assert text.find("bind_ngram_ports(rctx") > first_site


def test_every_rebuild_site_rebinds_through_one_function():
    text = _source()
    for pos in _create_sites(text)[1:]:
        rest = _block_after(text, pos)
        assert REBIND_CALL.search(rest), (
            "a request rebuilt after the load is not re-bound by a rebind_lane_request(...) "
            "call: its n-gram table ports (and state rows) would be unbound on the next forward")


def test_the_release_path_leaves_the_old_kv_pools_unbound_and_the_reset_path_binds_them():
    text = _source()
    rel = REBIND_CALL.findall(_function_named(text, "release_kv_pools"))
    rst = REBIND_CALL.findall(_function_named(text, "reset_lane_request"))
    assert rel == ["false"], f"release_kv_pools must rebind WITHOUT the KV pools it frees: {rel}"
    assert rst == ["true"], f"reset_lane_request must rebind WITH the KV pools: {rst}"


def test_the_rebind_covers_the_ngram_tables_pinned_and_staged():
    text = _source()
    body = _function_named(text, "rebind_lane_request")
    assert "bind_ngram_tables(" in body
    assert "la_state_names_" in body
    tables = _function_named(text, "bind_ngram_tables")
    assert "lane.ngram_staging" in tables, "the lane's staging window is not re-bound"
    assert re.search(r"chunks\[i\]\.name\s*,\s*ngram_table_tensors_\[i\]", tables), (
        "the pinned chunks are not re-bound pairwise (chunk i with tensor i)")


def test_the_staging_window_is_per_lane():
    text = _source()
    assert "ngram_staging_tensor_" not in text, "a shared staging tensor is back"
    feed = _function_named(text, "feed_ngram_ports")
    assert "lane.ngram_staging" in feed, "the feed does not fill the lane's own window"
