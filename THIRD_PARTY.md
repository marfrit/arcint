# Third-party components

Vendored, single-header, licenses unchanged:

| component | version | license | file |
|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT | `third_party/httplib.h` |
| nlohmann/json | 3.11.3 | MIT | `third_party/json.hpp` |
| minja (chat templating) | vendored | MIT | via `minja/chat-template.hpp` |

Runtime dependency (not vendored): OpenVINO (Apache-2.0), including
`openvino_tokenizers`. Model artifacts are not shipped; IRs exported from
Qwen checkpoints inherit the Qwen Community License.
