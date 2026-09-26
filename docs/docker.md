# Container images

arcint can be packed as a pull-and-run container set the way other Arc
serving stacks are. Two images, because the two halves of the dependency
have very different change rates:

| Tier | Image | Rebuilds when | Build cost |
|---|---|---|---|
| 1 | `arcint-ov` — the **patched** OpenVINO runtime | the patch series or the pin changes | heavy (~35 min) |
| 2 | `arcint` — the engine | every commit | light |

Splitting them is the whole point. Building them together would recompile a
pinned OpenVINO nightly on every engine commit.

The split is *per layer of divergence*, not per backend. The patch series
in `contrib/packaging/marfrit-openvino/patches/` is why a single
"install OpenVINO" Dockerfile is not available: the runtime arcint is
measured against is a source build, not a release.

## Building

From the repo root:

```bash
# tier 1 — the patched OpenVINO base
docker build -t arcint-ov:2026.4.0-ov71640275d29-ceiling0033 \
  -f docker/Dockerfile.openvino-patched .

# tier 2 — the engine on top of it
docker build -t arcint:local \
  --build-arg OV_BASE=arcint-ov:2026.4.0-ov71640275d29-ceiling0033 \
  -f docker/Dockerfile.arcint .
```

Or dispatch `.github/workflows/build-toolbox.yml`, which builds both and
publishes to GHCR under the running repo's own namespace. It takes a
`tiers` input (`ov`, `arcint`, or `all`) and a `channel` input. No
configuration is needed to run it from a fork — the namespace is derived
from `GITHUB_REPOSITORY_OWNER`.

## Tagging: why the tag names a ceiling, not a patch level

The runtime's version string carries a `marfrit-pN` level, but **that stamp
does not identify a patch set from 0046 onward.** The patch README
discloses it explicitly:

> the version stamp stays at `marfrit-p19` (disclosed: `p19` is also the
> 0003-0043 stamp, so the 0046 build is identified by its env/parse
> symbols, not the stamp)

The same string therefore covers several different patch sets. A tag that
names only the level cannot answer "which runtime is in this image", so the
tier-1 tag names the **OpenVINO pin plus the highest patch actually
applied**:

```
2026.4.0-ov71640275d29-ceiling0033
```

The ceiling is also recorded as an OCI label, so a running container can be
asked without opening it:

```bash
docker inspect --format '{{index .Config.Labels "ai.arcint.ov.patchceiling"}}' <image>
```

## Running

Device access follows the standard Intel Arc recipe — the host supplies the
kernel driver and `/dev/dri`, the image supplies the userspace compute
stack:

```bash
docker run --rm -it \
  --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$MODELS:/models:ro" \
  -p 8080:8080 \
  arcint:local \
  --model /models/ov/<ir-dir> --model-id <id> \
  --device GPU.0 --host 0.0.0.0 --port 8080
```

`docker/docker-compose.example.yaml` is the same thing in compose form,
including the writable blob-cache volume — a cold kernel cache turns every
start into a recompile, so make that volume persistent.

Models are never baked into an image. They are mounted read-only.

## What CI covers, and what it cannot

Covered, no GPU required:

- the full device-free unit ladder (`ctest -L unit`), run inside the tier-2 build;
- a stub + `/props` smoke test against the **final** image, before the
  moving channel tag is published. The smoke test has to run inside the
  container — the stub binds `127.0.0.1` in the container's network
  namespace, so a host-side probe connects to nothing.

Not covered, by design: anything that touches a real card. Card-requiring
acceptance cells live in the test ladder (`docs/design-0.3.1-test-ladder.md`)
and need a card. A green container build means the engine compiles against
the patched runtime and serves without one. It is not a claim about served
quality or throughput.

## Known issue: `patches/0037` does not apply

As of this writing `contrib/packaging/marfrit-openvino/patches/0037-moe-hybrid-prefill-split.patch`
on `main` is **corrupt and cannot be applied**:

```
error: corrupt patch at line 436
```

Its last hunk header declares 82 new lines / 6 old; the hunk carries 78 / 3,
and the empty context lines have lost their leading space. `git apply`
refuses the file outright, which means the published recipe cannot build the
runtime the published recipe depends on.

This is why the toolbox pins `PATCH_MAX=0033` rather than building the whole
series. The ceiling is a single build arg; when 0037 is re-exported, bumping
it is a one-line change here.
