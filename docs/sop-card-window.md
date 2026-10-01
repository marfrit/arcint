# SOP — the card window

The procedure for any leg that touches a GPU. Host names, unit names and
access paths are NOT here: they live in `CLAUDE.local.md` (git-ignored), as
`CLAUDE.md` requires. This file is the procedure and the card-identity facts.

Every rule below has a dated incident behind it, named in one line. Evidence
class per rule: `measured-here` means this repository measured it on its own
hardware.

---

## 1. The sampler starts BEFORE the leg, and the leg refuses to run without it

A GPU leg runs with a sampler on the **physical host** (not the container)
recording, every 2 s: the leg process's `fdinfo` `drm-resident-gtt` /
`-vram0`, its `VmRSS`, and the physical host's `MemAvailable` / `Shmem` /
ZFS `arcstats size`; with a watchdog that `SIGKILL`s the leg when physical
`MemAvailable` drops below 4 GiB.

- **The leg script starts the sampler itself** and aborts if the sampler's
  log is not growing after 5 s. Starting the two from separate commands is
  a race the leg wins.
- **The sampler's process match is anchored to the interpreter or binary**
  (`^/path/to/python3 tools/x.py`, `^/usr/bin/arcint `), never a bare
  substring. An unanchored `pgrep -f … | head -1` returns the `sudo`/`bash`/
  `timeout` wrapper.
- A cgroup fence does not bound this. Driver / USM-host memory is charged to
  the **physical host**, not the container.

*Incidents (measured-here, 2026-09-15):* a leg launched with the sampler
stopped drove the physical host to `global_oom`; an unanchored sampler
pattern matched the shell wrapper and voided a round's `fdinfo`/`RSS` columns.

## 2. Card identity is established by PCI id, never by number

**DRM numbering is INVERTED relative to OpenVINO numbering on the dev
hardware:**

| DRM node | PCI id | card | OpenVINO device |
|---|---|---|---|
| `card0` | `8086:56a0` | Arc A770, 16 GiB | **GPU.1** |
| `card1` | `8086:e211` | Arc Pro B60, 24 GiB | **GPU.0** |

Anything read per `/sys/class/drm/cardN` or per-card `fdinfo` and reported as
"GPU.N" is attributed to the **wrong card** unless mapped through
`/sys/class/drm/cardN/device/{vendor,device}` first. Confirm the OpenVINO
side from the plugin, not from memory:

```
core.get_property(dev, "FULL_DEVICE_NAME")
core.get_property(dev, "GPU_DEVICE_TOTAL_MEM_SIZE")
```

**`act_freq` is not a contention signal on every node.** The A770's node
returns a constant `act_freq=8517` with `cur_freq` and `max_freq` EMPTY (an
unpopulated interface); the B60's reads sanely (idle 0, `cur` 1200, `max`
2400). Read `fuser /dev/dri/*` and the unit state for contention instead.

*Evidence (measured-here, 2026-09-15):* PCI ids read off both nodes, plugin
enumeration cross-checked the same hour.

## 3. Before the window

- [ ] Cards free: units **inactive AND disabled**, `pgrep` for the serving
      binary empty, `fuser /dev/dri/*` empty.
- [ ] Host quiet: a hang measured while another session holds the host
      localises nothing.
- [ ] Seat announced in the session handoff with a timestamp, the card by
      **name and PCI id**, the tree, and the expected duration.
- [ ] Tree byte-verified both ends (per-file sha256 list, `LC_ALL=C` sorted,
      `cmp`), and the chain hash recorded.
- [ ] The runtime the leg will link named explicitly — two venvs on the dev
      host carry different OpenVINO builds and their numbers are not
      comparable.
- [ ] ARC state recorded at the start: ZFS `arc_c_max` is a large fraction of
      the host and **ARC is not counted in `MemAvailable`**, so a leg begun
      against a warm ARC races its eviction and reads a false ceiling.

## 4. During

- One process per leg; a fresh process per cell where the cell's own answer
  could be poisoned by a previous fault.
- Each leg carries its own hard `timeout -s KILL`.
- Predict before measuring; a cell that cannot fail measures nothing.

## 5. After

- [ ] Zombie sweep **by pid**, `SIGKILL`. A `pkill -f <pattern>` issued
      inside a command line that contains that pattern matches its own chain.
- [ ] Cards released: `pgrep` empty on host and container, device memory
      back, units left as they were found.
- [ ] Release line with a timestamp and leftover count in the handoff.
- [ ] Every number reported names the card, the depth, the KV precision, the
      configuration and the binary.

*Incident (measured-here, 2026-09-15):* a `pkill -f` inside an ssh command
line containing its own pattern killed the connection.

## 6. When the host does not come back

The local journal dies with the box; the kernel log does not. The netconsole
listener on the coordinator host holds it (path in `CLAUDE.local.md`). Check
it FIRST after a freeze — a `global_oom` burst, a panic, or nothing at all
are three different diagnoses and only that log separates them.
