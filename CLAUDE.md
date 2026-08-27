# ligence — session rules

Source of truth: README.md (scope), DESIGN.md (architecture, invariants,
milestones), llm.txt (machine summary). Read DESIGN.md before touching
anything; the invariants in §3.4/§3.8 and the gates in §5 are not negotiable.

## Infra (this session does NOT inherit the fleet CLAUDE.md or its memory)

- **Infra questions → the `his` agent** (Agent tool, subagent_type `his`):
  hosts, wake procedures, containers, MCP endpoints. Ask it instead of
  guessing.
- **Dev/test hardware = dirac** (Proxmox CT130 on host `data`, Ryzen 5700X,
  both Intel cards passed through, xe KMD):
  - Access: `ssh root@data 'pct exec 130 -- sudo -u mfritsche <cmd>'`, or the
    HTTP endpoints directly at `dirac.fritz.box`.
  - **GPU.0 = Arc Pro B60** (24 GB, ~22.7 usable): `openarc-coder.service`
    (user unit, mfritsche) serves `qwen3.6-coder` on :8080, ~60 t/s — the
    performance baseline to beat. **No live consumers currently: this session
    has full autonomy over both cards until further notice** (owner decision
    2026-08-28). Stop/start services as the work requires, no announcements
    needed.
  - **GPU.1 = Arc A770** (16 GB): `llama-agent.service` on :8087 (llama.cpp
    Vulkan, 262k). May be borrowed for experiments:
    `systemctl --user stop llama-agent`, restore after.
  - **The cards are FULL while their services run.** Both units hold their
    model VRAM permanently (B60 ~16 GiB coder IR, A770 the agent model).
    Any ligence process that wants a card must **stop the resident service
    first** — loading next to it fails with allocation errors at best, host
    OOM at worst. Test-window ritual (both cards):

        systemctl --user stop llama-agent      # A770 free
        systemctl --user stop openarc-coder    # B60 free
        ...experiment...
        systemctl --user start openarc-coder llama-agent   # restore when done

    Restore both services at the end of a work session so the fleet returns
    to its normal state (boot defaults stay enabled either way). Verify with
    `curl dirac.fritz.box:8080/v1/models` (coder, ~1 min reload) and
    `:8087/v1/models` (agent).
  - Unit changes on dirac/boltzmann/bosch go through the **Roundhouse MCP**
    (`switch_preview`/`edit_rollout` etc.), NOT hand-edited systemctl/sed.
    Hand edits without a commit in `~/.config/systemd/user` crash roundhouse.
  - OpenVINO venv: `dirac:~/openarc-venv` (OV 2026.4 dev + GenAI). Model IRs
    under `dirac:/models/ov/` (b5 = production coder artifact).
  - SYCL is dead under the xe KMD (memcpy abort); Vulkan works but is slow on
    BMG. OpenVINO's OpenCL path is the only proven fast route on both cards.

## data goes to sleep — hold it

Host `data` (and with it dirac and both GPUs) is **shut down nightly at 03:00
by a hertz cron** (`shutdown-data.sh`) unless a lock says otherwise. For any
session that needs the cards past 03:00:

    ssh hertz 'sudo /opt/herding/bin/hold-data.sh <hours> "ligence dev"'
    ssh hertz 'sudo /opt/herding/bin/hold-data.sh status'   # remaining time
    ssh hertz 'sudo /opt/herding/bin/hold-data.sh frei'     # release

The lock is self-expiring (epoch deadline; forgetting it costs watts, not
mornings) and never overwrites a running foreign lock. If data is off, wake:
**FRITZ!DECT plug, AIN `11630 0190371`** —
`ssh hertz 'sudo /opt/herding/power/plug-switch 116300190371 on'` (scripts
address the plug by AIN, not by name). The boot chain then brings dirac
(`onboot=1`) and its user units up by itself; the coder API answers a few
minutes after power-on.

**If data freezes or dies silently** (GPU experiments can do that): its
kernel log survives on another box. data ships kmsg via **netconsole**
(from .201/nic2, UDP :6666) to the hertz listener —
`ssh hertz 'sudo tail -100 /var/log/boltz-netcon.log'` (shared capture file,
lines carry the source IP; data's are the 192.168.88.201 ones). Check this
FIRST after a freeze; the local journal dies with the box.

## Measurement discipline

- Quality gate: the fleet Prüfstand harness lives at
  `~/claude/csv/` on this machine (`frage.py`, `pruefstand.lua`,
  `toolprobe.py`, `agent-tiefe.py`). 10/10 greedy for the coder artifact is
  the bar; equivalence (cold/warm cache, MTP on/off) must be byte-exact.
- A test must be able to fail (run the red case first), measurements at the
  endpoint that matters, no theatre.
- Services restored at the end of a work session; report what actually happened.

## Model selection for agents / subagents

* Daily Coding & Implementation: Use Sonnet 5 for feature work, routine
  debugging, and most subagent execution; it handles 80-90% of daily
  development tasks efficiently.
* Complex Architecture: Use Opus 5 for thorough code reviews, subtle bug
  detection, and architectural tradeoffs, as these high-value, low-volume
  tasks benefit from its deeper reasoning.
* Simple Subagent Tasks: Use Haiku 4.5 for bounded subagents performing
  searches, summaries, or mechanical file operations to minimize latency and
  cost.
* Hardest Agentic Work: Use Fable 5 for code reviews before pushing.

## Language

Code and docs in English. Conversation follows the user.
