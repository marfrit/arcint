# Prefix cache on agent traffic (2026-08-30)

A replay of 80 real agent sessions (`tools/cachesim.py`) against arcint's
cache policy reads 97.3 % of requests hitting and 96.0 % of prompt tokens
served from cache on the agent configuration; 37 % of the remaining prefill
re-prefills sessions evicted by pool pressure (`measured-here`, replay). The
host tier built for it: `--cache-host-mib N` parks an evicted entry's pages in
host RAM and promotes them back on a hit (DESIGN §4.4). The snapshot grid is
`--cache-grid N` (DESIGN §7.0.2l); the default is the prefill chunk.

Full history: `git show b0447b8:docs/prefix-cache-production.md`.
