# Decision: Q6_K and Q5_K weights are never computed through an int8 requantised copy

**Date:** 2026-11-26. **Decided by:** the operator. **Status:** in force.

## The decision

Q6_K and Q5_K weights are not computed through an int8 requantised copy: not as a default, and not as an opt-in
switch. The path is not built.

## The rationale, in the operator's words

> "Q6_K und Q5_K nicht per int8 rechnen und das als Entscheidung im Sinne der Korrektheit für die Nachwelt
> dokumentieren. Hier gilt: die paar Prozent decode sind mir den (vermutlich sogar kumulativen) Fehler nicht wert."

In English: do not compute Q6_K and Q5_K through int8, and document that as a decision made for correctness, for the
record. The few percent of decode are not worth the error, which is probably even cumulative.

## What was given up

An int8 path for Q6_K and Q5_K was the structural lever the kernel record names for the fp16 Q5_K / Q6_K GEMMs, which
run at bandwidth and not at the cores (`contrib/llama.cpp/README.md`, patch 0132: "the real gap ... needs a structural
change, an int8 path for those types"). The estimate for the dense 27B was 3-4 % of decode. That figure is an
inference from the device profile's shares, never measured on a built path.

## The reasoning

- Requantising a Q6_K or Q5_K weight to int8 is lossy: each block's values are rounded a second time, onto a
  coarser grid than the one the model was quantised to.
- The error is likely to compound. It enters every product of every layer on every token, and a long generation
  feeds its own outputs back in. A loss that is small per token need not stay small per answer.
- KL on short text can hide that damage. A mean KL inside the answer-level bar on a few thousand tokens measures the
  next-token distribution, not what an agent does over a long tool loop. The agent-level checks (a real task, loop
  metrics, valid calls, a blind review of the result; the method of the P27b and P27s windows) are where such a
  defect shows. A lossy arm that passes KL has not shown it is harmless.

## Scope

- **Lossless gains** keep the standing rule: no size threshold. A change that keeps the arithmetic's class (the same
  values, a different order or layout) is taken when it measures faster and passes the answer-level bar.
- **Lossy arms elsewhere** (for example an int8 activation arm of a GEMM) stay opt-in at most. Before any of them can
  become a default, it needs the agent-level check above, not only KL. This record does not remove existing opt-in
  arms; it fixes that none of them becomes a default on KL alone.
- This record covers requantising Q6_K and Q5_K weights. It does not reopen kernels that were already shipped with
  their own evidence (for example the integer dot products on int8 activations that the series takes from llama.cpp,
  or Q8_0, whose weights are int8 already); those keep their own records.
