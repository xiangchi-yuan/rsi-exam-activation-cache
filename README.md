# Causal Activation Cache Control

Initial RSI-Exam task proposal by [Xiangchi Yuan](https://xiangchi-yuan.github.io/).
Proposed task slug: `causal_activation_cache_control`.

Improve a reusable controller for sequential requests to intermediate values
of a frozen tensor program. Device and host caches have separate capacities.
The controller retains, offloads, loads or reconstructs values through the
dependency graph, minimizing charged arithmetic plus synchronous copy work.
All numerical operators remain fixed. Only the current request, current cache
snapshot, complete graph and public request law are available during inference.

The inherited starter already uses the workload law and marginal shared-ancestor
recovery prices. Further progress concerns joint cache choices, reconstruction
routes, memory pressure, future-use estimates and efficient bounded planning.

## Run the public starter

Python 3.11 is sufficient for this starter; it uses the standard library.

```bash
python3 run_visible.py --split validation
python3 run_visible.py --split train
```

Edit `public/methods/main/solver.py` and helpers, then rerun. Every episode is a
fresh process with initialization and all 96 current-only requests. Feedback
includes complete legality, geometric mean cost, relative improvement, CPU and
wall time. This public feedback tool is not the sealed final grader.

Visible data contain 72 training and 36 validation episodes, plus 24 legacy
training episodes used by the authored reference's value fit. All 24–118-node
scales and four original shape-consistent primitive families are visible, with
sticky, changing-cluster and IID request laws. Inputs and generated data are
original MIT-licensed work. No audio, pretrained weights, external dataset,
GPU or task-side model API is required.

The inference budget is 3 aggregate process CPU seconds per entire episode,
including startup and serialization, and 2 GiB address space. Threads count.
Initial state is only pinned input 0, with no full-graph warmup. Compute peaks
include parents, retained values, output and scratch; copies are charged and
both caches remain capacity constrained. See [instruction.md](public/instruction.md).

## Metric and measured anchors

For legal episode costs C_i, native metric G = exp(mean(log(C_i))) is measured
in analytical work units per 96-request episode; **lower is better**. Any
incomplete, illegal or timed-out episode gives benchmark reward zero. The
inherited baseline defines the normalized floor. An eligible executed reference
defines 0.60; a theoretical cost lower bound defines the performance ceiling.

These anchors are computed on the **36 private prototype evaluation
episodes**, not on the public validation split and not an official RSI score:

| Method/bound | Native geometric mean cost | Normalized prototype reward |
| --- | ---: | ---: |
| Unchanged baseline | 880,268.561936 | 0.000000 |
| Authored, executed reference | 823,633.599582 | 0.600000 |
| Frozen research method | 830,123.795480 | 0.529183 |
| Theoretical cost lower bound | 539,585.258931 | ceiling only |

**Published SOTA: none established for this protocol.** The authored reference
is reported separately and is not presented as a reproduction of published
SOTA. The lower bound is not a feasible optimum and does not quantify attainable
remaining improvement. Exact large-instance optimum: unknown.

All three methods were legal on all 36 certification episodes. The frozen
research method reduced geometric mean cost by 5.697%, with maximum complete
episode CPU 1.065s; the reference reduced cost by 6.434%, maximum CPU 1.400s.
A real Harbor 0.25 separate research/grading reproduction matched every one of
the 36 method costs and reward 0.5291826759, with zero exceptions.

## Difficulty evidence and limitations

An RSI-Exam autoresearch evaluation recorded 44m26s of productive research and
60 experimental records. Its final train/validation reductions were 8.770% and
9.472%, selected using complete train scores only. Validation on the largest
template subset showed 9.802%. Tested approaches included incremental recovery,
cut-dependency fallback, joint device/host candidates, phase-aware value fitting,
deeper sampled continuations, a compact nonlinear critic and one policy-iteration
evaluation. The selected 55-feature critic was fit on 4,346 public training
states. More expressive supervised fits did not improve the native train metric.

The fresh certification gain is smaller than visible validation and remains
below an actually executed stronger reference. This is evidence for transferable
progress and a remaining attainable comparison; it is **not** a twelve-hour
hardness certificate. Official long-horizon frontier calibration is pending.

Earlier offline variants saturated. An earlier online release had reference
35/36 and researcher 29/36 legal cases and both received zero reward; those
failures motivated visible scale coverage and complete-episode CPU reservation.
No failed cases were removed or relabeled as untouched. The new certification
grid was generated after method freeze and all 36 cases retained. All topology
templates are visible; six episodes reused a weighted graph with independent
new request seeds. No full graph-plus-request instance repeated visible data.

## Evidence and proposal

- [Exact form fields 2.1–5.2](PROPOSAL_FIELDS.md)
- [Native anchor calculations](evidence/native_anchors.json)
- [Aggregate prototype evaluation](evidence/certification_summary.json)
- [Research record summary](evidence/pilot_summary.json)
- [Measured public evaluation times](evidence/evaluation_timings.json)

This repository intentionally contains the starter, visible data and aggregate
evidence. It does not contain private certification request tapes/seeds, their
per-case action traces, author-only reference/optimized implementations, runtime
images or personal form contact details. Private reproducibility materials are
available to official reviewers on request. Future official sealing will use a
fresh private grid; used prototype evaluation data are not claimed as untouched.

Status: ready for initial proposal review; no official acceptance or full
frontier difficulty verdict. The private authoring-kit image/format/sealing
checks remain the next stage. Python 3.11, NumPy 2.3.5 and SciPy 1.17.1 are the
proposed runtime stack, baked into images at build time. The inference methods
tested here are standard-library code. The distributable runtime image remains
part of authoring-kit integration.

## Research context and task distinction

[Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616) and
[Checkmate](https://arxiv.org/abs/1910.02653) motivate activation-memory systems.
The current task fixes operators and uses causal sequential demands plus host
offload, rather than a known complete training demand schedule. It is an
analytical systems proxy, not a measurement of GPU training throughput.

Nearest RSI task descriptions concern contraction order, kernel implementation
or semantic memory workflows. This task optimizes residency, copy and reconstruction
of exact numerical values. Public task descriptions were checked; private
implementation overlap and scientific fit remain for official review.

[RSI-Exam contribution guide](https://rsi-exam.ai/contribute.html) ·
[Task guidelines](https://rsi-exam.ai/guidelines.html) · [MIT license](LICENSE)
