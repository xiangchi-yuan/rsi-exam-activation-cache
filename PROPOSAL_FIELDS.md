# RSI-Exam Task Proposal: technical fields

Task-specific answers for fields 2.1–5.2. Personal fields 1.1–1.5 are supplied
separately by the author. Prototype results are evidence for initial assessment,
not official acceptance or a published SOTA claim.

## 2.1 Task title

causal_activation_cache_control

## 2.2 Domain and field

Machine Learning / ML Systems / Activation Memory Management and Online
Stochastic Resource Control

## 2.3 Task description

The submitted controller starts from a working cache strategy for a frozen
tensor program and
improves how it retains, offloads and reconstructs intermediate activations.
Requests arrive sequentially; device and host memory have separate capacities.
The complete DAG, tensor/scratch sizes, operation/copy costs, target support and
request law are public, while inference receives only the current target and
cache state. Numerical operators and input/model version remain fixed.

The starter already uses public-law forecasts and marginal shared-ancestor
recovery prices. It is weak at coupled eviction, host placement and reconstruction
choices under transient memory pressure. We expect gains from better future-use
valuation, joint cache/recovery planning and efficient complete-episode budgeting.
The objective is to minimize charged arithmetic plus synchronous copy work while
serving every request legally within a three-CPU-second inference budget.

## 2.4 Metric

Validity-gated geometric mean episode work cost, measured in analytical work
units per 96-request episode; lower is better. For valid costs C_i, G =
exp(mean(log(C_i))). A trusted replay charges every compute and copy and checks
dependencies, both capacities, simultaneous output/scratch peaks and commits.
Any invalid, incomplete or timed-out episode gives benchmark reward zero.

For legal submissions, d = log(G_baseline/G_method) is mapped to one reward:
zero for d <= 0; 0.6*d/r for 0 < d <= r; min(1, 0.6 + 0.4*(d-r)/(u-r)) above
r. Here r is the executed frozen reference gain and u is the gain to the
theoretical cost lower bound. No human judgment enters scoring. Native G and
the normalized reward are reported separately; absolute cost sums are diagnostic.

## 2.5 Visible and hidden data

All data come from original shape-consistent dense/ReLU/residual/concatenation/
pooling/upsampling tensor programs and declared stochastic request generators.
Visible: 72 training and 36 validation episodes covering 24–118 nodes, four
program families and three laws, each with 96 requests. Another 24 legacy
training cases used by the authored reference's value fit are also public.
The self-check supplies complete costs, legality and full-process CPU.

Prototype hidden certification: 36 private evaluation episodes on the complete
predeclared four-family x three-law x three-replicate grid, generated using fresh
independent 128-bit graph/request seeds after the researcher method froze. All
cases were retained. All topology templates are visible; six episodes reused a
weighted graph but received independent new request tapes, and zero full
graph-plus-request instances repeated visible data. Inference never receives
an episode ID, seed, future request tape or another episode's state.

Those prototype evaluation inputs are now used evidence and remain outside this
public repository. Official sealing should generate a fresh 36-case private
grid after design/reference freeze, keep it outside the research environment,
and grade
each submitted artifact from scratch under the current-only protocol.

## 2.6 Source

Original task code and visible data:
https://github.com/xiangchi-yuan/rsi-exam-activation-cache

Research context:
- Dynamic Tensor Rematerialization: https://arxiv.org/abs/2006.09616
- Checkmate: https://arxiv.org/abs/1910.02653
- RSI task guidelines: https://rsi-exam.ai/guidelines.html
- Public task inventory: https://rsi-exam.ai/tasks.html
- Evaluation infrastructure: https://github.com/aiming-lab/RSI-Exam

The papers motivate the systems problem; their implementations/numbers are
not claimed as reproduced SOTA for this new protocol. No third-party dataset
or pretrained model weights are used.

## 2.7 Data licence

MIT. The code and generated program/request data are original and may become
part of the benchmark. Python/NumPy/SciPy retain their own licenses; runtime
binaries are excluded from this source/data repository. No proprietary,
unpublished or embargoed source dataset is required.

## 3.1 Anchors

Native metric: geometric mean analytical work units per episode, lower is better.
Measured/computed on all 36 private prototype evaluation episodes:

baseline = 880268.561936
published SOTA = none established for this protocol
executed authored competitive reference = 823633.599582
theoretical cost lower bound = 539585.258931
exact attainable optimum = none / unknown

Theoretical "ceiling" is a lower bound on cost because the metric is minimized:
for each episode, sum computation costs of the union of required ancestors, then
take the geometric mean across episodes. It may be unattainable due to memory
and causality; it is not an estimated optimum. Values above are deterministic
calculations from saved costs, not estimates. Normalized 0/0.6/1 are not native
metric anchors.

## 3.2 Can you reproduce the SOTA number yourself?

Not applicable / no established published SOTA for this protocol. We can
reproduce the authored reference reported separately in 3.1: it is legal on
36/36 certification episodes with native G = 823633.599582, 6.434% geometric
cost reduction and normalized reward 0.600. It was frozen before the research
pilot and ran under the same resource/input constraints. Private code and
certification data can be provided to official reviewers on request.

## 3.3 Why won't this saturate?

The starting method already captures easy gains from using the request law and
shared-ancestor prices, so replacing LRU is not the task. Cache values depend
jointly on other resident values, device/host placement, feasible reconstruction
order, remaining horizon and stochastic future reuse. Independent tensor prices
and one-step decisions miss these interactions. Improving the policy involves
both better planning/value estimates and making them efficient within the fixed
whole-episode CPU budget; no ready library call supplies the whole controller.

The frozen research method improved visible and new-stream performance but
remained below an executed stronger reference on the fixed new test: reward
0.529 versus 0.600. Further directions include per-node survival estimates,
deterministic multi-step law planning and tighter coupled spill/reconstruction.
We do not yet guarantee a twelve-hour gap or large-graph optimality. Official
long-horizon frontier calibration is requested, with revision if it saturates.

## 3.4 Evidence of difficulty

An RSI-Exam autoresearch evaluation did 44m26s of actual productive research and
retained 60 experimental records; it did not pad unused budget. Complete train
scores selected a 55-feature phase-sensitive value critic fit on 4,346 public
training states, incremental DAG recovery and bounded joint cache candidates
with a complete cheap fallback. Tested alternatives included survival/residual
values, host packing, deeper sampled continuations, reconstruction order,
a compact nonlinear critic and one offline policy-iteration evaluation.

Final visible train/validation: 72/72 and 36/36 legal, with 8.770% and 9.472%
geometric cost reduction. Largest-template validation: 12/12, 9.802% reduction.
Fresh certification after method freeze: 36/36, 5.697% reduction, G =
830123.795480, reward 0.529, max whole-episode CPU 1.065s. Reference: 36/36,
6.434% reduction, reward 0.600, max CPU 1.400s. A Harbor reproduction with separate research/
grading environments matched every one of the 36 method costs and the exact
same reward with zero exceptions; it is reproduction, not extra fresh evidence.

The evaluation reached a plateau among the tested algorithms and critics;
supervised-error improvements from nonlinear fitting did not improve native
train cost. Earlier variants and failures are retained: a 17m33s v1 phase had
127 records but 29/36 method and 35/36 reference legal test cases, hence reward
zero for both. This motivated visible scale coverage and resource reservation.
These phases are shorter than a full twelve-hour run and do not certify it.

A six-node exact finite-horizon causal witness additionally solves 196 states
with a public IID law, horizon 24: expected optimum 181.9989307423; realized
mean 182 versus starter 220.671875 over 128 streams. This shows attainable
small-case construction-order headroom, not the optimum gap on larger graphs.

## 3.5 Known shortcuts or traps

Future/seed leakage, invented residency/cost reports, ignored scratch/output
peaks, uncharged copies, memorized visible request tapes, missing completion,
cross-episode persistence and fork/thread CPU-accounting bypass are the main
traps. The trusted parent owns the future tape and replays each action before
sending the next current target. A fresh process/cache is used each episode;
inference files are read-only, source types/symlinks checked, and process/
filesystem/IPC restrictions apply. Empty or incomplete output fails, while
forced cold reconstruction is legal but much worse than the competent baseline.
Actual OCI tests exercised memory/aggregate-thread-CPU/fork limits and metadata/
IPC attempts. These bounded checks do not replace official sealing review.

## 4.1 Compute tier

CPU-only. Select the closest CPU-only research tier in the form; proposed
research/feedback container allocation is 8 vCPU and 16 GiB RAM. This is a
requested configuration, not a claim that existing tests had that exact quota.

## 4.2 GPU requirement

Leave blank (CPU-only).

## 4.3 CPU and memory

Proposed research container: 8 vCPU / 16 GiB RAM, Linux x86_64.
Separate strict inference contract: one process per 96-request episode,
3 aggregate CPU seconds including initialization/startup/serialization and
2 GiB address space; threads count. The inference memory limit is not the
whole research-container RAM allocation.

## 4.4 How long does one evaluation take?

Measured in a Linux x86_64 CPU environment, excluding image builds: unchanged starter
full validation (36 episodes) = 2.835s wall time; full train (72 episodes) =
5.501s. Those complete runs are reproduced by run_visible.py. An end-to-end
Harbor frozen-method reproduction on 36 used certification cases took about
68s including orchestration/artifact handoff/fresh verifier execution. Method
runtime can vary; the hard per-episode limit is 3 CPU seconds. Feedback is
comfortably fast enough for many rounds in twelve hours.

## 4.5 Packages needed, and any that must be banned

Python 3.11. The starter and frozen inference methods use the standard library.
NumPy 2.3.5 and SciPy 1.17.1 are proposed for CPU fitting and numeric diagnostics,
baked into the image at build time. No task-side model/API, pretrained assets or
GPU stack is required. The distributable runtime image/build route should be
aligned in the official authoring kit.

No general mathematical solver library is claimed to trivialize this protocol,
so no specific such library is banned. Access to sealed future seeds/request
tapes or answer tables is forbidden. Inference is single-process, current-only,
read-only and nonpersistent; submitted artifacts are Python code and small
JSON/NumPy parameters, not opaque executables. Official sealing should enforce
no network in both phases and the declared process/resource boundary.

## 5.1 Link to code or data

https://github.com/xiangchi-yuan/rsi-exam-activation-cache

Public starter, visible data, instructions and aggregate local evidence. Private
reference/optimized code, private certification tapes and full traces can be
provided to official reviewers on request; they are not in this public repo.

## 5.2 Attachments

Optional: public starter/data bundle and aggregate experiment evidence. A
separate author-review archive contains private inputs and should be supplied
only through an official private review channel. No passwords or credentials
are included.
