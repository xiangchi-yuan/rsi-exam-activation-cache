# Online activation-cache control under device and host memory limits

Improve the inherited cache controller for a frozen tensor program. Requests
for intermediate activations arrive sequentially. Keep useful values on the
device, offload them to a limited host cache, or discard and recompute them.
Minimize total charged arithmetic plus data-copy work while completing every
request. A new independent request stream is used for final evaluation.

This models multi-head or conditional consumers of the same frozen input batch
and model version. Values remain valid throughout one episode; caches reset
between episodes. The graphs, sizes and costs are known. The exact future
request sequence, random seed and next latent cluster are not provided.
The workload law and allowed targets are public inputs available to all methods.

The benchmark uses original shape-consistent dense, ReLU, residual, pooling,
upsampling and shared-projection programs. Costs are explicitly analytical
arithmetic/copy equivalents, not measured GPU timing. Everything runs on CPU;
there are no downloaded models, audio, training data assets or task-side APIs.

## Method interface

Edit `methods/main/solver.py` and packaged helpers:

```python
def initialize(public):
    return Policy(public)

class Policy:
    def step(self, observation):
        # observation = {"target": integer, "state": current_cache_snapshot}
        return [["compute", node_id], ..., ["commit", observation["target"]]]
```

`public` contains a topologically ordered `nodes` array, `memory`, `host_memory`,
`request_support`, `target_groups`, `request_distribution`, and `horizon`.
Each node declares `parents`, `size`, `scratch`, `cost`, and one-way
`transfer_cost`. Node 0 is the initially pinned input. No other tensor is
initially resident, and there is no compulsory whole-graph warmup.

The current snapshot contains only `device`, `host`, and `committed`.
Your strategy may maintain causal counters or estimates from observed requests.
It may simulate hypothetical requests using the public law. It may not inspect
an episode file, seed, scoring process, future stream or another episode during
inference. Training against the visible episodes is permitted during research.
Frozen parameters can be packaged beside the method.

## Actions and exact accounting

- `compute(i)`: node i must be absent from the device and all parents present.
  Pay its arithmetic cost. Parents, all retained tensors, its output and its
  scratch space must fit simultaneously.
- `free(i)`: remove an existing device value; input 0 is pinned.
- `store(i)`: copy a device value to host. Both copies remain until released.
  Pay one transfer cost and obey host capacity.
- `load(i)`: copy an existing host value to device, paying a transfer cost.
  The host copy remains. The device copy must be absent before loading.
- `drop_host(i)`: release a host copy.
- `commit(i)`: i must equal the current requested target and be device resident.
  It must be the final action in this response. There is one commit per request.

Represent each action as `[kind, integer_index]`. Source 0 cannot be computed,
freed, stored or loaded. There is no implicit movement, aliasing, compression,
free recomputation or asynchronous overlap. Operations are pure and fixed.
The trusted runtime owns caches and charges; simulated snapshots in your code
cannot invent real resident values.

## Resource and artifact contract

Every episode starts a fresh process. The full initialization and request loop,
including Python startup and serialization, has 3 CPU seconds and 2 GiB virtual
memory. Threads count toward the same CPU/address-space limit. Policy inference
cannot spawn a separate process, access the network, inspect another process,
or change its filesystem root. These restrictions apply equally to references.
Visible fitting/search may use ordinary research scripts outside inference.

Inference files are read-only. Runtime caches and scratch state must stay in
policy memory; protocol stdout/stderr are permitted. No state may
persist between episodes. Submit reusable `.py` code and optional small
`.json`/`.npy`/`.npz` parameters under `methods/main/`; `.md`/`.txt` documentation
is allowed. Do not include symlinks, opaque executables or bytecode caches.
Dependencies are Python 3.11, NumPy 2.3.5 and SciPy 1.17.1.

## Visible feedback and scoring

The development package has 72 training and 36 validation episodes, with four
program families and bursty, changing-cluster and independent requests.
Use `python selfcheck.py --split validation` for complete feedback, and
`--split train` for fitting. Full visible scores are comparable; partial
smoke tests are not full scores.

Native objective is total charged arithmetic plus transfers, lower is better.
The final native comparison uses equal-weight geometric cost ratios across
episodes. All episodes must complete legally; any invalid, missing or timed-out
episode gives zero benchmark reward. Failures are never removed from the data.
Starter is the zero anchor. An actually executed frozen stronger reference
defines 0.60 when calibration establishes a positive aggregate improvement.
The sum of unique required-ancestor computation costs is a theoretical lower
bound, not necessarily an attainable optimum under memory or causality.
The exact scoring map and release anchors are frozen before final comparison.

The inherited starter already uses the declared workload law and marginal
shared-ancestor recovery prices. Further improvements should address joint
device/host choices, constrained reconstruction, prefetch risk and efficient
planning. A simple LRU controller is not the benchmark starter.

## Research record

Keep `experiment_log.md` with hypotheses, complete visible costs, runtime,
failures and retain/revert decisions. Preserve numbered method versions.
The task supports an eventual up-to-12-hour research budget; pilot evaluations have
their explicitly recorded budgets and are not automatically formal RSI scores.
The real aim is transferable improvement, not memorizing visible request tapes.

## Visible scale and final grid (v2)

Visible data explicitly include all train/dev/held-out graph templates, from roughly 24 to 118 nodes. Plan for that full range under the same 3 CPU-second inference budget. Final evaluation has 36 independently generated held-out-scale graphs/streams (four families x three laws x three replicates), 96 requests each; device fraction .12, host/device ratio .50, transfer factor48, sticky .80, within-cluster repeat .35, phase16. Those resource/law settings occur in training. New private request seeds are independent of graph seeds; templates can repeat but a stored visible request tape cannot predict a new episode. A legal low-cost plan should always be available before expensive alternatives; inference must reserve resources to serve the entire remaining episode. Full resource and legality checks are part of the objective. V1 exposed scale-related planning overruns and was retained with reward0; v2 does not reuse that test.

An additional 24 legacy training episodes are available in visible/legacy_train.json for fitting; the main train self-check still covers the full 72-episode scale grid. The inherited reference value estimator was fit only on this now-public legacy training data.
