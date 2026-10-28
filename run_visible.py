"""Visible development feedback: complete current-only process costs."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
sys.path.insert(0, str(PUBLIC))
from selfcheck import one


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "validation"), default="validation")
    p.add_argument("--solver", default="public/methods/main/solver.py")
    p.add_argument("--output")
    args = p.parse_args()
    solver = Path(args.solver)
    if not solver.is_absolute(): solver = ROOT / solver
    cases = json.loads((PUBLIC / "visible" / (args.split + ".json")).read_text())
    start = time.monotonic()
    rows = [one(case, solver, PUBLIC) for case in cases]
    elapsed = time.monotonic() - start
    valid = len(rows) == len(cases) and all(row["valid"] for row in rows)
    summary = dict(split=args.split, episodes=len(cases), valid=sum(row["valid"] for row in rows),
                   all_valid=valid, wall_seconds=elapsed,
                   metric="geometric_mean_episode_work_cost", unit="analytical work units per episode",
                   direction="lower is better", formal_hidden_score=False)
    if valid:
        gm = math.exp(sum(math.log(row["cost"]) for row in rows) / len(rows))
        base = {row["id"]: row["cost"] for row in json.loads(
            (ROOT / "evidence/visible_baseline_costs.json").read_text())[args.split]}
        gain = sum(math.log(base[row["id"]] / row["cost"]) for row in rows) / len(rows)
        summary.update(geometric_mean_cost=gm, geometric_speedup=math.exp(gain),
                       geometric_cost_reduction=1-math.exp(-gain),
                       total_cost=sum(row["cost"] for row in rows),
                       max_policy_cpu_seconds=max(row["cpu_seconds"] for row in rows))
    else:
        summary["benchmark_reward_if_any_failure"] = 0
    out = Path(args.output) if args.output else ROOT / "results" / (args.split + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
