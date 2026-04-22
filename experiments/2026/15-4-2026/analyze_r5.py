import sys
import json
from pathlib import Path
from collections import defaultdict

try:
    from inspect_ai.log import read_eval_log
except ImportError:
    print("inspect_ai not available")
    sys.exit(1)

log_dir = Path("/Users/jackhopkins/PycharmProjects/ReasonMax/experiments/2026/15-4-2026/logs/step2_readers")

# Find all R5 logs - use glob *R5* but filter to ensure R5 is actually the reader tag
r5_logs = sorted(log_dir.glob("*R5*.eval"))

# Filter out false positives (e.g. R3-t5p matching because of "5p" containing "5")
filtered = []
for p in r5_logs:
    name = p.stem
    if "-R5-" in name or "_R5_" in name or "_R5-" in name or name.endswith("-R5") or name.endswith("_R5"):
        filtered.append(p)
    else:
        print(f"SKIPPING false positive: {p.name}")

r5_logs = filtered
print(f"Found {len(r5_logs)} R5 (Gemma) log files\n")

results = {}
errors = []

for log_path in r5_logs:
    fname = log_path.stem
    parts = fname.split("_")
    task_name = "_".join(parts[1:-1]) if len(parts) > 2 else fname

    try:
        log = read_eval_log(str(log_path), header_only=True)
        task_id = log.eval.task if hasattr(log.eval, "task") else task_name
        status = log.status

        metrics = {}
        if log.results and log.results.scores:
            for score_obj in log.results.scores:
                if score_obj.metrics:
                    for metric_name, metric_val in score_obj.metrics.items():
                        metrics[metric_name] = metric_val.value if hasattr(metric_val, "value") else metric_val

        n_samples = None
        if log.results:
            n_samples = getattr(log.results, "completed_samples", None) or getattr(log.results, "total_samples", None)

        model = str(log.eval.model) if hasattr(log.eval, "model") else "unknown"

        results[task_name] = {
            "task_id": task_id,
            "status": status,
            "model": model,
            "metrics": metrics,
            "n_samples": n_samples,
            "file": fname,
        }

        acc = metrics.get("accuracy", metrics.get("mean", "N/A"))
        print(f"{task_name:50s} status={status:10s} acc={acc}  n={n_samples}  model={model}")

    except Exception as e:
        errors.append((task_name, str(e)))
        print(f"{task_name:50s} ERROR: {e}")

if errors:
    print(f"\n--- Errors ({len(errors)}) ---")
    for name, err in errors:
        print(f"  {name}: {err}")

print()
print("=" * 80)
print("SUMMARY BY TRANSFORM AND GENERATOR")
print("=" * 80)

transform_gen = defaultdict(dict)
for task_name, data in results.items():
    # Remove the reader-c2-R5- prefix
    task_clean = task_name.replace("reader-c2-R5-", "").replace("reader_c2_R5_", "")

    gen = None
    transform = "standard"
    for g in ["G1", "G2", "G3"]:
        if g in task_clean:
            gen = g
            task_clean = task_clean.replace(g, "").strip("-").strip("_")
            break

    if task_clean:
        transform = task_clean.strip("-").strip("_")
    if not transform:
        transform = "standard"

    if gen:
        acc = data["metrics"].get("accuracy", data["metrics"].get("mean", None))
        transform_gen[transform][gen] = {
            "accuracy": acc,
            "status": data["status"],
            "n_samples": data["n_samples"],
        }

transforms_order = ["standard", "t64", "t5p", "mask", "tleak"]
gens = ["G1", "G2", "G3"]

header = f"\n{'Transform':<12}"
for g in gens:
    header += f" {g + ' Acc':>10} {g + ' n':>6}"
print(header)
print("-" * 70)

for transform in transforms_order:
    row = f"{transform:<12}"
    for gen in gens:
        data = transform_gen.get(transform, {}).get(gen, {})
        acc = data.get("accuracy", "N/A")
        n = data.get("n_samples", "N/A")
        status = data.get("status", "")
        if isinstance(acc, float):
            acc_str = f"{acc:.3f}"
        else:
            acc_str = str(acc)
        if status and status != "success":
            acc_str = f"{acc_str}*"
        row += f" {acc_str:>10} {str(n):>6}"
    print(row)

# Also print any transforms not in our expected list
extra = set(transform_gen.keys()) - set(transforms_order)
for transform in sorted(extra):
    row = f"{transform:<12}"
    for gen in gens:
        data = transform_gen.get(transform, {}).get(gen, {})
        acc = data.get("accuracy", "N/A")
        n = data.get("n_samples", "N/A")
        status = data.get("status", "")
        if isinstance(acc, float):
            acc_str = f"{acc:.3f}"
        else:
            acc_str = str(acc)
        if status and status != "success":
            acc_str = f"{acc_str}*"
        row += f" {acc_str:>10} {str(n):>6}"
    print(row)

# Cross-transform mean accuracy
print(f"\n{'Transform':<12} {'Mean Acc (G1+G2+G3)':>20}")
print("-" * 35)
all_transforms = transforms_order + sorted(extra)
for transform in all_transforms:
    accs = []
    for gen in gens:
        data = transform_gen.get(transform, {}).get(gen, {})
        acc = data.get("accuracy")
        if isinstance(acc, (int, float)):
            accs.append(acc)
    if accs:
        mean_acc = sum(accs) / len(accs)
        print(f"{transform:<12} {mean_acc:>20.3f}  (n={len(accs)} generators)")
    else:
        print(f"{transform:<12} {'N/A':>20}")

print("\n--- Non-success logs ---")
any_warning = False
for task_name, data in sorted(results.items()):
    if data["status"] != "success":
        print(f"  {task_name} status={data['status']}")
        any_warning = True
if not any_warning:
    print("  All logs completed successfully.")

# Print all metrics available (for debugging)
print("\n--- All metrics found across logs ---")
all_metrics = set()
for data in results.values():
    all_metrics.update(data["metrics"].keys())
for m in sorted(all_metrics):
    print(f"  {m}")

print("\nDone.")
