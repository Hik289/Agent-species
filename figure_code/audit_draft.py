"""Check saved data and print task success and cost per success."""
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest(path, expected_count):
    entries = load(path)
    if len(entries) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} entries")
    for name, entry in entries.items():
        artifact = path.parent / entry["path"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch")
        yield name, load(artifact)


def main():
    matrix = {}
    for _, cell in check_manifest(DATA / "matrix_cells_manifest.json", 16):
        name = cell["geno"]
        solved = sum(cell["per_task"])
        if not cell["n"] == len(cell["per_task"]) == len(cell["per_task_cost"]) == len(cell["splits"]):
            raise ValueError(f"{name}: task counts do not agree")
        if abs(solved / cell["n"] - cell["SR"]) > 0.00005:
            raise ValueError(f"{name}: success rate does not match the saved task outcomes")
        if abs(sum(cell["per_task_cost"]) / solved - cell["CoP"]) > 0.00005:
            raise ValueError(f"{name}: cost per success does not match the saved task costs")
        matrix[name] = cell
    if len(matrix) != 16:
        raise ValueError("Duplicate matrix entries")

    traces = load(DATA / "evo_runs/convergence_traj.json")
    for name, run in check_manifest(DATA / "evo_runs/manifest.json", 5):
        observed = [max(member["SR"] for member in row["front"]) for row in run["generations"]]
        if observed != traces[name]:
            raise ValueError(f"{name}: search curve does not match the saved generations")

    print("Checked 16 planner–executor pairs and five saved search runs.")
    baseline = load(DATA / "final_audit.json")["raw_checks"]["n250_single-opus"]
    print(f"Single Opus: {baseline['successes']}/{baseline['n']}, "
          f"${baseline['CoP_from_vectors']:.4f} per success (saved totals)")
    for name in ("opus->sol", "sol->opus", "opus->opus"):
        row = matrix[name]
        print(f"{name}: {sum(row['per_task'])}/{row['n']}, "
              f"${row['CoP']:.4f} per success")


if __name__ == "__main__":
    main()
