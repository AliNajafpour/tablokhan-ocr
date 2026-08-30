import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

for task in ("detection", "recognition"):
    path = ROOT / "data" / "datasets" / task
    rows = path.joinpath("labels.txt").read_text(encoding="utf-8").splitlines()

    random.Random(42).shuffle(rows)
    cut = int(len(rows) * 0.9)

    path.joinpath("train.txt").write_text("\n".join(rows[:cut]), encoding="utf-8")
    path.joinpath("val.txt").write_text("\n".join(rows[cut:]), encoding="utf-8")

    print(task, cut, len(rows) - cut)
