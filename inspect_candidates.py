import os, sys, json
import numpy as np

def show_task(tid):
    path = f"training/{tid}.json"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return
    with open(path, "r") as f:
        data = json.load(f)
    print("=" * 60)
    print(f"TASK: {tid}")
    print("=" * 60)
    for i, pair in enumerate(data["train"]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        print(f"\n--- Train {i} --- (In: {inp.shape}, Out: {out.shape})")
        print("INPUT:")
        print(inp)
        print("OUTPUT:")
        print(out)
    for i, pair in enumerate(data.get("test", [])):
        inp = np.array(pair["input"])
        out = np.array(pair.get("output", []))
        print(f"\n--- Test {i} --- (In: {inp.shape}, Out: {out.shape if len(out) else 'unknown'})")
        print("INPUT:")
        print(inp)
        if len(out):
            print("OUTPUT:")
            print(out)

if __name__ == "__main__":
    tids = sys.argv[1:]
    for tid in tids:
        show_task(tid)
