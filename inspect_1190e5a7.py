from thot_arc_core import ArcTask

def inspect_single(task_id: str):
    fpath = f"Atelier_ARC_Prize/training/{task_id}.json"
    task = ArcTask.load_from_file(fpath)
    print(f"=== {task_id} ===")
    for i, p in enumerate(task.train_pairs):
        print(f"Train {i+1} in ({p['input'].shape}):")
        for r in p["input"]: print(" ".join(map(str, r)))
        print(f"Train {i+1} out ({p['output'].shape}):")
        for r in p["output"]: print(" ".join(map(str, r)))
        
inspect_single("1190e5a7")
