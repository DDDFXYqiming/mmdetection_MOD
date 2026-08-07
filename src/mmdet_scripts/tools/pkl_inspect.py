"""查看 pkl 预测结果（原 read_pkl.py）。"""

import pickle


def inspect_pkl(path: str, limit: int = 3, verbose: bool = True) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    total = len(data)
    print(f"pkl 共 {total} 条记录")
    for i, item in enumerate(data[:limit]):
        if verbose:
            print(f"--- [{i}] ---")
            for key, value in item.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: tensor shape={tuple(value.shape)}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: list len={len(value)}")
                else:
                    print(f"  {key}: {value}")
    return {"total": total, "shown": min(limit, total)}
