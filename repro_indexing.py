
import torch

def test_indexing():
    data = torch.randn(1, 200, 5)
    indices = torch.tensor([0], dtype=torch.long)

    print(f"Indices type: {indices.dtype}")

    try:
        res = data[indices]
        print("data[indices] success")
    except Exception as e:
        print(f"data[indices] failed: {e}")

    try:
        res = data[(indices, ...)]
        print("data[(indices, ...)] success")
    except Exception as e:
        print(f"data[(indices, ...)] failed: {e}")

    indices_empty = torch.tensor([], dtype=torch.long)
    try:
        res = data[indices_empty]
        print("data[empty] success")
    except Exception as e:
        print(f"data[empty] failed: {e}")

if __name__ == "__main__":
    test_indexing()
