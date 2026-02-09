
import torch

def test_indexing():
    data = torch.randn(1, 200, 5)

    indices_empty = torch.tensor([], dtype=torch.long)
    try:
        res = data[(indices_empty, ...)]
        print(f"data[(empty, ...)] success. Shape: {res.shape}")
    except Exception as e:
        print(f"data[(empty, ...)] failed: {e}")

if __name__ == "__main__":
    test_indexing()
