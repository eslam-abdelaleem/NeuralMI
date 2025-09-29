import torch
import torch.multiprocessing as mp

def worker(tensor):
    """A simple worker function that receives a tensor."""
    print(f"Worker received tensor with sum: {tensor.sum()}")

if __name__ == '__main__':
    # Ensure the 'spawn' start method is used, which is crucial for PyTorch on macOS
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method was already set.")

    # Create a simple tensor
    t = torch.randn(2, 2)
    print(f"Main process created tensor with sum: {t.sum()}")

    # Create a process and pass the tensor to the worker
    p = mp.Process(target=worker, args=(t,))
    p.start()
    p.join()

    print("\nTest finished.")
    # If the script reaches this point without a 'torch_shm_manager' error,
    # the basic environment setup is likely correct.