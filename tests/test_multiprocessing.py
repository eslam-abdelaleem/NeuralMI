import torch
import torch.multiprocessing as mp
import platform
import pytest

# A simple worker function for the test
def _mp_worker(tensor):
    """Worker function that receives a tensor."""
    # This function will be run in a separate process.
    # The goal is just to ensure it can receive a torch tensor
    # without a shared memory error.
    assert tensor.sum() > -1e5 # A trivial check

@pytest.mark.skipif(platform.system() != "Darwin", reason="This test is specifically for the macOS shared memory issue.")
def test_torch_multiprocessing_spawn():
    """
    Tests that torch.multiprocessing can spawn a worker without error.
    This validates the fix for the 'torch_shm_manager' runtime error.
    """
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # This is fine, it just means the start method has already been set.
        pass

    # The test is successful if this code runs without a RuntimeError.
    tensor = torch.randn(2, 2)
    process = mp.Process(target=_mp_worker, args=(tensor,))
    process.start()
    process.join(timeout=10) # Add a timeout to prevent hangs
    process.terminate() # Ensure the process is cleaned up

    assert process.exitcode == 0, "The worker process should exit cleanly."