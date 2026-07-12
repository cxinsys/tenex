import pytest
import torch


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("cuda"):
            item.add_marker(pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA not available"
            ))
