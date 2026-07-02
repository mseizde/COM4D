import numpy as np
import torch
from PIL import Image

from src.pipelines.pipeline_partcrafter import memory_visibility_from_masks


def _mask(area):
    values = np.zeros((4, 4), dtype=np.uint8)
    values.reshape(-1)[:area] = 255
    return Image.fromarray(values)


def test_amodal_visibility_and_read_only_fallback():
    visible = [[_mask(4), _mask(8)]]
    amodal = [[_mask(8), _mask(16)]]
    result = memory_visibility_from_masks(visible, amodal)
    assert torch.allclose(result, torch.tensor([[0.5, 0.5]]))
    assert memory_visibility_from_masks(visible) is None


def test_trusted_full_area_visibility():
    result = memory_visibility_from_masks([[_mask(4), _mask(8)]], trusted_full_mask_areas=[8])
    assert torch.allclose(result, torch.tensor([[0.5, 1.0]]))
