import pytest
import torch

from src.models.object_memory import CanonicalObjectMemory
from src.models.transformers import PartFrameCrafterDiTModel


def _tiny_model(enable_object_memory: bool) -> PartFrameCrafterDiTModel:
    return PartFrameCrafterDiTModel(
        num_attention_heads=4,
        width=16,
        in_channels=8,
        num_layers=1,
        cross_attention_dim=12,
        max_num_parts=4,
        max_num_frames=4,
        enable_object_memory=enable_object_memory,
        object_memory_block_ids=[0],
    )


def test_forward_reads_and_returns_updated_object_memory():
    torch.manual_seed(0)
    model = _tiny_model(enable_object_memory=True)
    model.object_memory.read_scale.data.fill_(1.0)
    memory = model.initialize_object_memory(torch.randn(1, 2, 3, 8))
    output = model(
        hidden_states=torch.randn(4, 6, 8),
        timestep=torch.ones(4),
        encoder_hidden_states=torch.randn(4, 5, 12),
        attention_kwargs={"num_frames": 2, "num_parts": 2},
        object_memory=memory,
        memory_evidence=torch.randn(1, 2, 2, 4, 8),
        memory_visibility=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        update_object_memory=True,
    )
    assert output.sample.shape == (4, 6, 8)
    assert output.object_memory is not None
    assert not torch.equal(output.object_memory.tokens[:, 0], memory.tokens[:, 0])
    assert torch.equal(output.object_memory.tokens[:, 1], memory.tokens[:, 1])


def test_forward_rejects_memory_when_feature_is_disabled():
    model = _tiny_model(enable_object_memory=False)
    memory = CanonicalObjectMemory.initialize(torch.randn(1, 2, 3, 16))
    with pytest.raises(ValueError, match="enable_object_memory=False"):
        model(
            hidden_states=torch.randn(4, 6, 8),
            timestep=torch.ones(4),
            encoder_hidden_states=torch.randn(4, 5, 12),
            attention_kwargs={"num_frames": 2, "num_parts": 2},
            object_memory=memory,
        )
