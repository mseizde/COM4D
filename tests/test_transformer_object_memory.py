import pytest
import json
import torch
from omegaconf import OmegaConf

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

def test_object_memory_block_ids_config_is_json_serializable_when_disabled():
    block_ids = OmegaConf.create({"ids": [0]}).ids
    model = PartFrameCrafterDiTModel(
        num_attention_heads=4, width=16, in_channels=8, num_layers=1,
        cross_attention_dim=12, max_num_parts=4, max_num_frames=4,
        enable_object_memory=False, object_memory_block_ids=block_ids,
    )

    serialized = json.loads(model.to_json_string())
    assert serialized["object_memory_block_ids"] == [0]


def test_initialize_memory_accepts_half_latents_with_float_model():
    model = _tiny_model(enable_object_memory=True).float()
    memory = model.initialize_object_memory(torch.randn(1, 2, 3, 8).half())

    assert memory.tokens.dtype == model.proj_in.weight.dtype
    assert memory.tokens.shape == (1, 2, 3, model.inner_dim)


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


def test_forward_matrix_reads_persistent_memory():
    torch.manual_seed(0)
    model = _tiny_model(enable_object_memory=True)
    model.object_memory.read_scale.data.fill_(1.0)
    memory = model.initialize_object_memory(torch.randn(1, 2, 3, 8))
    output = model.forward_matrix(
        torch.randn(2, 2, 6, 8),
        torch.ones(1),
        torch.randn(2, 2, 5, 12),
        torch.randn(2, 2, 5, 12),
        static_count=1,
        dynamic_count=2,
        object_memory=memory,
    )
    assert output.sample.shape == (2, 2, 6, 8)
    assert output.object_memory is memory


def test_pose_head_returns_normalized_quaternions():
    model = PartFrameCrafterDiTModel(
        num_attention_heads=4, width=16, in_channels=8, num_layers=1,
        cross_attention_dim=12, max_num_parts=4, max_num_frames=4,
        enable_object_pose_prediction=True,
    )
    output = model(
        hidden_states=torch.randn(2, 6, 8), timestep=torch.ones(2),
        encoder_hidden_states=torch.randn(2, 5, 12),
        attention_kwargs={"num_frames": 2, "num_parts": 1},
    )
    assert output.object_pose.translation.shape == (2, 3)
    assert torch.allclose(
        output.object_pose.rotation.norm(dim=-1), torch.ones(2), atol=1e-5
    )


def test_training_forward_and_sliding_window_memory_lifecycle():
    torch.manual_seed(1)
    model = _tiny_model(enable_object_memory=True)
    memory = model.initialize_object_memory(torch.randn(1, 2, 3, 8))
    relative_pose = torch.zeros(1, 2, 2, 7)
    relative_pose[..., 6] = 1
    train_output = model(
        hidden_states=torch.randn(4, 6, 8),
        timestep=torch.ones(4),
        encoder_hidden_states=torch.randn(4, 5, 12),
        attention_kwargs={"num_frames": 2, "num_parts": 2},
        object_memory=memory,
        memory_evidence=torch.randn(1, 2, 2, 3, 8),
        memory_visibility=torch.ones(1, 2, 2),
        memory_relative_pose=relative_pose,
        update_object_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_output.sample.square().mean().backward()
    assert model.object_memory.read_attention.in_proj_weight.grad is not None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    memory = train_output.object_memory.detached()
    assert not memory.tokens.requires_grad

    for _ in range(2):
        window = model.forward_matrix(
            torch.randn(2, 2, 6, 8),
            torch.ones(1),
            torch.randn(2, 2, 5, 12),
            torch.randn(2, 2, 5, 12),
            static_count=1,
            dynamic_count=2,
            object_memory=memory,
            memory_relative_pose=relative_pose,
        )
        assert window.object_memory is memory
        memory, _ = model.object_memory.update(
            memory,
            model._project_memory_evidence(torch.randn(1, 2, 2, 3, 8)),
            torch.ones(1, 2, 2),
        )
        memory = memory.detached()
