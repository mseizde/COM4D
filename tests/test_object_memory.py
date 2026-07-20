import pytest
import torch

from src.models.object_memory import (
    CanonicalObjectMemory, ObjectPose, canonical_tokens_from_context,
    frame_major_tensor, relative_object_pose, visibility_from_masks,
)


def test_pose_round_trip_and_quaternion_normalization():
    pose = ObjectPose(torch.tensor([[1.0, -2.0, 0.5]]),
                      torch.tensor([[0.0, 0.0, 0.0, 2.0]]))
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]]])
    assert torch.allclose(pose.world_to_canonical(pose.canonical_to_world(points)), points)


def test_read_preserves_shape_and_starts_as_identity():
    module = CanonicalObjectMemory(channels=16, num_heads=4)
    state = torch.randn(2, 3, 4, 5, 16)
    memory = module.initialize(torch.randn(2, 4, 7, 16))
    output = module.read(state, memory)
    assert output.shape == state.shape
    assert torch.equal(output, state)


def test_reset_parameters_repairs_nonfinite_memory_weights():
    module = CanonicalObjectMemory(channels=16, num_heads=4)
    with torch.no_grad():
        module.read_attention.out_proj.weight.fill_(float("nan"))
        module.read_attention.in_proj_weight[0, 0] = float("inf")
        module.read_scale.fill_(1.0)

    module.reset_parameters()

    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
    assert module.read_scale.item() == 0.0
    assert torch.count_nonzero(module.pose_conditioner[-1].weight) == 0


def test_invisible_evidence_is_no_op():
    module = CanonicalObjectMemory(channels=8, num_heads=2)
    memory = module.initialize(torch.randn(1, 2, 3, 8), confidence=0.2)
    evidence = torch.randn(1, 4, 2, 5, 8)
    updated, gate = module.update(memory, evidence, torch.zeros(1, 4, 2))
    assert torch.equal(updated.tokens, memory.tokens)
    assert torch.equal(updated.confidence, memory.confidence)
    assert torch.count_nonzero(gate) == 0


def test_update_only_visible_object():
    torch.manual_seed(0)
    module = CanonicalObjectMemory(channels=8, num_heads=2)
    memory = module.initialize(torch.randn(1, 2, 3, 8))
    evidence = torch.randn(1, 4, 2, 5, 8)
    visibility = torch.zeros(1, 4, 2)
    visibility[:, :, 0] = 1
    updated, _ = module.update(memory, evidence, visibility)
    assert not torch.equal(updated.tokens[:, 0], memory.tokens[:, 0])
    assert torch.equal(updated.tokens[:, 1], memory.tokens[:, 1])
    assert torch.equal(updated.confidence, torch.tensor([[1.0, 0.0]]))


def test_frame_major_context_selection_and_visibility():
    values = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 1, 2)
    packed, valid = frame_major_tensor(values, torch.tensor([2]), torch.tensor([2]))
    masks = torch.zeros(1, 2, 2, 4, 4)
    masks[:, 0, 0, :2] = 1
    masks[:, 1, 0] = 1
    masks[:, :, 1, :1] = 1
    visibility = visibility_from_masks(masks)
    context = torch.tensor([[[True, True], [False, False]]])
    canonical, confidence = canonical_tokens_from_context(
        packed, visibility, valid=valid, context=context
    )
    assert canonical.shape == (1, 2, 1, 2)
    assert torch.equal(canonical, packed[:, 0])
    assert torch.all(confidence > 0)


def test_pose_prediction_head_backpropagates():
    from src.models.object_memory import ObjectPosePredictionHead
    head = ObjectPosePredictionHead(8)
    pose = head(torch.randn(3, 5, 8))
    loss = pose.translation.square().mean() + pose.rotation.square().mean()
    loss.backward()
    assert head.net[-1].weight.grad is not None


def test_memory_state_detached_and_relative_pose():
    translation = torch.tensor([[[[1., 0., 0.]], [[2., 0., 0.]]]])
    rotation = torch.tensor([[[[0., 0., 0., 1.]], [[0., 0., 0., 1.]]]])
    relative = relative_object_pose(translation, rotation, torch.tensor([[0]]))
    assert torch.allclose(relative[0, :, 0, :3], torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    assert torch.allclose(relative[0, :, 0, 3:], rotation[0, :, 0])
    state = CanonicalObjectMemory.initialize(torch.randn(1, 1, 2, 8, requires_grad=True))
    detached = state.detached()
    assert not detached.tokens.requires_grad
    assert detached.tokens.data_ptr() == state.tokens.data_ptr()


def test_relative_pose_broadcasts_over_batches_frames_and_objects():
    translation = torch.zeros(2, 6, 2, 3)
    translation[0, :, 0, 0] = torch.arange(6, dtype=torch.float32)
    translation[0, :, 1, 1] = torch.arange(6, dtype=torch.float32) + 10
    translation[1, :, 0, 2] = torch.arange(6, dtype=torch.float32) - 3
    translation[1, :, 1, 0] = 2 * torch.arange(6, dtype=torch.float32)
    rotation = torch.zeros(2, 6, 2, 4)
    rotation[..., 3] = 1
    reference_indices = torch.tensor([[1, 4], [3, 2]])

    relative = relative_object_pose(translation, rotation, reference_indices)

    assert relative.shape == (2, 6, 2, 7)
    for batch_index in range(2):
        for object_index in range(2):
            reference = reference_indices[batch_index, object_index]
            expected = (
                translation[batch_index, :, object_index]
                - translation[batch_index, reference, object_index]
            )
            assert torch.allclose(
                relative[batch_index, :, object_index, :3], expected
            )
            assert torch.allclose(
                relative[batch_index, :, object_index, 3:],
                rotation[batch_index, :, object_index],
            )


def test_pose_conditioned_update_backpropagates():
    module = CanonicalObjectMemory(channels=8, num_heads=2)
    module.pose_conditioner[-1].weight.data.normal_()
    memory = module.initialize(torch.randn(1, 1, 2, 8))
    evidence = torch.randn(1, 2, 1, 3, 8)
    relative_pose = torch.tensor([[[[0., 0., 0., 0., 0., 0., 1.]],
                                     [[1., 0., 0., 0., 0., 0., 1.]]]])
    updated, _ = module.update(
        memory, evidence, torch.ones(1, 2, 1), relative_pose=relative_pose
    )
    updated.tokens.sum().backward()
    assert module.pose_conditioner[0].weight.grad is not None


def test_memory_read_is_finite_for_half_precision_frame_state():
    module = CanonicalObjectMemory(channels=8, num_heads=2).float()
    memory = module.initialize(torch.randn(1, 2, 3, 8) * 1000)
    frame_state = (torch.randn(1, 4, 2, 5, 8) * 1000).half()

    result = module.read(frame_state, memory)

    assert result.dtype == frame_state.dtype
    assert torch.isfinite(result).all()
    assert torch.equal(result, frame_state)


def test_fp32_memory_update_accepts_half_precision_evidence():
    module = CanonicalObjectMemory(channels=8, num_heads=2).float()
    memory = module.initialize(torch.randn(1, 1, 2, 8).half())
    evidence = torch.randn(1, 2, 1, 3, 8).half()

    updated, gate = module.update(
        memory,
        evidence,
        torch.ones(1, 2, 1, dtype=torch.float16),
    )

    assert updated.tokens.dtype == torch.float32
    assert gate.dtype == torch.float32
    assert torch.isfinite(updated.tokens).all()


def test_memory_gate_exposes_nonfinite_readout_without_sanitization():
    module = CanonicalObjectMemory(channels=8, num_heads=2).float()
    memory = module.initialize(torch.randn(1, 1, 2, 8))
    frame_state = torch.randn(1, 2, 1, 3, 8).half()

    def nonfinite_attention(query, key, value, need_weights=False):
        readout = torch.full_like(query, float("inf"))
        readout[..., 0] = float("nan")
        return readout, None

    module.read_attention.forward = nonfinite_attention
    with pytest.raises(FloatingPointError, match="output under FP32 math SDPA"):
        module.read(frame_state, memory)
    assert not module.last_read_diagnostics["raw_all_finite"].item()
    assert module.last_read_diagnostics["raw_nonfinite_count"].item() > 0


def test_memory_read_rejects_nonfinite_memory_tokens_before_attention():
    module = CanonicalObjectMemory(channels=8, num_heads=2).float()
    memory = module.initialize(torch.randn(1, 1, 2, 8))
    memory.tokens[..., 0] = float("nan")
    frame_state = torch.randn(1, 2, 1, 3, 8).half()

    with pytest.raises(FloatingPointError, match="attention input"):
        module.read(frame_state, memory)


def test_memory_read_rejects_nonfinite_relative_pose_before_conditioning():
    module = CanonicalObjectMemory(channels=8, num_heads=2).float()
    memory = module.initialize(torch.randn(1, 1, 2, 8))
    frame_state = torch.randn(1, 2, 1, 3, 8).half()
    relative_pose = torch.zeros(1, 2, 1, 7)
    relative_pose[..., 0] = float("nan")

    with pytest.raises(FloatingPointError, match="relative pose"):
        module.read(frame_state, memory, relative_pose=relative_pose)


def test_context_selection_prefers_highest_quality_non_border_frame():
    tokens = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    visibility = torch.ones(1, 3, 1)
    quality = torch.tensor([[[0.9], [0.7], [1.0]]])
    non_border = torch.tensor([[[False], [True], [True]]])
    canonical, confidence, indices = canonical_tokens_from_context(
        tokens, visibility, quality=quality, non_border=non_border,
        return_indices=True,
    )
    assert indices.item() == 2
    assert canonical.item() == 2
    assert confidence.item() == 1.0


def test_context_selection_falls_back_to_earliest_visible_when_all_border():
    tokens = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
    visibility = torch.tensor([[[0.1], [0.8], [1.0]]])
    quality = torch.tensor([[[0.1], [0.4], [0.9]]])
    non_border = torch.zeros(1, 3, 1, dtype=torch.bool)
    canonical, confidence, indices = canonical_tokens_from_context(
        tokens, visibility, quality=quality, non_border=non_border,
        min_visibility=0.2, return_indices=True,
    )
    assert indices.item() == 1
    assert canonical.item() == 1
    assert torch.allclose(confidence, torch.tensor([[0.4]]))
