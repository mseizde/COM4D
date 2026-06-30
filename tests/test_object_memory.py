import torch

from src.models.object_memory import CanonicalObjectMemory, ObjectPose


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
