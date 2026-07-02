from dataclasses import dataclass
from typing import Optional

import torch

from ..object_memory import ObjectMemoryState, ObjectPose


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor
    object_memory: Optional[ObjectMemoryState] = None
    object_pose: Optional[ObjectPose] = None
