# src/text/scene_packet.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class ContinuityRequirements:
    preserve_character: bool = False
    preserve_location: bool = False
    preserve_object_state: bool = False
    preserve_style: bool = False
    preserve_camera: bool = False
    preserve_action: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenePacket:
    scene_id: int
    scene_text: str

    dependent_on_previous: bool = False
    dependency_type: List[str] = field(default_factory=list)
    dependency_strength: float = 0.0

    characters: List[str] = field(default_factory=list)
    location: str = ""
    time_hint: str = ""
    core_action: str = ""
    camera_style: str = ""

    duration_s: float = 4.0
    scene_role: str = "action"

    continuity_requirements: ContinuityRequirements = field(
        default_factory=ContinuityRequirements
    )

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["continuity_requirements"] = self.continuity_requirements.to_dict()
        return out