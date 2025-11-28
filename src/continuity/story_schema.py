from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class CharacterState:
    char_id: str
    name: str = ""
    aliases: List[str] = field(default_factory=list)

    face_desc: str = ""
    hair: str = ""
    clothing: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)

    emotion: str = ""
    action: str = ""
    pose: str = ""

    confidence: float = 0.5
    mention_count: int = 0
    last_seen_scene_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocationState:
    location_id: str
    name: str = ""
    category: str = ""
    anchors: List[str] = field(default_factory=list)

    lighting: str = ""
    weather: str = ""
    time_of_day: str = ""

    atmosphere: str = ""
    confidence: float = 0.5
    mention_count: int = 0
    last_seen_scene_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PropState:
    prop_id: str
    name: str = ""
    attributes: List[str] = field(default_factory=list)

    holder: str = ""
    status: str = ""
    confidence: float = 0.5
    mention_count: int = 0
    last_seen_scene_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleState:
    visual_style: str = ""
    color_tone: str = ""
    shot_type: str = ""
    camera_angle: str = ""
    camera_motion: str = ""
    mood: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneConstraintState:
    must_keep: Dict[str, Any] = field(default_factory=dict)
    can_change: Dict[str, Any] = field(default_factory=dict)
    priorities: Dict[str, float] = field(default_factory=dict)
    continuity_strength: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAnalysis:
    scene_id: str
    raw_text: str = ""
    normalized_text: str = ""

    entities: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    props: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)

    continuity_signals: Dict[str, Any] = field(default_factory=dict)
    same_as_previous: Dict[str, Any] = field(default_factory=dict)
    allowed_to_change: Dict[str, Any] = field(default_factory=dict)
    continuity_priority: Dict[str, float] = field(default_factory=dict)

    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)