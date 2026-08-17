from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Final, TypeAlias


NormalizedPoint: TypeAlias = tuple[float, float]
IntegerDelta: TypeAlias = tuple[int, int]

LEGACY_INPUT_PLAN_SCHEMA_VERSION: Final = 1
CAMERA_INPUT_PLAN_SCHEMA_VERSION: Final = 2
DIRECT_BUTTON_INPUT_PLAN_SCHEMA_VERSION: Final = 3
INPUT_PLAN_SCHEMA_VERSION: Final = 4
DEFAULT_INPUT_TRACK_ID: Final = "track-main"
DEFAULT_INPUT_TRACK_NAME: Final = "主轨道"
MAX_TEXT_LENGTH: Final = 128
MAX_SOURCE_EVENT_IDS: Final = 64
HARD_MAX_EVENT_COUNT: Final = 10_000
HARD_MAX_TRACK_COUNT: Final = 64
HARD_MAX_TIME_MS: Final = 86_400_000
FIRST_VERSION_MAX_EVENT_COUNT: Final = 500
FIRST_VERSION_MAX_TOTAL_DURATION_MS: Final = 60_000
FIRST_VERSION_MAX_HOLD_DURATION_MS: Final = 5_000
FIRST_VERSION_MAX_MOUSE_MOVE_DURATION_MS: Final = 5_000
FIRST_VERSION_MAX_MOUSE_UPDATE_RATE_HZ: Final = 240
FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS: Final = 50_000
DEFAULT_MAX_MOUSE_UPDATE_RATE_HZ: Final = 240
FIRST_VERSION_MAX_RELATIVE_DELTA_PER_AXIS: Final = 32_767
FIRST_VERSION_MAX_WHEEL_DELTA_PER_AXIS: Final = 1_200
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _text(value: object, name: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    output = value.strip()
    if not output:
        raise ValueError(f"{name} must be non-empty")
    if len(output) > max_length:
        raise ValueError(f"{name} exceeds {max_length} characters")
    if "\x00" in output:
        raise ValueError(f"{name} cannot contain NUL")
    return output


def _identifier(value: object, name: str) -> str:
    output = _text(value, name)
    if _SAFE_IDENTIFIER.fullmatch(output) is None:
        raise ValueError(
            f"{name} must contain only letters, digits, dot, underscore, or hyphen"
        )
    return output


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _optional_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum, maximum=maximum)


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _normalized_point(value: object, name: str) -> NormalizedPoint:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        if number < 0.0 or number > 1.0:
            raise ValueError(f"{name} coordinates must be between 0 and 1")
        output.append(number)
    return output[0], output[1]


def _integer_delta(value: object, name: str) -> IntegerDelta:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    output: list[int] = []
    for item in value:
        output.append(
            _integer(
                item,
                name,
                minimum=-(2**31),
                maximum=2**31 - 1,
            )
        )
    return output[0], output[1]


def _utc_timestamp(value: object, name: str) -> str:
    output = _text(value, name, max_length=40)
    if not output.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{output[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return output


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(f"{value[:-1]}+00:00")


def utc_now_iso() -> str:
    """Return a stable UTC timestamp suitable for the private plan schema."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _strict_mapping(
    value: object,
    *,
    context: str,
    expected_keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be a JSON object")
    mapping = value
    keys = frozenset(mapping)
    unknown = sorted(keys - expected_keys)
    missing = sorted(expected_keys - keys)
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{context} is missing keys: {', '.join(missing)}")
    return mapping


def _json_array(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


class InputPlanEventType(str, Enum):
    WAIT = "WAIT"
    KEY_DOWN = "KEY_DOWN"
    KEY_UP = "KEY_UP"
    MOUSE_BUTTON_DOWN = "MOUSE_BUTTON_DOWN"
    MOUSE_BUTTON_UP = "MOUSE_BUTTON_UP"
    MOUSE_BUTTON_DOWN_DIRECT = "MOUSE_BUTTON_DOWN_DIRECT"
    MOUSE_BUTTON_UP_DIRECT = "MOUSE_BUTTON_UP_DIRECT"
    MOUSE_MOVE_ABSOLUTE = "MOUSE_MOVE_ABSOLUTE"
    MOUSE_MOVE_RELATIVE = "MOUSE_MOVE_RELATIVE"
    CAMERA_MOVE_RELATIVE = "CAMERA_MOVE_RELATIVE"
    MOUSE_WHEEL = "MOUSE_WHEEL"


_DIRECT_MOUSE_BUTTON_EVENT_TYPES: Final = frozenset(
    {
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
    }
)
_MOUSE_BUTTON_DOWN_TO_UP: Final = {
    InputPlanEventType.MOUSE_BUTTON_DOWN: InputPlanEventType.MOUSE_BUTTON_UP,
    InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT: (
        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT
    ),
}
_MOUSE_BUTTON_UP_TO_DOWN: Final = {
    up: down for down, up in _MOUSE_BUTTON_DOWN_TO_UP.items()
}
_MOUSE_BUTTON_EVENT_TYPES: Final = frozenset(
    {*_MOUSE_BUTTON_DOWN_TO_UP, *_MOUSE_BUTTON_UP_TO_DOWN}
)


_RELATIVE_MOVE_EVENT_TYPES: Final = frozenset(
    {
        InputPlanEventType.MOUSE_MOVE_RELATIVE,
        InputPlanEventType.CAMERA_MOVE_RELATIVE,
    }
)
_MOUSE_MOVE_EVENT_TYPES: Final = frozenset(
    {
        InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
        *_RELATIVE_MOVE_EVENT_TYPES,
    }
)


class InputPlanSource(str, Enum):
    MANUAL = "MANUAL"
    RECORDED = "RECORDED"
    IMPORTED = "IMPORTED"


class MouseButton(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MIDDLE = "MIDDLE"
    X1 = "X1"
    X2 = "X2"


class MouseInterpolation(str, Enum):
    LINEAR = "LINEAR"
    EASE_IN = "EASE_IN"
    EASE_OUT = "EASE_OUT"
    EASE_IN_OUT = "EASE_IN_OUT"
    SMOOTHSTEP = "EASE_IN_OUT"


@dataclass(frozen=True, slots=True)
class InputPlanSafetyLimits:
    """Per-plan limits that may only tighten the experiment hard bounds."""

    max_event_count: int = FIRST_VERSION_MAX_EVENT_COUNT
    max_total_duration_ms: int = FIRST_VERSION_MAX_TOTAL_DURATION_MS
    max_hold_duration_ms: int = FIRST_VERSION_MAX_HOLD_DURATION_MS
    max_mouse_move_duration_ms: int = FIRST_VERSION_MAX_MOUSE_MOVE_DURATION_MS
    max_mouse_update_rate_hz: int = DEFAULT_MAX_MOUSE_UPDATE_RATE_HZ
    max_relative_delta_per_axis: int = FIRST_VERSION_MAX_RELATIVE_DELTA_PER_AXIS
    max_wheel_delta_per_axis: int = FIRST_VERSION_MAX_WHEEL_DELTA_PER_AXIS
    allowed_normalized_min: NormalizedPoint = (0.0, 0.0)
    allowed_normalized_max: NormalizedPoint = (1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_event_count",
            _integer(
                self.max_event_count,
                "max_event_count",
                minimum=1,
                maximum=FIRST_VERSION_MAX_EVENT_COUNT,
            ),
        )
        for name, maximum in (
            (
                "max_total_duration_ms",
                FIRST_VERSION_MAX_TOTAL_DURATION_MS,
            ),
            (
                "max_hold_duration_ms",
                FIRST_VERSION_MAX_HOLD_DURATION_MS,
            ),
            (
                "max_mouse_move_duration_ms",
                FIRST_VERSION_MAX_MOUSE_MOVE_DURATION_MS,
            ),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    minimum=1,
                    maximum=maximum,
                ),
            )
        object.__setattr__(
            self,
            "max_mouse_update_rate_hz",
            _integer(
                self.max_mouse_update_rate_hz,
                "max_mouse_update_rate_hz",
                minimum=1,
                maximum=FIRST_VERSION_MAX_MOUSE_UPDATE_RATE_HZ,
            ),
        )
        for name, maximum in (
            (
                "max_relative_delta_per_axis",
                FIRST_VERSION_MAX_RELATIVE_DELTA_PER_AXIS,
            ),
            (
                "max_wheel_delta_per_axis",
                FIRST_VERSION_MAX_WHEEL_DELTA_PER_AXIS,
            ),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    minimum=1,
                    maximum=maximum,
                ),
            )
        minimum = _normalized_point(
            self.allowed_normalized_min,
            "allowed_normalized_min",
        )
        maximum = _normalized_point(
            self.allowed_normalized_max,
            "allowed_normalized_max",
        )
        if minimum[0] > maximum[0] or minimum[1] > maximum[1]:
            raise ValueError("allowed normalized minimum cannot exceed maximum")
        object.__setattr__(self, "allowed_normalized_min", minimum)
        object.__setattr__(self, "allowed_normalized_max", maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_event_count": self.max_event_count,
            "max_total_duration_ms": self.max_total_duration_ms,
            "max_hold_duration_ms": self.max_hold_duration_ms,
            "max_mouse_move_duration_ms": self.max_mouse_move_duration_ms,
            "max_mouse_update_rate_hz": self.max_mouse_update_rate_hz,
            "max_relative_delta_per_axis": self.max_relative_delta_per_axis,
            "max_wheel_delta_per_axis": self.max_wheel_delta_per_axis,
            "allowed_normalized_min": list(self.allowed_normalized_min),
            "allowed_normalized_max": list(self.allowed_normalized_max),
        }

    @classmethod
    def from_dict(cls, value: object) -> InputPlanSafetyLimits:
        expected = frozenset(cls.__dataclass_fields__)
        mapping = _strict_mapping(
            value,
            context="safety_limits",
            expected_keys=expected,
        )
        return cls(
            max_event_count=mapping["max_event_count"],
            max_total_duration_ms=mapping["max_total_duration_ms"],
            max_hold_duration_ms=mapping["max_hold_duration_ms"],
            max_mouse_move_duration_ms=mapping["max_mouse_move_duration_ms"],
            max_mouse_update_rate_hz=mapping["max_mouse_update_rate_hz"],
            max_relative_delta_per_axis=mapping["max_relative_delta_per_axis"],
            max_wheel_delta_per_axis=mapping["max_wheel_delta_per_axis"],
            allowed_normalized_min=_point_from_json(
                mapping["allowed_normalized_min"],
                "allowed_normalized_min",
            ),
            allowed_normalized_max=_point_from_json(
                mapping["allowed_normalized_max"],
                "allowed_normalized_max",
            ),
        )


_LEGACY_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "offset_ms",
        "event_type",
        "enabled",
        "source_event_ids",
        "key",
        "virtual_key",
        "scan_code",
        "is_extended",
        "button",
        "position",
        "delta",
        "wheel_delta",
        "duration_ms",
        "update_rate_hz",
        "interpolation",
    }
)
_EVENT_FIELDS = _LEGACY_EVENT_FIELDS | {"track_id"}


@dataclass(frozen=True, slots=True)
class InputPlanEvent:
    """One structurally valid atomic event at a track-local time offset."""

    event_id: str
    offset_ms: int
    event_type: InputPlanEventType
    enabled: bool = True
    source_event_ids: tuple[str, ...] = ()
    key: str | None = None
    virtual_key: int | None = None
    scan_code: int | None = None
    is_extended: bool = False
    button: MouseButton | None = None
    position: NormalizedPoint | None = None
    delta: IntegerDelta | None = None
    wheel_delta: IntegerDelta | None = None
    duration_ms: int | None = None
    update_rate_hz: int | None = None
    interpolation: MouseInterpolation | None = None
    track_id: str = DEFAULT_INPUT_TRACK_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "offset_ms",
            _integer(
                self.offset_ms,
                "offset_ms",
                minimum=0,
                maximum=HARD_MAX_TIME_MS,
            ),
        )
        if not isinstance(self.event_type, InputPlanEventType):
            raise TypeError("event_type must be an InputPlanEventType")
        object.__setattr__(self, "enabled", _strict_bool(self.enabled, "enabled"))
        if type(self.source_event_ids) is not tuple:
            raise TypeError("source_event_ids must be a tuple")
        if len(self.source_event_ids) > MAX_SOURCE_EVENT_IDS:
            raise ValueError(
                f"source_event_ids cannot exceed {MAX_SOURCE_EVENT_IDS} entries"
            )
        source_ids = tuple(
            _identifier(item, "source_event_id") for item in self.source_event_ids
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_event_ids cannot contain duplicates")
        object.__setattr__(self, "source_event_ids", source_ids)

        if self.key is not None:
            object.__setattr__(self, "key", _text(self.key, "key"))
        object.__setattr__(
            self,
            "virtual_key",
            _optional_integer(
                self.virtual_key,
                "virtual_key",
                minimum=0,
                maximum=0xFFFF,
            ),
        )
        object.__setattr__(
            self,
            "scan_code",
            _optional_integer(
                self.scan_code,
                "scan_code",
                minimum=0,
                maximum=0xFFFF,
            ),
        )
        object.__setattr__(
            self,
            "is_extended",
            _strict_bool(self.is_extended, "is_extended"),
        )
        if self.button is not None and not isinstance(self.button, MouseButton):
            raise TypeError("button must be a MouseButton")
        if self.position is not None:
            object.__setattr__(
                self,
                "position",
                _normalized_point(self.position, "position"),
            )
        if self.delta is not None:
            object.__setattr__(self, "delta", _integer_delta(self.delta, "delta"))
        if self.wheel_delta is not None:
            object.__setattr__(
                self,
                "wheel_delta",
                _integer_delta(self.wheel_delta, "wheel_delta"),
            )
        object.__setattr__(
            self,
            "duration_ms",
            _optional_integer(
                self.duration_ms,
                "duration_ms",
                minimum=1,
                maximum=HARD_MAX_TIME_MS,
            ),
        )
        object.__setattr__(
            self,
            "update_rate_hz",
            _optional_integer(
                self.update_rate_hz,
                "update_rate_hz",
                minimum=1,
                maximum=1_000,
            ),
        )
        if self.interpolation is not None and not isinstance(
            self.interpolation,
            MouseInterpolation,
        ):
            raise TypeError("interpolation must be a MouseInterpolation")
        self._validate_conditional_fields()

    @property
    def end_offset_ms(self) -> int:
        return self.offset_ms + (self.duration_ms or 0)

    @property
    def relative_speed_units_per_second(self) -> float | None:
        """Derive relative movement speed; it is never stored in the plan."""

        if (
            self.event_type not in _RELATIVE_MOVE_EVENT_TYPES
            or self.delta is None
            or self.duration_ms is None
        ):
            return None
        distance = math.hypot(self.delta[0], self.delta[1])
        return distance * 1_000.0 / self.duration_ms

    def derived_absolute_speed_per_second(
        self,
        start_position: NormalizedPoint,
    ) -> float:
        """Derive normalized client units per second for an absolute move."""

        if (
            self.event_type is not InputPlanEventType.MOUSE_MOVE_ABSOLUTE
            or self.position is None
            or self.duration_ms is None
        ):
            raise ValueError("absolute speed is only available for absolute moves")
        start = _normalized_point(start_position, "start_position")
        distance = math.hypot(
            self.position[0] - start[0],
            self.position[1] - start[1],
        )
        return distance * 1_000.0 / self.duration_ms

    def _validate_conditional_fields(self) -> None:
        event_type = self.event_type
        if event_type is InputPlanEventType.WAIT:
            self._require_fields("duration_ms")
            self._forbid_action_fields(except_fields={"duration_ms"})
            return
        if event_type in {
            InputPlanEventType.KEY_DOWN,
            InputPlanEventType.KEY_UP,
        }:
            self._require_fields("key")
            self._forbid_action_fields(
                except_fields={
                    "key",
                    "virtual_key",
                    "scan_code",
                    "is_extended",
                }
            )
            return
        if event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN,
            InputPlanEventType.MOUSE_BUTTON_UP,
        }:
            self._require_fields("button", "position")
            self._forbid_action_fields(except_fields={"button", "position"})
            return
        if event_type in _DIRECT_MOUSE_BUTTON_EVENT_TYPES:
            self._require_fields("button")
            self._forbid_action_fields(except_fields={"button"})
            return
        if event_type is InputPlanEventType.MOUSE_MOVE_ABSOLUTE:
            self._require_fields(
                "position",
                "duration_ms",
                "update_rate_hz",
                "interpolation",
            )
            self._forbid_action_fields(
                except_fields={
                    "position",
                    "duration_ms",
                    "update_rate_hz",
                    "interpolation",
                }
            )
            return
        if event_type in _RELATIVE_MOVE_EVENT_TYPES:
            self._require_fields(
                "delta",
                "duration_ms",
                "update_rate_hz",
                "interpolation",
            )
            if self.delta == (0, 0):
                raise ValueError("relative mouse delta cannot be zero")
            self._forbid_action_fields(
                except_fields={
                    "delta",
                    "duration_ms",
                    "update_rate_hz",
                    "interpolation",
                }
            )
            return
        if event_type is InputPlanEventType.MOUSE_WHEEL:
            self._require_fields("position", "wheel_delta")
            if self.wheel_delta == (0, 0):
                raise ValueError("wheel_delta cannot be zero")
            self._forbid_action_fields(except_fields={"position", "wheel_delta"})
            return
        raise AssertionError(f"unhandled input plan event type: {event_type}")

    def _require_fields(self, *names: str) -> None:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"{self.event_type.value} requires fields: {', '.join(missing)}"
            )

    def _forbid_action_fields(self, *, except_fields: set[str]) -> None:
        nullable_fields = {
            "key",
            "virtual_key",
            "scan_code",
            "button",
            "position",
            "delta",
            "wheel_delta",
            "duration_ms",
            "update_rate_hz",
            "interpolation",
        }
        populated = [
            name
            for name in sorted(nullable_fields - except_fields)
            if getattr(self, name) is not None
        ]
        if "is_extended" not in except_fields and self.is_extended:
            populated.append("is_extended")
        if populated:
            raise ValueError(
                f"{self.event_type.value} forbids fields: {', '.join(populated)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "offset_ms": self.offset_ms,
            "event_type": self.event_type.value,
            "enabled": self.enabled,
            "source_event_ids": list(self.source_event_ids),
            "key": self.key,
            "virtual_key": self.virtual_key,
            "scan_code": self.scan_code,
            "is_extended": self.is_extended,
            "button": self.button.value if self.button is not None else None,
            "position": list(self.position) if self.position is not None else None,
            "delta": list(self.delta) if self.delta is not None else None,
            "wheel_delta": (
                list(self.wheel_delta) if self.wheel_delta is not None else None
            ),
            "duration_ms": self.duration_ms,
            "update_rate_hz": self.update_rate_hz,
            "interpolation": (
                self.interpolation.value if self.interpolation is not None else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        source_schema_version: int | None = None,
    ) -> InputPlanEvent:
        if source_schema_version is None:
            version = (
                INPUT_PLAN_SCHEMA_VERSION
                if type(value) is dict and "track_id" in value
                else DIRECT_BUTTON_INPUT_PLAN_SCHEMA_VERSION
            )
        else:
            version = _integer(
                source_schema_version,
                "source_schema_version",
                minimum=LEGACY_INPUT_PLAN_SCHEMA_VERSION,
                maximum=INPUT_PLAN_SCHEMA_VERSION,
            )
        mapping = _strict_mapping(
            value,
            context="input plan event",
            expected_keys=(
                _LEGACY_EVENT_FIELDS
                if version < INPUT_PLAN_SCHEMA_VERSION
                else _EVENT_FIELDS
            ),
        )
        return cls(
            event_id=mapping["event_id"],
            track_id=(
                DEFAULT_INPUT_TRACK_ID
                if version < INPUT_PLAN_SCHEMA_VERSION
                else mapping["track_id"]
            ),
            offset_ms=mapping["offset_ms"],
            event_type=_enum_from_json(
                InputPlanEventType,
                mapping["event_type"],
                "event_type",
            ),
            enabled=_strict_bool(mapping["enabled"], "enabled"),
            source_event_ids=_string_tuple_from_json(
                mapping["source_event_ids"],
                "source_event_ids",
            ),
            key=_optional_text_from_json(mapping["key"], "key"),
            virtual_key=mapping["virtual_key"],
            scan_code=mapping["scan_code"],
            is_extended=_strict_bool(mapping["is_extended"], "is_extended"),
            button=_optional_enum_from_json(
                MouseButton,
                mapping["button"],
                "button",
            ),
            position=_optional_point_from_json(mapping["position"], "position"),
            delta=_optional_delta_from_json(mapping["delta"], "delta"),
            wheel_delta=_optional_delta_from_json(
                mapping["wheel_delta"],
                "wheel_delta",
            ),
            duration_ms=mapping["duration_ms"],
            update_rate_hz=mapping["update_rate_hz"],
            interpolation=_optional_enum_from_json(
                MouseInterpolation,
                mapping["interpolation"],
                "interpolation",
            ),
        )


_TRACK_FIELDS = frozenset(
    {
        "track_id",
        "name",
        "enabled",
        "locked",
        "start_offset_ms",
    }
)


@dataclass(frozen=True, slots=True)
class InputTrack:
    """Persisted editing lane sharing the plan's single monotonic clock."""

    track_id: str
    name: str
    enabled: bool = True
    locked: bool = False
    start_offset_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(self, "name", _text(self.name, "track name"))
        object.__setattr__(self, "enabled", _strict_bool(self.enabled, "enabled"))
        object.__setattr__(self, "locked", _strict_bool(self.locked, "locked"))
        object.__setattr__(
            self,
            "start_offset_ms",
            _integer(
                self.start_offset_ms,
                "start_offset_ms",
                minimum=0,
                maximum=HARD_MAX_TIME_MS,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "enabled": self.enabled,
            "locked": self.locked,
            "start_offset_ms": self.start_offset_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> InputTrack:
        mapping = _strict_mapping(
            value,
            context="input track",
            expected_keys=_TRACK_FIELDS,
        )
        return cls(
            track_id=mapping["track_id"],
            name=mapping["name"],
            enabled=_strict_bool(mapping["enabled"], "enabled"),
            locked=_strict_bool(mapping["locked"], "locked"),
            start_offset_ms=mapping["start_offset_ms"],
        )


DEFAULT_INPUT_TRACK: Final = InputTrack(
    track_id=DEFAULT_INPUT_TRACK_ID,
    name=DEFAULT_INPUT_TRACK_NAME,
)
DEFAULT_INPUT_TRACKS: Final = (DEFAULT_INPUT_TRACK,)


_LEGACY_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "plan_id",
        "name",
        "revision",
        "source",
        "created_at_utc",
        "updated_at_utc",
        "events",
        "safety_limits",
    }
)
_PLAN_FIELDS = _LEGACY_PLAN_FIELDS | {"tracks"}


@dataclass(frozen=True, slots=True)
class InputPlan:
    """Experiment-private editable plan; semantic validity is checked separately."""

    schema_version: int
    plan_id: str
    name: str
    revision: int
    source: InputPlanSource
    created_at_utc: str
    updated_at_utc: str
    events: tuple[InputPlanEvent, ...]
    safety_limits: InputPlanSafetyLimits
    tracks: tuple[InputTrack, ...] = DEFAULT_INPUT_TRACKS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _integer(
                self.schema_version,
                "schema_version",
                minimum=INPUT_PLAN_SCHEMA_VERSION,
                maximum=INPUT_PLAN_SCHEMA_VERSION,
            ),
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(
            self,
            "revision",
            _integer(
                self.revision,
                "revision",
                minimum=1,
                maximum=2**31 - 1,
            ),
        )
        if not isinstance(self.source, InputPlanSource):
            raise TypeError("source must be an InputPlanSource")
        created = _utc_timestamp(self.created_at_utc, "created_at_utc")
        updated = _utc_timestamp(self.updated_at_utc, "updated_at_utc")
        if _parse_utc(updated) < _parse_utc(created):
            raise ValueError("updated_at_utc cannot precede created_at_utc")
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "updated_at_utc", updated)
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if len(self.events) > HARD_MAX_EVENT_COUNT:
            raise ValueError(
                f"events cannot exceed the hard limit of {HARD_MAX_EVENT_COUNT}"
            )
        if any(type(event) is not InputPlanEvent for event in self.events):
            raise TypeError("events must contain exact InputPlanEvent values")
        if type(self.tracks) is not tuple:
            raise TypeError("tracks must be a tuple")
        if len(self.tracks) > HARD_MAX_TRACK_COUNT:
            raise ValueError(
                f"tracks cannot exceed the hard limit of {HARD_MAX_TRACK_COUNT}"
            )
        if any(type(track) is not InputTrack for track in self.tracks):
            raise TypeError("tracks must contain exact InputTrack values")
        if type(self.safety_limits) is not InputPlanSafetyLimits:
            raise TypeError(
                "safety_limits must be an exact InputPlanSafetyLimits value"
            )

    @property
    def duration_ms(self) -> int:
        return plan_duration_ms(self)

    def revised(
        self,
        *,
        events: tuple[InputPlanEvent, ...] | None = None,
        tracks: tuple[InputTrack, ...] | None = None,
        name: str | None = None,
        source: InputPlanSource | None = None,
        updated_at_utc: str | None = None,
    ) -> InputPlan:
        if self.revision >= 2**31 - 1:
            raise ValueError("plan revision is exhausted")
        return InputPlan(
            schema_version=self.schema_version,
            plan_id=self.plan_id,
            name=self.name if name is None else name,
            revision=self.revision + 1,
            source=self.source if source is None else source,
            created_at_utc=self.created_at_utc,
            updated_at_utc=updated_at_utc or utc_now_iso(),
            events=self.events if events is None else events,
            safety_limits=self.safety_limits,
            tracks=self.tracks if tracks is None else tracks,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "name": self.name,
            "revision": self.revision,
            "source": self.source.value,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "events": [event.to_dict() for event in self.events],
            "tracks": [track.to_dict() for track in self.tracks],
            "safety_limits": self.safety_limits.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> InputPlan:
        if type(value) is not dict:
            raise TypeError("input plan must be a JSON object")
        if "schema_version" not in value:
            raise ValueError("input plan is missing keys: schema_version")
        source_schema_version = _integer(
            value["schema_version"],
            "schema_version",
            minimum=LEGACY_INPUT_PLAN_SCHEMA_VERSION,
            maximum=INPUT_PLAN_SCHEMA_VERSION,
        )
        mapping = _strict_mapping(
            value,
            context="input plan",
            expected_keys=(
                _LEGACY_PLAN_FIELDS
                if source_schema_version < INPUT_PLAN_SCHEMA_VERSION
                else _PLAN_FIELDS
            ),
        )
        events = tuple(
            InputPlanEvent.from_dict(
                item,
                source_schema_version=source_schema_version,
            )
            for item in _json_array(mapping["events"], "events")
        )
        if source_schema_version < CAMERA_INPUT_PLAN_SCHEMA_VERSION and any(
            event.event_type is InputPlanEventType.CAMERA_MOVE_RELATIVE
            for event in events
        ):
            raise ValueError("schema_version 1 does not support CAMERA_MOVE_RELATIVE")
        if source_schema_version < DIRECT_BUTTON_INPUT_PLAN_SCHEMA_VERSION and any(
            event.event_type in _DIRECT_MOUSE_BUTTON_EVENT_TYPES for event in events
        ):
            raise ValueError(
                f"schema_version {source_schema_version} does not support "
                "direct mouse button events"
            )
        return cls(
            schema_version=INPUT_PLAN_SCHEMA_VERSION,
            plan_id=mapping["plan_id"],
            name=mapping["name"],
            revision=mapping["revision"],
            source=_enum_from_json(
                InputPlanSource,
                mapping["source"],
                "source",
            ),
            created_at_utc=mapping["created_at_utc"],
            updated_at_utc=mapping["updated_at_utc"],
            events=events,
            safety_limits=InputPlanSafetyLimits.from_dict(mapping["safety_limits"]),
            tracks=(
                DEFAULT_INPUT_TRACKS
                if source_schema_version < INPUT_PLAN_SCHEMA_VERSION
                else tuple(
                    InputTrack.from_dict(item)
                    for item in _json_array(mapping["tracks"], "tracks")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanValidationIssue:
    code: str
    message: str
    event_id: str | None = None


class PlanValidationError(ValueError):
    def __init__(self, issues: tuple[PlanValidationIssue, ...]) -> None:
        if not issues:
            raise ValueError("PlanValidationError requires at least one issue")
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def plan_duration_ms(plan: InputPlan) -> int:
    if type(plan) is not InputPlan:
        raise TypeError("plan must be an exact InputPlan")
    tracks = {track.track_id: track for track in plan.tracks}
    return max(
        (
            track.start_offset_ms + event.end_offset_ms
            for event in plan.events
            if event.enabled
            and (track := tracks.get(event.track_id)) is not None
            and track.enabled
        ),
        default=0,
    )


def validate_plan(plan: InputPlan) -> None:
    """Validate ordering, resource limits, and input press/release integrity."""

    if type(plan) is not InputPlan:
        raise TypeError("plan must be an exact InputPlan")
    limits = plan.safety_limits
    issues: list[PlanValidationIssue] = []
    tracks: dict[str, InputTrack] = {}
    for track in plan.tracks:
        if track.track_id in tracks:
            issues.append(
                PlanValidationIssue(
                    "DUPLICATE_TRACK_ID",
                    f"duplicate track_id: {track.track_id}",
                )
            )
            continue
        tracks[track.track_id] = track
    if not plan.tracks:
        issues.append(
            PlanValidationIssue(
                "NO_TRACKS",
                "plan must contain at least one input track",
            )
        )
    if len(plan.events) > limits.max_event_count:
        issues.append(
            PlanValidationIssue(
                "EVENT_COUNT_LIMIT",
                "plan event count exceeds safety_limits.max_event_count",
            )
        )
    if not plan.events:
        issues.append(
            PlanValidationIssue(
                "EMPTY_PLAN",
                "plan must contain at least one event",
            )
        )
    previous_offset = -1
    event_ids: set[str] = set()
    for event in plan.events:
        if event.event_id in event_ids:
            issues.append(
                PlanValidationIssue(
                    "DUPLICATE_EVENT_ID",
                    f"duplicate event_id: {event.event_id}",
                    event.event_id,
                )
            )
        event_ids.add(event.event_id)
        if event.offset_ms < previous_offset:
            issues.append(
                PlanValidationIssue(
                    "NON_MONOTONIC_OFFSET",
                    "event offsets must be non-decreasing",
                    event.event_id,
                )
            )
        previous_offset = event.offset_ms
        if event.track_id not in tracks:
            issues.append(
                PlanValidationIssue(
                    "UNKNOWN_TRACK_ID",
                    f"{event.event_id} references unknown track {event.track_id}",
                    event.event_id,
                )
            )

    effective_events = _effective_enabled_events(plan, tracks, issues)
    if plan.events and not effective_events:
        issues.append(
            PlanValidationIssue(
                "NO_ENABLED_EVENTS",
                "plan must contain at least one event on an enabled track",
            )
        )

    open_keys: dict[tuple[str, int | None, bool], InputPlanEvent] = {}
    open_buttons: dict[MouseButton, InputPlanEvent] = {}
    reserved_abort_reported = False
    for event in effective_events:
        _validate_mouse_limits(event, limits, issues)
        if event.event_type in {
            InputPlanEventType.KEY_DOWN,
            InputPlanEventType.KEY_UP,
        }:
            if event.virtual_key is None and event.scan_code is None:
                issues.append(
                    PlanValidationIssue(
                        "KEY_CODE_REQUIRED",
                        "keyboard events require virtual_key or scan_code",
                        event.event_id,
                    )
                )
            identity = _key_delivery_identity(event)
            _validate_press_pair(
                event,
                identity,
                open_keys,
                limits,
                issues,
                label=f"key {event.key}",
            )
            if (
                not reserved_abort_reported
                and event.event_type is InputPlanEventType.KEY_DOWN
                and _contains_reserved_abort_hotkey(tuple(open_keys.values()))
            ):
                issues.append(
                    PlanValidationIssue(
                        "RESERVED_ABORT_HOTKEY",
                        (
                            "Ctrl+Shift+F12 is reserved for the global "
                            "emergency stop and cannot appear in a plan"
                        ),
                        event.event_id,
                    )
                )
                reserved_abort_reported = True
        elif event.event_type in _MOUSE_BUTTON_EVENT_TYPES:
            assert event.button is not None
            _validate_press_pair(
                event,
                event.button,
                open_buttons,
                limits,
                issues,
                label=f"mouse button {event.button.value}",
            )

    for event in (*open_keys.values(), *open_buttons.values()):
        issues.append(
            PlanValidationIssue(
                "UNRELEASED_INPUT",
                f"{event.event_id} has no matching release",
                event.event_id,
            )
        )
    _validate_expanded_schedule_limit(effective_events, issues)
    _validate_blocking_segments(effective_events, issues)
    duration = plan_duration_ms(plan)
    if duration > limits.max_total_duration_ms:
        issues.append(
            PlanValidationIssue(
                "TOTAL_DURATION_LIMIT",
                "plan duration exceeds safety_limits.max_total_duration_ms",
            )
        )
    if issues:
        raise PlanValidationError(tuple(issues))


def _effective_enabled_events(
    plan: InputPlan,
    tracks: dict[str, InputTrack],
    issues: list[PlanValidationIssue],
) -> tuple[InputPlanEvent, ...]:
    scheduled: list[tuple[int, int, InputPlanEvent]] = []
    for original_index, event in enumerate(plan.events):
        track = tracks.get(event.track_id)
        if track is None or not track.enabled or not event.enabled:
            continue
        effective_offset_ms = track.start_offset_ms + event.offset_ms
        effective_end_offset_ms = track.start_offset_ms + event.end_offset_ms
        if (
            effective_offset_ms > HARD_MAX_TIME_MS
            or effective_end_offset_ms > HARD_MAX_TIME_MS
        ):
            issues.append(
                PlanValidationIssue(
                    "EFFECTIVE_TIME_LIMIT",
                    (
                        f"{event.event_id} exceeds the hard effective time "
                        f"limit after applying track {track.track_id}"
                    ),
                    event.event_id,
                )
            )
            continue
        scheduled.append(
            (
                effective_offset_ms,
                original_index,
                replace(event, offset_ms=effective_offset_ms),
            )
        )
    scheduled.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in scheduled)


def _contains_reserved_abort_hotkey(
    held_keys: tuple[InputPlanEvent, ...],
) -> bool:
    roles: set[str] = set()
    for event in held_keys:
        key_name = (event.key or "").casefold()
        virtual_key = event.virtual_key
        if key_name in {"ctrl", "control", "ctrl_l", "ctrl_r"} or virtual_key in {
            0x11,
            0xA2,
            0xA3,
        }:
            roles.add("ctrl")
        if key_name in {"shift", "shift_l", "shift_r"} or virtual_key in {
            0x10,
            0xA0,
            0xA1,
        }:
            roles.add("shift")
        if key_name == "f12" or virtual_key == 0x7B:
            roles.add("f12")
    return roles == {"ctrl", "shift", "f12"}


def _key_delivery_identity(
    event: InputPlanEvent,
) -> tuple[str, int | None, bool]:
    if event.scan_code is not None:
        return "scan", event.scan_code, event.is_extended
    return "virtual", event.virtual_key, event.is_extended


def _validate_blocking_segments(
    enabled: tuple[InputPlanEvent, ...],
    issues: list[PlanValidationIssue],
) -> None:
    segments = tuple(
        event
        for event in enabled
        if event.event_type in {InputPlanEventType.WAIT, *_MOUSE_MOVE_EVENT_TYPES}
    )
    for segment in segments:
        for event in enabled:
            if event is segment:
                continue
            if not (segment.offset_ms <= event.offset_ms < segment.end_offset_ms):
                continue
            if segment.track_id == event.track_id:
                issues.append(
                    PlanValidationIssue(
                        "BLOCKING_SEGMENT_OVERLAP",
                        (
                            f"{event.event_id} starts inside same-track blocking "
                            f"segment {segment.event_id}"
                        ),
                        event.event_id,
                    )
                )
                continue
            if _cross_track_segment_allows(segment, event):
                continue
            issues.append(
                PlanValidationIssue(
                    "CROSS_TRACK_RESOURCE_CONFLICT",
                    (
                        f"{event.event_id} uses a mouse resource incompatible "
                        f"with cross-track segment {segment.event_id}"
                    ),
                    event.event_id,
                )
            )


def _cross_track_segment_allows(
    segment: InputPlanEvent,
    event: InputPlanEvent,
) -> bool:
    if segment.event_type is InputPlanEventType.WAIT:
        return True
    keyboard_types = {
        InputPlanEventType.KEY_DOWN,
        InputPlanEventType.KEY_UP,
    }
    if segment.event_type is InputPlanEventType.CAMERA_MOVE_RELATIVE:
        return event.event_type in {
            *keyboard_types,
            *_DIRECT_MOUSE_BUTTON_EVENT_TYPES,
            InputPlanEventType.WAIT,
        }
    if segment.event_type in {
        InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
        InputPlanEventType.MOUSE_MOVE_RELATIVE,
    }:
        return event.event_type in {
            *keyboard_types,
            InputPlanEventType.WAIT,
        }
    raise AssertionError(f"unexpected blocking segment: {segment.event_type}")


def _validate_expanded_schedule_limit(
    enabled: tuple[InputPlanEvent, ...],
    issues: list[PlanValidationIssue],
) -> None:
    expanded_slots = 0
    for event in enabled:
        if event.event_type in _MOUSE_MOVE_EVENT_TYPES:
            assert event.duration_ms is not None
            assert event.update_rate_hz is not None
            expanded_slots += (event.duration_ms * event.update_rate_hz + 999) // 1_000
        else:
            expanded_slots += 1
    if expanded_slots > FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS:
        issues.append(
            PlanValidationIssue(
                "EXPANDED_SCHEDULE_LIMIT",
                (
                    f"expanded schedule requires {expanded_slots} slots and "
                    f"exceeds the hard limit of "
                    f"{FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS}"
                ),
            )
        )


def _validate_press_pair(
    event: InputPlanEvent,
    identity: object,
    open_inputs: dict[object, InputPlanEvent],
    limits: InputPlanSafetyLimits,
    issues: list[PlanValidationIssue],
    *,
    label: str,
) -> None:
    is_down = event.event_type in {
        InputPlanEventType.KEY_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
    }
    if is_down:
        if identity in open_inputs:
            down = open_inputs[identity]
            if down.track_id != event.track_id:
                code = "CROSS_TRACK_INPUT_CONFLICT"
                message = (
                    f"overlapping {label} is owned by tracks "
                    f"{down.track_id} and {event.track_id}"
                )
            else:
                code = "DUPLICATE_DOWN"
                message = f"duplicate down for {label}"
            issues.append(
                PlanValidationIssue(
                    code,
                    message,
                    event.event_id,
                )
            )
            return
        open_inputs[identity] = event
        return
    down = open_inputs.pop(identity, None)
    if down is None:
        issues.append(
            PlanValidationIssue(
                "ISOLATED_UP",
                f"release without matching down for {label}",
                event.event_id,
            )
        )
        return
    if down.track_id != event.track_id:
        issues.append(
            PlanValidationIssue(
                "CROSS_TRACK_INPUT_PAIR",
                (f"matching {label} down/up events must belong to the same track"),
                event.event_id,
            )
        )
    if event.event_type in _MOUSE_BUTTON_UP_TO_DOWN:
        expected_down_type = _MOUSE_BUTTON_UP_TO_DOWN[event.event_type]
        if down.event_type is not expected_down_type:
            issues.append(
                PlanValidationIssue(
                    "MOUSE_BUTTON_DELIVERY_MISMATCH",
                    (
                        "matching mouse button down/up events must use the "
                        "same delivery mode"
                    ),
                    event.event_id,
                )
            )
    if (
        event.event_type is InputPlanEventType.KEY_UP
        and (down.key or "").casefold() != (event.key or "").casefold()
    ):
        issues.append(
            PlanValidationIssue(
                "KEY_LABEL_MISMATCH",
                "matching keyboard down/up events must use the same key label",
                event.event_id,
            )
        )
    hold_duration = event.offset_ms - down.offset_ms
    if hold_duration > limits.max_hold_duration_ms:
        issues.append(
            PlanValidationIssue(
                "HOLD_DURATION_LIMIT",
                f"hold duration for {label} exceeds safety limit",
                event.event_id,
            )
        )


def _validate_mouse_limits(
    event: InputPlanEvent,
    limits: InputPlanSafetyLimits,
    issues: list[PlanValidationIssue],
) -> None:
    if event.position is not None:
        minimum = limits.allowed_normalized_min
        maximum = limits.allowed_normalized_max
        if not (
            minimum[0] <= event.position[0] <= maximum[0]
            and minimum[1] <= event.position[1] <= maximum[1]
        ):
            issues.append(
                PlanValidationIssue(
                    "MOUSE_POSITION_LIMIT",
                    "mouse position lies outside the plan safety rectangle",
                    event.event_id,
                )
            )
    if event.delta is not None and any(
        abs(value) > limits.max_relative_delta_per_axis for value in event.delta
    ):
        issues.append(
            PlanValidationIssue(
                "MOUSE_DELTA_LIMIT",
                "relative mouse delta exceeds the per-axis safety limit",
                event.event_id,
            )
        )
    if event.wheel_delta is not None and any(
        abs(value) > limits.max_wheel_delta_per_axis for value in event.wheel_delta
    ):
        issues.append(
            PlanValidationIssue(
                "WHEEL_DELTA_LIMIT",
                "mouse wheel delta exceeds the per-axis safety limit",
                event.event_id,
            )
        )
    if event.event_type in _MOUSE_MOVE_EVENT_TYPES:
        assert event.duration_ms is not None
        assert event.update_rate_hz is not None
        if event.duration_ms > limits.max_mouse_move_duration_ms:
            issues.append(
                PlanValidationIssue(
                    "MOUSE_MOVE_DURATION_LIMIT",
                    "mouse move duration exceeds the safety limit",
                    event.event_id,
                )
            )
        if event.update_rate_hz > limits.max_mouse_update_rate_hz:
            issues.append(
                PlanValidationIssue(
                    "MOUSE_UPDATE_RATE_LIMIT",
                    "mouse move update rate exceeds the safety limit",
                    event.event_id,
                )
            )


def _enum_from_json(
    enum_type: type[Enum],
    value: object,
    name: str,
) -> Enum:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{name} has an unknown value: {value}") from exc


def _optional_enum_from_json(
    enum_type: type[Enum],
    value: object,
    name: str,
) -> Enum | None:
    if value is None:
        return None
    return _enum_from_json(enum_type, value, name)


def _optional_text_from_json(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _string_tuple_from_json(value: object, name: str) -> tuple[str, ...]:
    return tuple(_json_array(value, name))


def _point_from_json(value: object, name: str) -> NormalizedPoint:
    return _normalized_point(tuple(_json_array(value, name)), name)


def _optional_point_from_json(
    value: object,
    name: str,
) -> NormalizedPoint | None:
    if value is None:
        return None
    return _point_from_json(value, name)


def _optional_delta_from_json(
    value: object,
    name: str,
) -> IntegerDelta | None:
    if value is None:
        return None
    return _integer_delta(tuple(_json_array(value, name)), name)


__all__ = [
    "DEFAULT_INPUT_TRACK",
    "DEFAULT_INPUT_TRACK_ID",
    "DEFAULT_INPUT_TRACK_NAME",
    "DEFAULT_INPUT_TRACKS",
    "DIRECT_BUTTON_INPUT_PLAN_SCHEMA_VERSION",
    "FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS",
    "HARD_MAX_EVENT_COUNT",
    "HARD_MAX_TRACK_COUNT",
    "INPUT_PLAN_SCHEMA_VERSION",
    "LEGACY_INPUT_PLAN_SCHEMA_VERSION",
    "InputPlan",
    "InputPlanEvent",
    "InputPlanEventType",
    "InputPlanSafetyLimits",
    "InputPlanSource",
    "InputTrack",
    "IntegerDelta",
    "MouseButton",
    "MouseInterpolation",
    "NormalizedPoint",
    "PlanValidationError",
    "PlanValidationIssue",
    "plan_duration_ms",
    "utc_now_iso",
    "validate_plan",
]
