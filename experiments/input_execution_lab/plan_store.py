from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Final

from .contracts import InputPlan, validate_plan


DEFAULT_MAX_PLAN_FILE_BYTES: Final = 1_048_576
HARD_MAX_PLAN_FILE_BYTES: Final = 4_194_304
_SAFE_PLAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class InputPlanStoreError(RuntimeError):
    pass


class InputPlanNotFoundError(InputPlanStoreError):
    pass


class InputPlanFileTooLargeError(InputPlanStoreError):
    pass


class InputPlanJsonError(InputPlanStoreError):
    pass


class DuplicateJsonKeyError(InputPlanJsonError):
    pass


def serialize_plan(plan: InputPlan) -> bytes:
    """Serialize one plan to canonical, finite, UTF-8 JSON."""

    if type(plan) is not InputPlan:
        raise TypeError("plan must be an exact InputPlan")
    validate_plan(plan)
    try:
        text = json.dumps(
            plan.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InputPlanJsonError(
            f"plan cannot be encoded as strict JSON: {exc}"
        ) from exc
    return f"{text}\n".encode("utf-8")


def deserialize_plan(
    payload: bytes,
    *,
    max_file_bytes: int = DEFAULT_MAX_PLAN_FILE_BYTES,
) -> InputPlan:
    """Decode strict JSON and validate the resulting plan before returning it."""

    limit = _validated_max_file_bytes(max_file_bytes)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > limit:
        raise InputPlanFileTooLargeError(f"plan payload exceeds {limit} bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputPlanJsonError("plan file must be valid UTF-8") from exc
    if text.startswith("\ufeff"):
        raise InputPlanJsonError("UTF-8 BOM is not permitted")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except DuplicateJsonKeyError:
        raise
    except (json.JSONDecodeError, InputPlanJsonError, RecursionError) as exc:
        if isinstance(exc, InputPlanJsonError):
            raise
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise InputPlanJsonError(f"invalid plan JSON: {detail}") from exc
    try:
        plan = InputPlan.from_dict(decoded)
        validate_plan(plan)
    except (TypeError, ValueError) as exc:
        raise InputPlanJsonError(f"invalid input plan schema: {exc}") from exc
    return plan


class InputPlanStore:
    """Thread-safe, caller-rooted, atomic JSON store for experiment plans."""

    def __init__(
        self,
        root_directory: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_PLAN_FILE_BYTES,
    ) -> None:
        if isinstance(root_directory, bytes):
            raise TypeError("root_directory must be text or a path-like value")
        try:
            root = Path(root_directory).expanduser().resolve(strict=False)
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError("root_directory is invalid") from exc
        self._root_directory = root
        self._max_file_bytes = _validated_max_file_bytes(max_file_bytes)
        self._lock = threading.RLock()

    @property
    def root_directory(self) -> Path:
        return self._root_directory

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    def path_for(self, plan_id: str) -> Path:
        safe_id = _validated_plan_id(plan_id)
        return self._root_directory / f"{safe_id}.json"

    def save(self, plan: InputPlan) -> Path:
        """Validate and atomically replace the plan file in the store directory."""

        if type(plan) is not InputPlan:
            raise TypeError("plan must be an exact InputPlan")
        payload = serialize_plan(plan)
        if len(payload) > self._max_file_bytes:
            raise InputPlanFileTooLargeError(
                f"serialized plan exceeds {self._max_file_bytes} bytes"
            )
        target = self.path_for(plan.plan_id)
        with self._lock:
            self._prepare_directory()
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    dir=self._root_directory,
                    prefix=f".{plan.plan_id}.",
                    suffix=".tmp",
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, target)
                temporary_path = None
            except OSError as exc:
                raise InputPlanStoreError(
                    f"cannot atomically save input plan: {exc}"
                ) from exc
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass
        return target

    def load(self, plan_id: str) -> InputPlan:
        path = self.path_for(plan_id)
        with self._lock:
            return self._load_path(path)

    def list_plan_ids(self) -> tuple[str, ...]:
        with self._lock:
            if not self._root_directory.exists():
                return ()
            if not self._root_directory.is_dir():
                raise InputPlanStoreError("plan store root is not a directory")
            plan_ids: list[str] = []
            for path in self._root_directory.glob("*.json"):
                if path.is_symlink() or not path.is_file():
                    continue
                if _SAFE_PLAN_ID.fullmatch(path.stem) is not None:
                    plan_ids.append(path.stem)
            return tuple(sorted(plan_ids, key=str.casefold))

    def list_plans(self) -> tuple[InputPlan, ...]:
        with self._lock:
            return tuple(
                self._load_path(self.path_for(plan_id))
                for plan_id in self.list_plan_ids()
            )

    def delete(self, plan_id: str) -> bool:
        """Delete one exact regular plan file; callers own user confirmation."""

        path = self.path_for(plan_id)
        with self._lock:
            if path.is_symlink():
                raise InputPlanStoreError(
                    "only regular non-symbolic-link plan files can be deleted"
                )
            if not path.exists():
                return False
            if not path.is_file():
                raise InputPlanStoreError(
                    "only regular non-symbolic-link plan files can be deleted"
                )
            try:
                path.unlink()
            except OSError as exc:
                raise InputPlanStoreError(f"cannot delete input plan: {exc}") from exc
            return True

    def _load_path(self, path: Path) -> InputPlan:
        if not path.exists():
            raise InputPlanNotFoundError(f"input plan does not exist: {path.stem}")
        if path.is_symlink():
            raise InputPlanStoreError("symbolic-link plan files are not permitted")
        if not path.is_file():
            raise InputPlanStoreError("input plan path is not a regular file")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise InputPlanStoreError(f"cannot inspect input plan: {exc}") from exc
        if size > self._max_file_bytes:
            raise InputPlanFileTooLargeError(
                f"plan file exceeds {self._max_file_bytes} bytes"
            )
        try:
            with path.open("rb") as stream:
                payload = stream.read(self._max_file_bytes + 1)
        except OSError as exc:
            raise InputPlanStoreError(f"cannot read input plan: {exc}") from exc
        if len(payload) > self._max_file_bytes:
            raise InputPlanFileTooLargeError(
                f"plan file exceeds {self._max_file_bytes} bytes"
            )
        return deserialize_plan(payload, max_file_bytes=self._max_file_bytes)

    def _prepare_directory(self) -> None:
        try:
            self._root_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InputPlanStoreError(
                f"cannot create plan store directory: {exc}"
            ) from exc
        if not self._root_directory.is_dir():
            raise InputPlanStoreError("plan store root is not a directory")


PlanStore = InputPlanStore


def _validated_plan_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_PLAN_ID.fullmatch(value) is None:
        raise ValueError("plan_id is not safe for use as a file name")
    return value


def _validated_max_file_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_file_bytes must be an integer")
    if value <= 0 or value > HARD_MAX_PLAN_FILE_BYTES:
        raise ValueError(
            f"max_file_bytes must be between 1 and {HARD_MAX_PLAN_FILE_BYTES}"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_non_finite_constant(value: str) -> object:
    raise InputPlanJsonError(f"non-finite JSON number is not permitted: {value}")


__all__ = [
    "DEFAULT_MAX_PLAN_FILE_BYTES",
    "DuplicateJsonKeyError",
    "HARD_MAX_PLAN_FILE_BYTES",
    "InputPlanFileTooLargeError",
    "InputPlanJsonError",
    "InputPlanNotFoundError",
    "InputPlanStore",
    "InputPlanStoreError",
    "PlanStore",
    "deserialize_plan",
    "serialize_plan",
]
