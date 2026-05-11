"""JSON file persistence for skills and lineage data."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.skill import Skill
from .config import get_config


def _data_dir() -> Path:
    path = Path(get_config().data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_skill(skill: Skill) -> Path:
    """Persist a skill to disk."""
    filepath = _data_dir() / f"{skill.id}.json"
    filepath.write_text(json.dumps(skill.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath


def load_skill(skill_id: str) -> Skill | None:
    """Load a skill by ID."""
    filepath = _data_dir() / f"{skill_id}.json"
    if not filepath.exists():
        return None
    return Skill.from_dict(json.loads(filepath.read_text(encoding="utf-8")))


def list_skills() -> list[Skill]:
    """Load all persisted skills."""
    skills = []
    for f in sorted(_data_dir().glob("*.json")):
        if f.name == "lineage.json" or f.name == "state.json":
            continue
        try:
            skills.append(Skill.from_dict(json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return skills


def delete_skill(skill_id: str) -> bool:
    """Delete a skill file."""
    filepath = _data_dir() / f"{skill_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def save_lineage(lineage_data: dict) -> Path:
    """Save the lineage tree data."""
    filepath = _data_dir() / "lineage.json"
    filepath.write_text(json.dumps(lineage_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath


def load_lineage() -> dict:
    """Load the lineage tree data."""
    filepath = _data_dir() / "lineage.json"
    if not filepath.exists():
        return {"nodes": {}, "edges": []}
    return json.loads(filepath.read_text(encoding="utf-8"))


def save_state(state: dict) -> Path:
    """Save simulator state."""
    filepath = _data_dir() / "state.json"
    filepath.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath


def load_state() -> dict:
    """Load simulator state."""
    filepath = _data_dir() / "state.json"
    if not filepath.exists():
        return {}
    return json.loads(filepath.read_text(encoding="utf-8"))
