"""User-level configuration stored in ~/.evoskill/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

_CONFIG_DIR = Path.home() / ".evoskill"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"

_VALID_KEYS = {"provider", "model", "anthropic_api_key", "openai_api_key", "deepseek_api_key", "data_dir"}


@dataclass
class Config:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    data_dir: str = ""

    def __post_init__(self) -> None:
        if not self.data_dir:
            self.data_dir = str(_CONFIG_DIR)

    def with_env(self) -> Config:
        """Return copy with environment variable overrides (for CI/CD compat)."""
        return Config(
            provider=os.environ.get("EVOSKILL_PROVIDER") or self.provider,
            model=os.environ.get("EVOSKILL_MODEL") or self.model,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or self.anthropic_api_key,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or self.openai_api_key,
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") or self.deepseek_api_key,
            data_dir=os.environ.get("EVOSKILL_HOME") or self.data_dir,
        )


_file_cache: Config | None = None
_runtime_overrides: dict[str, str] = {}


def get_config() -> Config:
    """Return active config: file values → env var overrides → CLI runtime overrides."""
    global _file_cache
    if _file_cache is None:
        _file_cache = _load_file()
    cfg = _file_cache.with_env()
    if _runtime_overrides:
        cfg = Config(**{f.name: _runtime_overrides.get(f.name, getattr(cfg, f.name)) for f in fields(cfg)})
    return cfg


def apply_overrides(**kwargs: str | None) -> None:
    """Apply runtime overrides from CLI flags (highest priority, non-persistent)."""
    _runtime_overrides.update({k: v for k, v in kwargs.items() if v is not None})


def save_config(cfg: Config) -> Path:
    """Persist config to ~/.evoskill/config.toml."""
    global _file_cache
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _write_toml(cfg)
    _file_cache = None
    return _CONFIG_FILE


def load_config_file() -> Config:
    """Load config from file only, without env or runtime overrides (for 'config set')."""
    return _load_file()


def get_config_file() -> Path:
    return _CONFIG_FILE


def _load_file() -> Config:
    if not _CONFIG_FILE.exists():
        return Config()
    with _CONFIG_FILE.open("rb") as f:
        data = tomllib.load(f)
    return Config(
        provider=data.get("provider", "anthropic"),
        model=data.get("model", "claude-sonnet-4-20250514"),
        anthropic_api_key=data.get("anthropic_api_key", ""),
        openai_api_key=data.get("openai_api_key", ""),
        deepseek_api_key=data.get("deepseek_api_key", ""),
        data_dir=data.get("data_dir", ""),
    )


def _toml_str(val: str) -> str:
    return val.replace("\\", "\\\\")


def _write_toml(cfg: Config) -> None:
    lines = [
        "# EvoSkill user configuration",
        "# Edit directly or use: evoskill config set KEY VALUE",
        "",
        "# LLM settings",
        f'provider = "{_toml_str(cfg.provider)}"',
        f'model = "{_toml_str(cfg.model)}"',
        "",
        "# API keys",
        f'anthropic_api_key = "{_toml_str(cfg.anthropic_api_key)}"',
        f'openai_api_key = "{_toml_str(cfg.openai_api_key)}"',
        f'deepseek_api_key = "{_toml_str(cfg.deepseek_api_key)}"',
        "",
        "# Storage",
        f'data_dir = "{_toml_str(cfg.data_dir)}"',
        "",
    ]
    _CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
