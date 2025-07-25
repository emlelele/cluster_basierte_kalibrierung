from __future__ import annotations

from pathlib import Path

from dynaconf import Dynaconf

settings_files = ["settings.toml", ".secrets.toml"]
settings_files_absolute = [
    Path(__file__).resolve().parent / file for file in settings_files
]

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=settings_files_absolute,
)
