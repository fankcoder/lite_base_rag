"""
Skill Loader
- 从 manifest.json 加载 SkillManifest
- 支持按 path 或名称加载
- lazy import entry module
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .skill import Skill, SkillManifest

logger = logging.getLogger("rag.skills.loader")


class SkillLoader:
    """从目录 / manifest dict 加载 Skill 实例"""

    @staticmethod
    def from_directory(skill_dir: str) -> Skill:
        skill_dir = os.path.abspath(skill_dir)
        manifest_path = os.path.join(skill_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"manifest.json 不存在: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = SkillManifest.from_dict(data)
        return SkillLoader._build(manifest, skill_dir)

    @staticmethod
    def from_manifest_dict(data: Dict[str, Any], skill_dir: Optional[str] = None) -> Skill:
        manifest = SkillManifest.from_dict(data)
        return SkillLoader._build(manifest, skill_dir or os.getcwd())

    @staticmethod
    def _build(manifest: SkillManifest, skill_dir: str) -> Skill:
        # 解析 entry 路径（相对 skill_dir）
        entry_path = manifest.entry
        if entry_path and not os.path.isabs(entry_path):
            candidate = os.path.join(skill_dir, entry_path)
            if os.path.isfile(candidate):
                entry_path = candidate
        module = None
        run_fn = None
        if entry_path and os.path.isfile(entry_path):
            module = SkillLoader._load_module(manifest.name, entry_path)
            # 约定：entry 模块提供 run(ctx) -> SkillResult
            run_fn = getattr(module, "run", None)

        return Skill(
            manifest=manifest,
            skill_dir=skill_dir,
            entry_path=entry_path,
            module=module,
            run_fn=run_fn,
        )

    @staticmethod
    def _load_module(name: str, path: str):
        spec = importlib.util.spec_from_file_location(f"skill_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 skill 模块: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"skill_{name}"] = module
        spec.loader.exec_module(module)
        return module
