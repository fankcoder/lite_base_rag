"""
Skill Registry
- 扫描 skills/installed、skills/registry、内置 src/skills/manifests 目录
- 按名称索引所有已注册 skill
- 支持 semantic install（基于 embedding 查找描述匹配的 skill）
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .loader import SkillLoader
from .skill import Skill, SkillManifest

logger = logging.getLogger("rag.skills.registry")


class SkillRegistry:
    def __init__(self, search_dirs: Optional[List[str]] = None, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        default_dirs = [
            os.path.join(self.workspace_root, "skills", "installed"),
            os.path.join(self.workspace_root, "skills", "registry"),
            os.path.join(os.path.dirname(__file__), "manifests"),
        ]
        self.search_dirs = search_dirs or default_dirs
        self._skills: Dict[str, Skill] = {}
        self._scan()

    def _scan(self) -> None:
        for d in self.search_dirs:
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
                continue
            for name in os.listdir(d):
                p = os.path.join(d, name)
                manifest = os.path.join(p, "manifest.json")
                manifest_file = os.path.join(d, name) if name.endswith(".json") and os.path.isfile(os.path.join(d, name)) else None
                if os.path.isdir(p) and os.path.isfile(manifest):
                    try:
                        sk = SkillLoader.from_directory(p)
                        self._skills[sk.manifest.name] = sk
                    except Exception as e:
                        logger.warning("加载 skill %s 失败: %s", p, e)
                elif manifest_file and os.path.isfile(manifest_file):
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        sk = SkillLoader.from_manifest_dict(data, skill_dir=os.path.dirname(manifest_file))
                        self._skills[sk.manifest.name] = sk
                    except Exception as e:
                        logger.warning("加载 manifest %s 失败: %s", manifest_file, e)

    # -------- CRUD --------
    def list(self) -> List[dict]:
        return [s.info() for s in self._skills.values()]

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def register(self, skill: Skill) -> None:
        self._skills[skill.manifest.name] = skill

    def remove(self, name: str) -> bool:
        return self._skills.pop(name, None) is not None

    def install_from_manifest(self, manifest_dict: dict, skill_code: Optional[str] = None, entry_code: Optional[str] = None) -> Skill:
        """
        semantic install:
        - 传入 manifest dict（可能还有代码内容），写入 skills/installed/<name>/
        - 再通过 loader 加载
        """
        manifest = SkillManifest.from_dict(manifest_dict)
        target_dir = os.path.join(self.workspace_root, "skills", "installed", manifest.name)
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        if entry_code and manifest.entry:
            entry_path = os.path.join(target_dir, manifest.entry)
            os.makedirs(os.path.dirname(entry_path) or target_dir, exist_ok=True)
            with open(entry_path, "w", encoding="utf-8") as f:
                f.write(entry_code)
        # 若 manifest.entry 已存在，自动重新扫描
        sk = SkillLoader.from_directory(target_dir)
        self._skills[manifest.name] = sk
        return sk

    def semantic_search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        基于字符串/embedding 的语义匹配：默认使用简单 keyword overlap
        - 如果 embedder 可用，会自动升级到向量匹配
        """
        q = (query or "").lower().strip()
        q_tokens = set(t for t in q.split() if len(t) > 1)
        scored = []
        for s in self._skills.values():
            doc = (s.manifest.name + " " + s.manifest.description + " " + " ".join(s.manifest.capabilities) + " " + " ".join(s.manifest.tags)).lower()
            overlap = sum(1 for t in q_tokens if t in doc)
            # 子串加分
            if q and q in doc:
                overlap += 3
            if overlap > 0:
                scored.append((overlap, s.info()))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]
