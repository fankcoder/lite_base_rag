"""
OpenClaw-style Skill system
目录布局：
  skills/registry      — 内置 registry（manifest json 索引）
  skills/installed     — 用户安装的 skill 目录
  skills/runtime       — 运行时加载/执行入口
  skills/manifests     — 存放 manifest json
"""
from .skill import Skill, SkillManifest, SkillContext, SkillResult
from .loader import SkillLoader
from .registry import SkillRegistry
from .runtime import SkillRuntime

__all__ = [
    "Skill", "SkillManifest", "SkillContext", "SkillResult",
    "SkillLoader", "SkillRegistry", "SkillRuntime",
]
