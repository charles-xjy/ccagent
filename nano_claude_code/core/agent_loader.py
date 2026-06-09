"""
Skill 加载器。

skills/*.md 格式：
  ---
  description: 一行描述（用于构建 skill 目录）
  ---
  正文内容（注入到 agent 上下文）
"""

from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_BASE = Path(__file__).parent.parent
SKILLS_DIR = _BASE / "skills"


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """解析 YAML frontmatter，返回 (meta_dict, body_str)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_meta = parts[1].strip()
    body = parts[2].strip()
    if not raw_meta:
        return {}, body
    if _YAML_OK:
        try:
            meta = yaml.safe_load(raw_meta) or {}
        except Exception:
            meta = {}
    else:
        meta = {}
        for line in raw_meta.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
    return meta, body


def load_skill(name: str) -> str:
    """加载 skills/{name}.md 的正文内容。"""
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        available = [f.stem for f in SKILLS_DIR.glob("*.md")] if SKILLS_DIR.exists() else []
        return f"[!] skill '{name}' 不存在。可用: {available}"
    _, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    return body


def list_skills() -> List[Dict]:
    """扫描 skills/ 目录，返回 [{name, description, allowed_tools}] 列表。"""
    if not SKILLS_DIR.exists():
        return []
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.md")):
        meta, _ = _parse_frontmatter(f.read_text(encoding="utf-8"))
        skills.append({
            "name":          f.stem,
            "description":   meta.get("description", ""),
            "allowed_tools": meta.get("allowed_tools"),  # None 表示无限制
        })
    return skills


def resolve_allowed_tools(skill_names: List[str]) -> set | None:
    """
    计算一组 skill 的工具白名单（取并集）。

    规则（同 DeerFlow）：
    - 所有 skill 都没有 allowed_tools → 返回 None（不过滤，全量给）
    - 任意 skill 声明了 allowed_tools → 返回并集（硬过滤）
    """
    allowed: set = set()
    has_declaration = False
    for name in skill_names:
        path = SKILLS_DIR / f"{name}.md"
        if not path.exists():
            continue
        meta, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))
        tools = meta.get("allowed_tools")
        if tools is not None:
            has_declaration = True
            allowed.update(tools)
    return allowed if has_declaration else None


def build_skill_index() -> str:
    """构建 skill 目录字符串，注入 system prompt。"""
    skills = list_skills()
    if not skills:
        return ""
    lines = [f"- `{s['name']}`: {s['description']}" for s in skills]
    return "## 可用知识库（需要时调用 load_skill 工具获取详情）\n" + "\n".join(lines)
