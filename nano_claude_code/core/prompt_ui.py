"""
方向键选择菜单（依赖 questionary，未安装时降级为数字菜单）
"""
from typing import List, Tuple

try:
    import questionary
    _HAS_Q = True
except ImportError:
    _HAS_Q = False


async def select(message: str, choices: List[Tuple[str, str]]) -> str:
    """
    显示方向键菜单，返回所选项的 value。
    必须在 async 上下文中 await 调用。
    choices: [(显示文字, 返回值), ...]
    """
    if _HAS_Q:
        result = await questionary.select(
            message,
            choices=[questionary.Choice(label, value=val) for label, val in choices],
            use_shortcuts=False,
            style=questionary.Style([
                ("selected", "fg:cyan bold"),
                ("pointer",  "fg:cyan bold"),
                ("question", "bold"),
            ]),
        ).ask_async()
        return result if result is not None else choices[-1][1]

    # fallback：数字菜单
    print(f"\n{message}")
    for i, (label, _) in enumerate(choices, 1):
        print(f"  {i}. {label}")
    raw = input("请输入数字: ").strip()
    try:
        return choices[int(raw) - 1][1]
    except (ValueError, IndexError):
        return choices[-1][1]
