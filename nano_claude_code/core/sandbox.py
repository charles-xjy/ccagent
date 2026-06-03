import os
import atexit
from typing import Optional

_instance: Optional["SandboxSession"] = None

WORKDIR = "/home/user/workspace"


class SandboxSession:
    def __init__(self):
        from e2b_code_interpreter import Sandbox

        template = os.environ.get("CUBE_TEMPLATE_ID", "python3")
        self._sb = Sandbox(template=template)
        self._sb.commands.run(f"mkdir -p {WORKDIR}")

    def run_command(self, command: str, timeout: int = 120) -> str:
        result = self._sb.commands.run(command, workdir=WORKDIR, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()[:5000]

    def run_code(self, code: str) -> str:
        execution = self._sb.run_code(code)
        lines = list(execution.logs.stdout) + list(execution.logs.stderr)
        output = "\n".join(lines)
        if execution.error:
            output += f"\nError: {execution.error.name}: {execution.error.value}"
        return output.strip()[:5000]

    def write_file(self, path: str, content: str) -> None:
        full_path = f"{WORKDIR}/{path.lstrip('/')}"
        self._sb.files.write(full_path, content)

    def read_file(self, path: str) -> str:
        full_path = f"{WORKDIR}/{path.lstrip('/')}"
        return self._sb.files.read(full_path)

    def kill(self):
        try:
            self._sb.kill()
        except Exception:
            pass


def get_sandbox() -> SandboxSession:
    global _instance
    if _instance is None:
        _instance = SandboxSession()
        atexit.register(_instance.kill)
    return _instance


def reset_sandbox():
    """销毁当前沙箱实例，下次调用 get_sandbox() 时创建新实例。"""
    global _instance
    if _instance is not None:
        _instance.kill()
        _instance = None
