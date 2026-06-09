"""
Tests for nano_claude_code/core/sandbox.py

运行方式:
    pytest nano_claude_code/tests/test_sandbox.py -v
"""

import importlib
import os
import socket
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_sandbox():
    """重新加载 sandbox 模块，使环境变量的改动生效。"""
    mod_name = "nano_claude_code.core.sandbox"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# 1. DNS patch
# ---------------------------------------------------------------------------

class TestDnsPatch:
    """_patched_getaddrinfo 应该把 *.cube.app 域名替换为 CUBE_PROXY_IP。"""

    def setup_method(self):
        import nano_claude_code.core.sandbox as sb
        self.sb = sb
        self._orig = sb._orig_getaddrinfo

    def test_cube_app_host_replaced(self, monkeypatch):
        proxy_ip = "192.168.1.100"
        monkeypatch.setenv("CUBE_PROXY_IP", proxy_ip)

        captured = {}

        def fake_getaddrinfo(host, port, *a, **kw):
            captured["host"] = host
            return []

        monkeypatch.setattr(self.sb, "_orig_getaddrinfo", fake_getaddrinfo)

        self.sb._patched_getaddrinfo("my-sandbox.cube.app", 443)
        assert captured["host"] == proxy_ip

    def test_non_cube_host_unchanged(self, monkeypatch):
        monkeypatch.setenv("CUBE_PROXY_IP", "192.168.1.100")

        captured = {}

        def fake_getaddrinfo(host, port, *a, **kw):
            captured["host"] = host
            return []

        monkeypatch.setattr(self.sb, "_orig_getaddrinfo", fake_getaddrinfo)

        self.sb._patched_getaddrinfo("example.com", 80)
        assert captured["host"] == "example.com"

    def test_no_proxy_ip_env_host_unchanged(self, monkeypatch):
        monkeypatch.delenv("CUBE_PROXY_IP", raising=False)

        captured = {}

        def fake_getaddrinfo(host, port, *a, **kw):
            captured["host"] = host
            return []

        monkeypatch.setattr(self.sb, "_orig_getaddrinfo", fake_getaddrinfo)

        self.sb._patched_getaddrinfo("my-sandbox.cube.app", 443)
        assert captured["host"] == "my-sandbox.cube.app"

    def test_cube_app_pattern_case_insensitive(self, monkeypatch):
        proxy_ip = "10.0.0.1"
        monkeypatch.setenv("CUBE_PROXY_IP", proxy_ip)

        captured = {}

        def fake_getaddrinfo(host, port, *a, **kw):
            captured["host"] = host
            return []

        monkeypatch.setattr(self.sb, "_orig_getaddrinfo", fake_getaddrinfo)

        self.sb._patched_getaddrinfo("My-Sandbox.CUBE.APP", 443)
        assert captured["host"] == proxy_ip

    def test_socket_getaddrinfo_is_patched(self):
        """模块加载后 socket.getaddrinfo 应该已被替换。"""
        import nano_claude_code.core.sandbox as sb
        assert socket.getaddrinfo is sb._patched_getaddrinfo


# ---------------------------------------------------------------------------
# 2. SSL setup
# ---------------------------------------------------------------------------

class TestSetupSsl:
    """_setup_ssl 根据环境变量选择 Option A（注入 CA）或 Option B（禁用验证）。"""

    def test_option_a_custom_ca_injected(self, monkeypatch, tmp_path):
        """SSL_CERT_FILE 存在时，将自定义 CA 追加到 certifi bundle。"""
        ca_file = tmp_path / "custom-ca.pem"
        ca_file.write_text("-----BEGIN CERTIFICATE-----\nFAKECA\n-----END CERTIFICATE-----\n")

        monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
        monkeypatch.delenv("E2B_API_URL", raising=False)

        fake_certifi = types.SimpleNamespace(
            where=lambda: str(ca_file),
        )
        # certifi.where 会被替换为返回合并文件路径的 lambda
        patched_where = MagicMock(return_value=str(ca_file))
        fake_certifi.where = patched_where

        # 确保 combined 文件不存在，触发写入逻辑
        combined = Path(tempfile.gettempdir()) / "cube-combined-ca.pem"
        if combined.exists():
            combined.unlink()

        with patch.dict("sys.modules", {"certifi": fake_certifi}):
            import nano_claude_code.core.sandbox as sb
            sb._setup_ssl()

        # combined 文件应当被创建
        assert combined.exists()
        content = combined.read_text(encoding="utf-8")
        assert "FAKECA" in content

    def test_option_b_no_env_vars_no_op(self, monkeypatch):
        """没有环境变量时，_setup_ssl 不修改 ssl 模块。"""
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("E2B_API_URL", raising=False)

        import ssl as _ssl
        original_ctx = _ssl._create_default_https_context

        import nano_claude_code.core.sandbox as sb
        sb._setup_ssl()

        assert _ssl._create_default_https_context is original_ctx

    def test_option_b_e2b_api_url_disables_ssl(self, monkeypatch):
        """E2B_API_URL 设置时，应将默认 HTTPS context 替换为不验证版本。"""
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setenv("E2B_API_URL", "http://my-cube-server:3000")

        import ssl as _ssl

        import nano_claude_code.core.sandbox as sb
        sb._setup_ssl()

        assert _ssl._create_default_https_context is _ssl._create_unverified_context


# ---------------------------------------------------------------------------
# 3. SandboxSession
# ---------------------------------------------------------------------------

class TestSandboxSession:
    """SandboxSession 封装 e2b_code_interpreter.Sandbox，测试各方法行为。"""

    def _make_fake_e2b(self):
        """构造一个最小化的 e2b_code_interpreter 模拟模块。"""
        fake_sb_instance = MagicMock()
        fake_sb_cls = MagicMock(return_value=fake_sb_instance)
        fake_sb_cls.create = MagicMock(return_value=fake_sb_instance)

        fake_module = types.ModuleType("e2b_code_interpreter")
        fake_module.Sandbox = fake_sb_cls
        return fake_module, fake_sb_cls, fake_sb_instance

    def test_init_raises_without_template(self, monkeypatch):
        monkeypatch.delenv("CUBE_TEMPLATE_ID", raising=False)

        fake_mod, _, _ = self._make_fake_e2b()
        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            with pytest.raises(RuntimeError, match="CUBE_TEMPLATE_ID"):
                sb.SandboxSession()

    def test_init_uses_template_id(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "my-template")

        fake_mod, fake_cls, _ = self._make_fake_e2b()
        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            sb.SandboxSession()
            fake_cls.create.assert_called_once_with(template="my-template")

    def test_run_command_returns_combined_output(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_result = MagicMock()
        fake_result.stdout = "hello "
        fake_result.stderr = "world"
        fake_sb.commands.run.return_value = fake_result

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            out = session.run_command("echo hello")

        assert out == "hello world"
        fake_sb.commands.run.assert_called_once_with("echo hello", timeout=120)

    def test_run_command_custom_timeout(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_result = MagicMock(stdout="ok", stderr="")
        fake_sb.commands.run.return_value = fake_result

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            session.run_command("sleep 1", timeout=30)

        fake_sb.commands.run.assert_called_once_with("sleep 1", timeout=30)

    def test_run_command_truncates_at_5000(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_result = MagicMock(stdout="x" * 6000, stderr="")
        fake_sb.commands.run.return_value = fake_result

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            out = session.run_command("big output")

        assert len(out) == 5000

    def test_run_code_returns_stdout_stderr(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_exec = MagicMock()
        fake_exec.logs.stdout = ["line1\n", "line2\n"]
        fake_exec.logs.stderr = ["warn\n"]
        fake_exec.error = None
        fake_sb.run_code.return_value = fake_exec

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            out = session.run_code("print('hi')")

        assert "line1" in out
        assert "warn" in out

    def test_run_code_appends_error(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_exec = MagicMock()
        fake_exec.logs.stdout = []
        fake_exec.logs.stderr = []
        fake_exec.error = MagicMock(name="NameError", value="x is not defined")
        fake_exec.error.name = "NameError"
        fake_exec.error.value = "x is not defined"
        fake_sb.run_code.return_value = fake_exec

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            out = session.run_code("x")

        assert "NameError" in out
        assert "x is not defined" in out

    def test_run_code_truncates_at_5000(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_exec = MagicMock()
        fake_exec.logs.stdout = ["y" * 6000]
        fake_exec.logs.stderr = []
        fake_exec.error = None
        fake_sb.run_code.return_value = fake_exec

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            out = session.run_code("...")

        assert len(out) == 5000

    def test_kill_calls_sandbox_kill(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            session.kill()

        fake_sb.kill.assert_called_once()

    def test_kill_swallows_exceptions(self, monkeypatch):
        monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl")

        fake_mod, _, fake_sb = self._make_fake_e2b()
        fake_sb.kill.side_effect = RuntimeError("already dead")

        with patch.dict("sys.modules", {"e2b_code_interpreter": fake_mod}):
            import nano_claude_code.core.sandbox as sb
            session = sb.SandboxSession()
            session.kill()  # 不应抛出异常


# ---------------------------------------------------------------------------
# 4. get_sandbox / reset_sandbox  (singleton 行为)
# ---------------------------------------------------------------------------

class TestGetSandboxSingleton:

    def _patch_session(self, monkeypatch, fake_session):
        """把 SandboxSession 替换为返回 fake_session 的工厂。"""
        import nano_claude_code.core.sandbox as sb
        monkeypatch.setattr(sb, "SandboxSession", lambda: fake_session)
        return sb

    def test_get_sandbox_returns_same_instance(self, monkeypatch):
        import nano_claude_code.core.sandbox as sb
        sb._instance = None  # 重置单例

        fake = MagicMock()
        monkeypatch.setattr(sb, "SandboxSession", lambda: fake)

        s1 = sb.get_sandbox()
        s2 = sb.get_sandbox()

        assert s1 is s2
        assert s1 is fake

        sb._instance = None  # 清理

    def test_get_sandbox_registers_atexit(self, monkeypatch):
        import nano_claude_code.core.sandbox as sb
        sb._instance = None

        fake = MagicMock()
        monkeypatch.setattr(sb, "SandboxSession", lambda: fake)

        registered = []
        monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn))

        sb.get_sandbox()

        assert len(registered) == 1
        registered[0]()
        fake.kill.assert_called_once()

        sb._instance = None

    def test_reset_sandbox_kills_and_clears(self, monkeypatch):
        import nano_claude_code.core.sandbox as sb

        fake = MagicMock()
        sb._instance = fake

        sb.reset_sandbox()

        fake.kill.assert_called_once()
        assert sb._instance is None

    def test_reset_sandbox_noop_when_none(self):
        import nano_claude_code.core.sandbox as sb
        sb._instance = None
        sb.reset_sandbox()  # 不应抛出
        assert sb._instance is None

    def test_get_sandbox_after_reset_creates_new(self, monkeypatch):
        import nano_claude_code.core.sandbox as sb
        sb._instance = None

        calls = []

        class FakeSession:
            def __init__(self):
                calls.append(self)
            def kill(self):
                pass

        monkeypatch.setattr(sb, "SandboxSession", FakeSession)
        monkeypatch.setattr("atexit.register", lambda fn: None)

        s1 = sb.get_sandbox()
        sb.reset_sandbox()
        s2 = sb.get_sandbox()

        assert s1 is not s2
        assert len(calls) == 2

        sb._instance = None
