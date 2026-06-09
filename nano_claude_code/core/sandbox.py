import os
import re
import socket
import atexit
import tempfile
from pathlib import Path
from typing import Optional

# ── DNS patch: resolve *.cube.app → CUBE_PROXY_IP (avoids wildcard DNS on Windows) ──
_CUBE_APP_RE = re.compile(r".+\.cube\.app$", re.IGNORECASE)
_orig_getaddrinfo = socket.getaddrinfo

def _patched_getaddrinfo(host, port, *args, **kwargs):
    proxy_ip = os.environ.get("CUBE_PROXY_IP", "")
    if proxy_ip and isinstance(host, str) and _CUBE_APP_RE.match(host):
        host = proxy_ip
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = _patched_getaddrinfo


# ── SSL patch ─────────────────────────────────────────────────────────────────
# Self-hosted CubeSandbox uses a mkcert self-signed wildcard cert for *.cube.app.
# Python's certifi bundle does NOT include system-installed mkcert CAs, so SSL
# verification fails even on the server itself.
#
# We disable verification globally whenever E2B_API_URL points to a self-hosted
# endpoint (http://...), covering both Windows (needs DNS patch too) and Linux.
# If SSL_CERT_FILE is provided, we inject the CA into certifi instead.
def _setup_ssl():
    import ssl as _ssl

    # Option A: explicit CA cert → inject into certifi (proper solution).
    # Only treat SSL_CERT_FILE as a custom CA when E2B_API_URL is NOT set
    # (cloud e2b). When self-hosted (E2B_API_URL set), the server uses a
    # mkcert self-signed cert that is NOT in any standard bundle, so we fall
    # through to Option B which disables verification entirely.
    api_url = os.environ.get("E2B_API_URL", "")
    custom_ca = os.environ.get("SSL_CERT_FILE", "")
    if custom_ca and Path(custom_ca).exists() and not api_url:
        try:
            import certifi
            combined = Path(tempfile.gettempdir()) / "cube-combined-ca.pem"
            if not combined.exists():
                combined.write_text(
                    Path(certifi.where()).read_text(encoding="utf-8")
                    + "\n"
                    + Path(custom_ca).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            certifi.where = lambda: str(combined)
        except Exception:
            pass
        return

    # Option B: self-hosted deployment (E2B_API_URL set) → disable SSL verify.
    # E2B_API_URL is only set for self-hosted CubeSandbox; cloud e2b doesn't use it.
    if not api_url:
        return

    # Layer 1: override the global default HTTPS context factory.
    # httpcore uses ssl.create_default_context() when ssl_context is None,
    # and many other libraries go through this same chokepoint.
    _ssl._create_default_https_context = _ssl._create_unverified_context

    # Layer 2: patch httpx.HTTPTransport to pass verify=False explicitly,
    # covering the path where httpx builds its own SSL context.
    try:
        import httpx
        _orig = httpx.HTTPTransport.__init__
        def _noverify(self, *a, **kw):
            kw["verify"] = False
            _orig(self, *a, **kw)
        httpx.HTTPTransport.__init__ = _noverify

        # Invalidate cached transports built before this patch ran.
        from e2b.api.client_sync import TransportWithLogger, EnvdTransportWithLogger
        TransportWithLogger._instances.clear()
        if hasattr(EnvdTransportWithLogger._thread_local, "instances"):
            EnvdTransportWithLogger._thread_local.instances = {}
    except Exception:
        pass

_setup_ssl()

_instance: Optional["SandboxSession"] = None


class SandboxSession:
    def __init__(self):
        from e2b_code_interpreter import Sandbox

        template = os.environ.get("CUBE_TEMPLATE_ID", "")
        if not template:
            raise RuntimeError("CUBE_TEMPLATE_ID 未设置，请在 .env 中配置")
        self._sb = Sandbox.create(template=template)

    def run_command(self, command: str, timeout: int = 120) -> str:
        result = self._sb.commands.run(command, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()[:5000]

    def run_code(self, code: str) -> str:
        execution = self._sb.run_code(code)
        lines = list(execution.logs.stdout) + list(execution.logs.stderr)
        output = "\n".join(lines)
        if execution.error:
            output += f"\nError: {execution.error.name}: {execution.error.value}"
        return output.strip()[:5000]

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
