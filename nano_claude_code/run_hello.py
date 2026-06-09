"""
把 Hello World 发到 CubeSandbox 沙箱执行，打印返回结果。

用法:
    python nano_claude_code/run_hello.py

所需环境变量（.env 或 shell 中设置）:
    CUBE_TEMPLATE_ID   — 沙箱模板 ID（必填）
    E2B_API_URL        — 自托管沙箱地址（可选，云端 e2b 不需要）
    CUBE_PROXY_IP      — Windows DNS 代理 IP（可选）
    SSL_CERT_FILE      — 自签名 CA 路径（可选）
"""

import os
import sys

from dotenv import load_dotenv

# .env 在 nano_claude_code/ 下，显式指定路径
_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_FILE)

# sandbox.py 在 nano_claude_code/core/ 下，把项目根加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nano_claude_code.core.sandbox import get_sandbox, reset_sandbox

HELLO_CODE = """\
message = "Hello, World!"
print(message)
print(f"Python 版本: {__import__('sys').version}")
print(f"运行平台: {__import__('platform').system()} {__import__('platform').release()}")
"""


def main():
    print("[*] 正在连接沙箱...")
    sb = get_sandbox()
    print("[OK] 沙箱已就绪，开始执行代码...\n")

    result = sb.run_code(HELLO_CODE)

    print("=" * 40)
    print("  沙箱返回结果")
    print("=" * 40)
    print(result)
    print("=" * 40)

    reset_sandbox()
    print("\n[OK] 沙箱已销毁。")


if __name__ == "__main__":
    main()
