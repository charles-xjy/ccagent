import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def _connect_server(name: str, config: dict, max_retries: int = 3, delay: float = 2.0) -> list:
    for attempt in range(1, max_retries + 1):
        try:
            tools = await MultiServerMCPClient({name: config}).get_tools()
            if tools:
                return tools
        except Exception:
            pass
        if attempt < max_retries:
            print(f"\033[33m[!] MCP {name} 第{attempt}次失败，{delay}s 后重试...\033[0m")
            await asyncio.sleep(delay)
    print(f"\033[33m[!] MCP {name} 连接失败，已跳过\033[0m")
    return []


async def get_mcp_tools() -> list:
    """并发连接所有 MCP 服务器，返回所有成功服务器的工具合并列表。"""
    tasks = {}

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if github_token:
        tasks["github"] = _connect_server("github", {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            "transport": "stdio",
        })
    else:
        print("\033[33m[!] 未设置 GITHUB_PERSONAL_ACCESS_TOKEN，GitHub MCP 工具不可用\033[0m")

    tasks["fetch"] = _connect_server("fetch", {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": dict(os.environ),
        "transport": "stdio",
    })

    tasks["langchain_docs"] = _connect_server("langchain_docs", {
        "url": "https://docs.langchain.com/mcp",
        "transport": "streamable_http",
    })

    results = await asyncio.gather(*tasks.values())

    all_tools = []
    for name, tools in zip(tasks.keys(), results):
        if tools:
            print(f"\033[32m[OK] MCP {name}: {len(tools)} 个工具\033[0m")
            all_tools.extend(tools)

    return all_tools
