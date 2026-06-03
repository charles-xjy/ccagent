import asyncio
import os
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """使用 DuckDuckGo 搜索网络信息，无需 API Key。返回标题、URL 和摘要。"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "未找到相关结果。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


async def _connect_server(name: str, config: dict, max_retries: int = 3, delay: float = 2.0) -> list:
    """连接单个 MCP 服务器，失败后自动重试，全部失败返回空列表。"""
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
    """
    并发连接所有 MCP 服务器，各自独立重试，返回所有成功服务器的工具合并列表。

    支持的服务器：
    - github        搜索仓库/查看代码/阅读 issue·PR（需设 GITHUB_PERSONAL_ACCESS_TOKEN）
    - fetch         抓取任意网页内容，无需 API Key
    - langchain_docs LangChain 官方文档（HTTP，连接失败自动重试）
    """
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


def create_mcp_client() -> MultiServerMCPClient:
    """兼容旧接口，返回 fetch 服务器的客户端（单服务器模式）。"""
    return MultiServerMCPClient({"fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": dict(os.environ),
        "transport": "stdio",
    }})
