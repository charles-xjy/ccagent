import os
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """使用 DuckDuckGo 搜索网络信息，无需 API Key。返回标题、URL 和摘要。"""
    try:
        from duckduckgo_search import DDGS
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


def create_mcp_client() -> MultiServerMCPClient:
    """
    构建研究员 MCP 客户端配置：

    - github  搜索仓库/查看代码/阅读 issue·PR（需设 GITHUB_TOKEN）
    - fetch   抓取任意网页内容，无需 API Key

    用法（langchain-mcp-adapters >= 0.1.0 不支持 context manager）：
        tools = await create_mcp_client().get_tools()
    """
    servers = {}

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if github_token:
        servers["github"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            "transport": "stdio",
        }
    else:
        print("\033[33m[!] 未设置 GITHUB_PERSONAL_ACCESS_TOKEN，GitHub MCP 工具不可用\033[0m")

    servers["fetch"] = {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": dict(os.environ),
        "transport": "stdio",
    }

    # LangChain 官方文档 MCP（HTTP，无需认证）
    servers["langchain_docs"] = {
        "url": "https://docs.langchain.com/mcp",
        "transport": "streamable_http",
    }

    return MultiServerMCPClient(servers)
