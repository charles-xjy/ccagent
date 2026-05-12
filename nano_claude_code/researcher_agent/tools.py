# researcher 特有工具 (如 mcp) 可以在这里定义
from langchain_mcp_adapters.client import MultiServerMCPClient


async def fetch_mcp_tools():
    """连接远程 MCP 服务器并获取动态工具"""
    mcp_servers = {
        "langchain_docs": {
            "url": "https://docs.langchain.com/mcp",
            "transport": "http"
        }
    }
    client = MultiServerMCPClient(mcp_servers)
    try:
        return await client.get_tools()
    except Exception as e:
        print(f"[!] MCP 连接失败: {e}")
        return []
