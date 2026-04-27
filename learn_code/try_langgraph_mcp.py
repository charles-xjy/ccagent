from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio


async def langgraph_mcp():
    # 3. 配置 LangChain 官方文档 MCP 服务器
    #     这里的 URL 是你提供的官方 MCP 端点
    mcp_servers = {
        "langchain_docs": {
            "url": "https://docs.langchain.com/mcp",
            "transport": "http"
        }
    }
    print(f"[*] 正在尝试连接到 LangChain 官方 MCP 服务器...")

    mcp_client = MultiServerMCPClient(mcp_servers)
    try:
        # 获取 MCP 工具
        mcp_tools = await mcp_client.get_tools()
        print(f"[*] 连接成功！已加载 {len(mcp_tools)} 个文档搜索相关工具")
        print(mcp_tools)
    finally:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(langgraph_mcp())
    except KeyboardInterrupt:
        pass
