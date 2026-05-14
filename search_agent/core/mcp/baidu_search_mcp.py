"""
百度 MCP (Model Context Protocol) 集成工具类

提供 MCP 协议客户端实现，支持 HTTP/WebSocket 连接和 JSON-RPC 2.0 协议
集成百度搜索引擎 MCP 服务
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== 异常类定义 ====================

class MCPException(Exception):
    """MCP 基类异常"""
    def __init__(self, message: str, code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class MCPConnectionError(MCPException):
    """MCP 连接异常"""
    pass


class MCPParseError(MCPException):
    """MPC解析异常"""
    pass


class MCPToolError(MCPException):
    """MCP 工具调用异常"""
    pass


class MCPServiceUnavailable(MCPException):
    """MCP 服务不可用"""
    pass


class MCPRetryExhausted(MCPException):
    """MCP 重试耗尽"""
    pass


# ==================== MCP 协议常量 ====================

MCP_PROTOCOL_VERSION = "2.0"
DEFAULT_RETRY_ATTEMPTS = 3
BASE_RETRY_DELAY = 1.0  # 秒


# ==================== Pydantic 模型定义 ====================

class MCPRequest(BaseModel):
    """JSON-RPC 请求模型"""
    jsonrpc: str = Field(MCP_PROTOCOL_VERSION, description="JSON-RPC 版本")
    method: str = Field(..., description="方法名称")
    params: Optional[Dict[str, Any]] = Field(default=None, description="参数")
    id: Optional[Union[str, int]] = Field(None, description="请求 ID")


class MCPResponse(BaseModel):
    """JSON-RPC 响应模型"""
    jsonrpc: str = Field(MCP_PROTOCOL_VERSION, description="JSON-RPC 版本")
    result: Optional[Dict[str, Any]] = Field(None, description="响应结果")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")
    id: Optional[Union[str, int]] = Field(None, description="响应 ID")


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# ==================== 百度搜索引擎 MCP 服务信息 ====================

BAIDU_MCP_INFO = {
    "version": "1.0.0",
    "description": "百度搜索引擎 Model Context Protocol",
    "capabilities": {
        "tools": ["baidu_search", "serp_features", "knowledge_graph"]
    }
}


# ==================== 核心 MCP 客户端类 ====================

class BaiduSearchMCP:
    """
    百度搜索引擎 MCP 客户端
    
    提供 MCP 协议客户端实现，支持 HTTP/WebSocket 通信
    
    Attributes:
        url: MCP 服务地址
        token: 认证令牌
        retry_count: 当前重试计数
    
    Example:
        >>> client = BaiduSearchMCP(url="wss://mcp.baidu.com/ws", token="your_token")
        >>> await client.connect()
        >>> tools = await client.list_tools()
        >>> results = await client.call_tool("baidu_search", params={"query": "AI"})
    """
    
    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        protocol: str = "wss",
        timeout: int = 30,
        client_name: str = "search_agent"
    ):
        """
        初始化 MCP 客户端
        
        Args:
            url: MCP 服务 WebSocket 地址（不含协议前缀）
            token: OAuth 认证令牌（可选）
            protocol: 协议类型 (ws/wss)
            timeout: 请求超时时间（秒）
            client_name: 客户端名称
        """
        self.url = url
        self.token = token
        self.protocol = protocol
        self.timeout = timeout
        self.client_name = client_name
        
        # 状态管理
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._retry_count = 0
        self._max_retries = DEFAULT_RETRY_ATTEMPTS
        
        # 工具缓存
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._tools_last_fetch: Optional[datetime] = None
        self._TOOLS_CACHE_TTL = 300  # 5 分钟
        
        # 历史消息
        self._message_history: List[Dict[str, Any]] = []
        self._message_id_counter = 0

    @property
    def is_connected(self) -> bool:
        """检查当前连接状态"""
        return self._connection_state == ConnectionState.CONNECTED

    @property
    def connection_state(self) -> ConnectionState:
        """获取当前连接状态"""
        return self._connection_state

    @property
    def tools(self) -> Dict[str, Dict[str, Any]]:
        """获取工具信息（带缓存）"""
        if self._tools_last_fetch is None:
            self._tools = {}
            self._tools_last_fetch = None
        elif (datetime.now() - self._tools_last_fetch).total_seconds() > self._TOOLS_CACHE_TTL:
            self._tools = {}
            self._tools_last_fetch = None
        return self._tools

    # ==================== 连接管理 ====================

    async def connect(self) -> bool:
        """
        建立 WebSocket 连接
        
        Returns:
            True 如果连接成功，否则 False
        
        Raises:
            MCPConnectionError: 连接失败
        """
        if self.is_connected:
            logger.debug(f"客户端 '{self.client_name}' 已连接")
            return True

        try:
            self._retry_count = 0
            
            if ':' in self.url and self.url.startswith(('http://', 'https://', 'ws://', 'wss://')):
                self._ws_url = self.url
            else:
                self._ws_url = f"ws://{self.url}" if self.protocol == 'ws' else f"wss://{self.url}"
            
            logger.info(f"客户端 '{self.client_name}' 正在连接到 {self._ws_url}")
            self._connection_state = ConnectionState.CONNECTING

            connector = aiohttp.TCPConnector(ssl=False)
            
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                self._session = session
                ws = await session.ws_connect(
                    self._ws_url,
                    timeout=aiohttp.ClientWSTimeout(ws_close=self.timeout, ws_close_wait=self.timeout)
                )
                
                self._ws = ws
                
                # 发送握手消息
                await self._send_message({
                    "jsonrpc": MCP_PROTOCOL_VERSION,
                    "method": "mcp/connect",
                    "params": {
                        "client_name": self.client_name,
                        "server_info": BAIDU_MCP_INFO
                    }
                })
                
                # 等待握手响应
                response = await self._receive_response()
                if response.get("error") is None:
                    self._connection_state = ConnectionState.CONNECTED
                    self._tools = {}  # 清空工具缓存
                    self._tools_last_fetch = None
                    logger.info(f"客户端 '{self.client_name}' 连接成功")
                    return True
                else:
                    self._connection_state = ConnectionState.ERROR
                    logger.error(f"MCP 握手失败：{json.dumps(response.get('error'))}")
                    return False
                    
        except aiohttp.ClientError as e:
            self._connection_state = ConnectionState.ERROR
            logger.error(f"客户端 '{self.client_name}' WebSocket 连接失败：{e}")
            raise MCPConnectionError(
                f"无法连接到 MCP 服务：{e}",
                code=503,
                details={
                    "url": self.url,
                    "timeout": self.timeout
                }
            )
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            logger.exception(f"客户端 '{self.client_name}' 未知错误：{e}")
            raise MCPConnectionError(f"连接异常：{e}", code=500)

    async def disconnect(self) -> None:
        """断开 WebSocket 连接"""
        logger.info(f"客户端 '{self.client_name}' 正在断开连接...")
        
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
                self._ws = None
        except Exception as e:
            logger.warning(f"断开 WebSocket 时出错：{e}")
            pass
        
        self._session = None
        self._connection_state = ConnectionState.DISCONNECTED
        logger.info(f"客户端 '{self.client_name}' 已断开连接")

    async def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            健康检查结果字典
        """
        if not self.is_connected:
            return {
                "status": "disconnected",
                "connected": False,
                "error": "未连接"
            }
        
        try:
            # 使用 ping 方法进行健康检查
            await self._receive_response(timeout=5)
            
            return {
                "status": "healthy",
                "connected": True,
                "ping": "pong",
                "timestamp": datetime.now().isoformat(),
                "tools_cached": len(self.tools)
            }
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "connected": True,
                "error": "Ping 超时",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"健康检查失败：{e}")
            return {
                "status": "error",
                "connected": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ==================== 工具查询 ====================

    async def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        获取可用工具列表
        
        Returns:
            工具字典，格式：{tool_name: {schema, description, etc.}}
        """
        if not self.is_connected:
            tools = self._get_cached_tools()
            logger.warning("未连接，返回缓存工具列表")
            if not tools:
                logger.warning("无缓存工具可用")
            return tools
        
        try:
            async with self._session.post(
                urljoin(self._ws_url, "/api/tools/list"),
                headers={
                    "Authorization": f"Bearer {self.token}" if self.token else "",
                    "Content-Type": "application/json"
                },
                json={
                    "client_name": self.client_name,
                    "include_schema": True
                },
                timeout=self.timeout
            ) as resp:
                if resp.status == 200:
                    tools = await resp.json()
                    
                    # 更新缓存
                    self._tools = tools
                    self._tools_last_fetch = datetime.now()
                    
                    logger.info(f"成功获取工具列表，共 {len(tools)} 个工具")
                    return tools if tools else {}
                else:
                    error_data = await resp.text()
                    raise MCPToolError(
                        f"获取工具列表失败：HTTP {resp.status}",
                        details={"response": error_data}
                    )
        except Exception as e:
            tools = self._get_cached_tools()
            logger.warning(f"获取工具列表失败，使用缓存：{e}")
            return tools if tools else {}

    def _get_cached_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取工具缓存"""
        if self._tools and self._tools_last_fetch:
            ttl = (datetime.now() - self._tools_last_fetch).total_seconds()
            if ttl < self._TOOLS_CACHE_TTL:
                logger.debug(f"使用缓存工具列表（延迟 {ttl:.1f}s）")
                return self._tools.copy()
            else:
                self._tools = {}
                self._tools_last_fetch = None
        
        return self._tools.copy()

    # ==================== 工具调用 ====================

    async def call_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            timeout: 超时时间（秒），默认使用实例超时
        
        Returns:
            工具返回结果
        
        Raises:
            MCPToolError: 工具调用失败
        """
        effective_timeout = timeout or self.timeout
        
        logger.info(f"调用工具：{tool_name} / 参数：{json.dumps(params)}")
        
        for attempt in range(self._max_retries):
            try:
                result = await self._invoke_tool_internal(tool_name, params, effective_timeout)
                return result
            except MCPRetryExhausted:
                raise
            except MCPException as e:
                # 非可重试错误直接抛出
                if isinstance(e, (MCPConnectionError, MCPParseError, MCPToolError)):
                    raise
                logger.warning(f"第 {attempt + 1}/{self._max_retries} 次调用失败：{e}")
                if attempt < self._max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"等待 {delay:.1f}秒后重试...")
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.warning(f"第 {attempt + 1}/{self._max_retries} 次调用异常：{e}")
                if attempt < self._max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise MCPRetryExhausted(
                        f"工具调用 '{tool_name}' 重试耗尽",
                        details={"attempt": attempt + 1, "error": str(e)}
                    )

        raise MCPRetryExhausted(
            f"工具调用 '{tool_name}' 重试耗尽",
            details={"attempt": self._max_retries}
        )

    async def _invoke_tool_internal(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """内部工具调用方法"""
        
        # 1. 如果没有工具列表，先获取
        if not self.tools:
            tools = await self.list_tools()
            if tools and tool_name in tools:
                self._tools = tools
                self._tools_last_fetch = datetime.now()
            else:
                raise MCPToolError(f"未知工具：{tool_name}")
        
        # 2. 检查工具是否存在
        if tool_name not in self.tools:
            raise MCPToolError(f"未知工具：{tool_name}")
        
        tool_info = self.tools[tool_name]
        
        # 3. 根据需要发送不同类型的请求
        if tool_name == "baidu_search":
            return await self._execute_baidu_search(params, timeout)
        elif tool_name == "serp_features":
            return await self._execute_serp_features(params, timeout)
        elif tool_name == "knowledge_graph":
            return await self._execute_knowledge_graph(params, timeout)
        else:
            # 默认使用通用工具执行
            return await self._execute_generic_tool(tool_name, params, timeout)

    async def _execute_baidu_search(self, params: Dict, timeout: int) -> Dict[str, Any]:
        """执行百度搜索"""
        payload = {
            "tool": "baidu_search",
            "query": params.get("query", ""),
            "count": params.get("count", 10),
            "context": params.get("context", None)
        }
        
        async with self._session.post(
            urljoin(self._ws_url, "/api/baidu/s"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}" if self.token else ""
            },
            json=payload,
            timeout=timeout
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "tool_name": "baidu_search",
                    "results": data.get("results", []),
                    "total": data.get("total", 0),
                    "query": data.get("query", ""),
                    "success": True
                }
            raise MCPToolError(f"百度搜索失败：HTTP {resp.status}")

    async def _execute_serp_features(self, params: Dict, timeout: int) -> Dict[str, Any]:
        """执行 SERP 分析"""
        payload = {
            "tool": "serp_features",
            "query": params.get("query", ""),
            "deep": params.get("depth", 1)
        }
        
        async with self._session.post(
            urljoin(self._ws_url, "/api/serp"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}" if self.token else ""
            },
            json=payload,
            timeout=timeout
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            raise MCPToolError(f"SERP 分析失败：HTTP {resp.status}")

    async def _execute_knowledge_graph(self, params: Dict, timeout: int) -> Dict[str, Any]:
        """执行知识图谱查询"""
        payload = {
            "tool": "knowledge_graph",
            "query": params.get("query", ""),
            "lang": params.get("lang", "zh")
        }
        
        async with self._session.post(
            urljoin(self._ws_url, "/api/knowledge"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}" if self.token else ""
            },
            json=payload,
            timeout=timeout
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            raise MCPToolError(f"知识图谱查询失败：HTTP {resp.status}")

    async def _execute_generic_tool(self, tool_name: str, params: Dict, timeout: int) -> Dict[str, Any]:
        """通用工具执行"""
        # 使用 MCP 标准工具调用流程
        payload = {
            "tool": tool_name,
            "params": params
        }
        
        async with self._session.post(
            urljoin(self._ws_url, "/api/tools/execute"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}" if self.token else ""
            },
            json=payload,
            timeout=timeout
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            raise MCPToolError(f"工具执行失败：HTTP {resp.status}")

    # ==================== 消息底层操作 ====================

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """发送消息到 WebSocket"""
        if self._ws is None or self._ws.closed:
            raise MCPConnectionError("WebSocket 未连接")
        
        self._message_id_counter += 1
        if "id" not in message:
            message["id"] = self._message_id_counter
        
        message["timestamp"] = datetime.now().isoformat()
        
        self._message_history.append({
            "direction": "outbound",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        await self._ws.send_str(json.dumps(message))
        # 不等待响应

    async def _receive_response(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        从 WebSocket 接收并解析响应
        
        Args:
            timeout: 接收超时时间（秒）
        
        Returns:
            响应字典
        
        Raises:
            MCPConnectionError: 连接错误
            MCPParseError: 解析错误
        """
        effective_timeout = timeout or self.timeout
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=effective_timeout)
        
        try:
            if self._ws is None or self._ws.closed:
                raise MCPConnectionError("WebSocket 未连接")
            
            message = await asyncio.wait_for(
                self._ws.receive(),
                timeout=ws_timeout
            )
            
            if message.type == aiohttp.WSMsgType.CLOSED:
                raise MCPConnectionError("WebSocket 已关闭")
            
            data = json.loads(message.data)
            self._message_history.append({
                "direction": "inbound",
                "message": data,
                "timestamp": datetime.now().isoformat()
            })
            
            return data
            
        except asyncio.TimeoutError:
            raise MCPParseError("接收响应超时")
        except aiohttp.WSMsgType_CLOSE:
            raise MCPConnectionError("WebSocket 连接关闭")
        except json.JSONDecodeError as e:
            raise MCPParseError(f"JSON 解析失败：{e}")

    # ==================== 工具快捷方法 ====================

    async def search(
        self,
        query: str,
        count: int = 10,
        use_serp: bool = True
    ) -> List[Dict[str, Any]]:
        """
        快速搜索接口
        
        Args:
            query: 搜索关键词
            count: 返回结果数量
            use_serp: 是否启用 SERP 分析
        
        Returns:
            搜索结果列表
        """
        results = await self.call_tool("baidu_search", {
            "query": query,
            "count": count
        })
        
        if use_serp and self.is_connected:
            serp_data = await self.call_tool("serp_features", {
                "query": query,
                "depth": 1
            })
            results["serp_info"] = ser data
            
        return results.get("results", [])

    # ==================== 实用方法 ====================

    async def close(self) -> None:
        """优雅关闭客户端"""
        await self.disconnect()
        
        self._session = None
        self._ws = None
        logger.info(f"客户端 '{self.client_name}' 已关闭")

    def get_message_history(self) -> List[Dict[str, Any]]:
        """获取消息历史"""
        return self._message_history.copy()

    def clear_history(self) -> None:
        """清除消息历史"""
        self._message_history.clear()


# ==================== 工厂函数 ====================

def create_baidu_search_mcp(
    url: str,
    token: Optional[str] = None,
    protocol: str = "wss"
) -> BaiduSearchMCP:
    """
    创建百度 MCP 客户端的工厂函数
    
    Args:
        url: MCP 服务地址
        token: OAuth 令牌
        protocol: 协议类型
        
    Returns:
        BaiduSearchMCP 实例
    """
    return BaiduSearchMCP(
        url=url,
        token=token,
        protocol=protocol
    )


# ==================== 辅助函数 ====================

async def ensure_connected(client: BaiduSearchMCP) -> BaiduSearchMCP:
    """
    确保客户端已连接
    
    Args:
        client: MCP 客户端实例
        
    Returns:
        已连接的客户端实例
        
    Raises:
        MCPConnectionError: 连接失败
    """
    if not client.is_connected:
        await client.connect()
    return client


async def safe_call_tool(
    client: BaiduSearchMCP,
    tool_name: str,
    params: Dict[str, Any],
    max_retries: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    安全调用工具（带返回 None 选项）
    
    Args:
        client: MCP 客户端实例
        tool_name: 工具名称
        params: 工具参数
        max_retries: 最大重试次数
        
    Returns:
        工具结果，失败时返回 None
    """
    from search_agent.core.mcp.baidu_search_mcp import MCPRetryExhausted
    
    try:
        return await client.call_tool(tool_name, params, max_retries)
    except (MCPException, MCPRetryExhausted):
        return None
