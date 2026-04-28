"""
Calculator Agent 测试文件

本文件包含对 Calculator Agent 的完整测试，包括：
1. 工具函数测试 - 测试四则运算函数的正确性
2. Agent 结构测试 - 测试 LangGraph 图的节点和边
3. 集成测试 - 测试完整的 Agent 流程（需要配置 LLM）

作者：Charles
日期：2024
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import List

# 导入被测试的模块
from calculator_agent import (
    add, subtract, multiply, divide,
    AgentState, create_calculator_agent
)

# 由于 @tool 装饰器将函数转换为 StructuredTool 对象，
# 需要通过 .func 属性访问原始函数
_add = add.func if hasattr(add, 'func') else add
_subtract = subtract.func if hasattr(subtract, 'func') else subtract
_multiply = multiply.func if hasattr(multiply, 'func') else multiply
_divide = divide.func if hasattr(divide, 'func') else divide


# =============================================================================
# 第一部分：工具函数测试
# =============================================================================

class TestCalculatorTools:
    """测试四则运算工具函数"""
    
    # -------------------------------------------------------------------------
    # 加法测试
    # -------------------------------------------------------------------------
    
    def test_add_positive_numbers(self):
        """测试正数加法"""
        assert _add(5, 3) == 8
        assert _add(10, 20) == 30
        assert _add(100, 200) == 300
    
    def test_add_negative_numbers(self):
        """测试负数加法"""
        assert _add(-5, -3) == -8
        assert _add(-10, -20) == -30
        assert _add(-100, -200) == -300
    
    def test_add_mixed_numbers(self):
        """测试正负数混合加法"""
        assert _add(-5, 10) == 5
        assert _add(5, -3) == 2
        assert _add(-10, 10) == 0
    
    def test_add_decimal_numbers(self):
        """测试小数加法"""
        assert _add(2.5, 3.5) == 6.0
        assert _add(1.1, 2.2) == pytest.approx(3.3)
        assert _add(-1.5, 2.5) == 1.0
    
    def test_add_zero(self):
        """测试与零的加法"""
        assert _add(5, 0) == 5
        assert _add(0, 5) == 5
        assert _add(0, 0) == 0
    
    # -------------------------------------------------------------------------
    # 减法测试
    # -------------------------------------------------------------------------
    
    def test_subtract_positive_numbers(self):
        """测试正数减法"""
        assert _subtract(10, 3) == 7
        assert _subtract(20, 5) == 15
        assert _subtract(100, 50) == 50
    
    def test_subtract_negative_numbers(self):
        """测试负数减法"""
        assert _subtract(-10, -3) == -7
        assert _subtract(-20, -5) == -15
        assert _subtract(-5, -10) == 5
    
    def test_subtract_mixed_numbers(self):
        """测试正负数混合减法"""
        assert _subtract(5, -3) == 8
        assert _subtract(-5, 3) == -8
        assert _subtract(10, -10) == 20
    
    def test_subtract_equal_numbers(self):
        """测试相等数相减"""
        assert _subtract(5, 5) == 0
        assert _subtract(-5, -5) == 0
        assert _subtract(0, 0) == 0
    
    # -------------------------------------------------------------------------
    # 乘法测试
    # -------------------------------------------------------------------------
    
    def test_multiply_positive_numbers(self):
        """测试正数乘法"""
        assert _multiply(4, 5) == 20
        assert _multiply(10, 10) == 100
        assert _multiply(3, 7) == 21
    
    def test_multiply_negative_numbers(self):
        """测试负数乘法"""
        assert _multiply(-3, 7) == -21
        assert _multiply(3, -7) == -21
        assert _multiply(-3, -7) == 21
    
    def test_multiply_with_zero(self):
        """测试与零的乘法"""
        assert _multiply(5, 0) == 0
        assert _multiply(0, 5) == 0
        assert _multiply(0, 0) == 0
    
    def test_multiply_decimal_numbers(self):
        """测试小数乘法"""
        assert _multiply(2.5, 4) == 10.0
        assert _multiply(1.5, 2.0) == 3.0
        assert _multiply(-2.5, 2.0) == -5.0
    
    # -------------------------------------------------------------------------
    # 除法测试
    # -------------------------------------------------------------------------
    
    def test_divide_positive_numbers(self):
        """测试正数除法"""
        assert _divide(20, 4) == 5.0
        assert _divide(100, 10) == 10.0
        assert _divide(15, 3) == 5.0
    
    def test_divide_negative_numbers(self):
        """测试负数除法"""
        assert _divide(-20, 4) == -5.0
        assert _divide(20, -4) == -5.0
        assert _divide(-20, -4) == 5.0
    
    def test_divide_decimal_numbers(self):
        """测试小数除法"""
        assert _divide(7, 2) == 3.5
        assert _divide(10, 4) == 2.5
        assert _divide(1.5, 0.5) == 3.0
    
    def test_divide_by_zero(self):
        """测试除数为零的情况 - 应抛出 ValueError"""
        with pytest.raises(ValueError, match="除数不能为 0"):
            _divide(10, 0)
        
        with pytest.raises(ValueError, match="除数不能为 0"):
            _divide(0, 0)
        
        with pytest.raises(ValueError, match="除数不能为 0"):
            _divide(-10, 0)
    
    def test_divide_result_decimal(self):
        """测试除法结果为小数的情况"""
        result = _divide(10, 3)
        assert result == pytest.approx(3.3333333333333335)
        
        result = _divide(1, 3)
        assert result == pytest.approx(0.3333333333333333)


# =============================================================================
# 第二部分：Agent 结构测试
# =============================================================================

class TestAgentStructure:
    """测试 Agent 的图结构"""
    
    @pytest.fixture
    def mock_llm(self):
        """创建模拟的 LLM 对象"""
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=Mock(
            content="测试回复",
            tool_calls=[]
        ))
        return mock_llm
    
    def test_agent_creation(self, mock_llm):
        """测试 Agent 创建成功"""
        agent = create_calculator_agent(mock_llm)
        assert agent is not None
        assert hasattr(agent, 'get_graph')
    
    def test_agent_has_required_nodes(self, mock_llm):
        """测试 Agent 图包含必需的节点"""
        agent = create_calculator_agent(mock_llm)
        graph = agent.get_graph()
        
        # 获取所有节点
        nodes = list(graph.nodes.keys())
        
        # 检查必需节点存在
        assert "agent" in nodes, "缺少 'agent' 节点"
        assert "tools" in nodes, "缺少 'tools' 节点"
    
    def test_agent_node_count(self, mock_llm):
        """测试 Agent 图的节点数量"""
        agent = create_calculator_agent(mock_llm)
        graph = agent.get_graph()
        
        nodes = list(graph.nodes.keys())
        # 应该有 4 个节点：__start__, agent, tools, __end__
        assert len(nodes) == 4, f"期望 4 个节点，实际有 {len(nodes)} 个"
        
        # 验证所有必需节点存在
        assert "__start__" in nodes
        assert "agent" in nodes
        assert "tools" in nodes
        assert "__end__" in nodes
    
    def test_agent_has_edges(self, mock_llm):
        """测试 Agent 图包含边"""
        agent = create_calculator_agent(mock_llm)
        graph = agent.get_graph()
        
        # 获取所有边
        edges = list(graph.edges)
        
        # 应该有边存在
        assert len(edges) > 0, "图应该至少包含一条边"
    
    def test_agent_entry_point(self, mock_llm):
        """测试 Agent 图的入口点"""
        agent = create_calculator_agent(mock_llm)
        graph = agent.get_graph()
        
        # 检查入口点
        entry_point = graph.nodes.get('__start__')
        assert entry_point is not None, "应该存在入口点"
    
    def test_agent_graph_compilation(self, mock_llm):
        """测试 Agent 图可以成功编译"""
        agent = create_calculator_agent(mock_llm)
        
        # 编译后的 agent 应该有 invoke 方法
        assert hasattr(agent, 'invoke'), "编译后的 agent 应该有 invoke 方法"
        assert hasattr(agent, 'get_graph'), "编译后的 agent 应该有 get_graph 方法"
    
    def test_agent_graph_visualization(self, mock_llm):
        """测试 Agent 图的可视化输出"""
        agent = create_calculator_agent(mock_llm)
        
        # 获取图对象
        graph = agent.get_graph()
        
        # 测试获取图的 Mermaid 表示
        mermaid = graph.draw_mermaid()
        assert isinstance(mermaid, str)
        assert len(mermaid) > 0, "图的 Mermaid 表示不应该为空"
        
        # 检查 Mermaid 中包含节点信息
        assert "agent" in mermaid.lower()
        assert "tools" in mermaid.lower()
    
    def test_agent_state_structure(self):
        """测试 Agent 状态结构"""
        # 创建状态实例
        state: AgentState = {
            "messages": []
        }
        
        assert "messages" in state
        assert isinstance(state["messages"], list)


# =============================================================================
# 第三部分：集成测试（需要配置 LLM）
# =============================================================================

class TestIntegration:
    """集成测试 - 测试完整的 Agent 流程"""
    
    @pytest.fixture
    def mock_llm_with_tools(self):
        """创建模拟的 LLM 对象，支持工具调用"""
        mock_llm = Mock()
        
        # 创建模拟的 LLM 响应
        mock_response = Mock()
        mock_response.content = "计算结果是 8"
        mock_response.tool_calls = []
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=mock_response)
        
        return mock_llm
    
    @pytest.fixture
    def mock_llm_with_tool_call(self):
        """创建模拟的 LLM 对象，会触发工具调用"""
        mock_llm = Mock()
        
        # 第一次调用：返回工具调用
        mock_tool_call = Mock()
        mock_tool_call.name = "add"
        mock_tool_call.args = {"a": 5, "b": 3}
        mock_tool_call.id = "call_123"
        
        mock_first_response = Mock()
        mock_first_response.content = ""
        mock_first_response.tool_calls = [mock_tool_call]
        
        # 第二次调用：返回最终结果
        mock_final_response = Mock()
        mock_final_response.content = "5 + 3 = 8"
        mock_final_response.tool_calls = []
        
        # 设置调用序列
        call_count = [0]
        def invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_first_response
            return mock_final_response
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=invoke_side_effect)
        
        return mock_llm
    
    def test_simple_addition(self, mock_llm_with_tools):
        """测试简单的加法运算"""
        agent = create_calculator_agent(mock_llm_with_tools)
        
        # 准备输入
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="5 + 3 = ?")]
        }
        
        # 执行
        result = agent.invoke(state)
        
        # 验证
        assert "messages" in result
        assert len(result["messages"]) > 0
    
    @pytest.mark.skip(reason="工具调用模拟需要更复杂的设置")
    def test_agent_with_tool_invocation(self, mock_llm_with_tool_call):
        """测试 Agent 工具调用流程（需要更复杂的模拟）"""
        agent = create_calculator_agent(mock_llm_with_tool_call)
        
        # 准备输入
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="计算 5 + 3")]
        }
        
        # 执行
        result = agent.invoke(state)
        
        # 验证 LLM 被调用了两次（一次工具调用，一次返回结果）
        assert mock_llm_with_tool_call.invoke.call_count == 2
    
    def test_division_by_zero_handling(self, mock_llm_with_tools):
        """测试除数为零的情况处理"""
        # 直接测试工具函数
        with pytest.raises(ValueError):
            _divide(10, 0)
    
    @pytest.mark.skip(reason="需要配置真实的 LLM 密钥")
    def test_real_llm_integration(self):
        """
        真实 LLM 集成测试（需要配置）
        
        要运行此测试，需要：
        1. 设置 OPENAI_API_KEY 环境变量
        2. 取消 @pytest.mark.skip 装饰器
        
        示例：
        export OPENAI_API_KEY="your-api-key"
        pytest test_calculator.py::TestIntegration::test_real_llm_integration -v
        """
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        agent = create_calculator_agent(llm)
        
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="计算 25 + 17")]
        }
        
        result = agent.invoke(state)
        
        assert "messages" in result
        assert len(result["messages"]) > 0


# =============================================================================
# 第四部分：性能测试（可选）
# =============================================================================

class TestPerformance:
    """性能测试"""
    
    def test_add_performance(self):
        """测试加法性能"""
        import time
        
        iterations = 10000
        start = time.time()
        
        for _ in range(iterations):
            _add(123.456, 789.012)
        
        elapsed = time.time() - start
        # 10000 次运算应该在 1 秒内完成
        assert elapsed < 1.0, f"加法性能测试失败：{elapsed:.4f}秒"
    
    def test_divide_performance(self):
        """测试除法性能"""
        import time
        
        iterations = 10000
        start = time.time()
        
        for _ in range(iterations):
            _divide(1234.567, 890.123)
        
        elapsed = time.time() - start
        # 10000 次运算应该在 1 秒内完成
        assert elapsed < 1.0, f"除法性能测试失败：{elapsed:.4f}秒"


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])
