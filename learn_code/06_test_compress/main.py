#!/usr/bin/env python3
"""
LangGraph 四则运算 Agent - 主程序入口

本文件演示如何使用 LangGraph 创建和运行一个能够执行四则运算的智能 Agent。

运行方式：
    python main.py

作者：Charles
日期：2024
"""

import os
from calculator_agent import (
    create_calculator_agent,
    add, subtract, multiply, divide,
    print_tool_info
)

# =============================================================================
# 配置部分
# =============================================================================

# 设置 OpenAI API Key（如果使用 OpenAI 模型）
# 建议从环境变量读取，避免硬编码
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# 模型配置
MODEL_NAME = "gpt-4o-mini"  # 可以替换为其他模型


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主函数：创建并测试四则运算 Agent。
    """
    print("=" * 60)
    print("LangGraph 四则运算 Agent 示例")
    print("=" * 60)
    
    # 显示可用工具信息
    print_tool_info()
    
    # ========================================================================
    # 方式一：使用 OpenAI 模型（需要 API Key）
    # ========================================================================
    try:
        from langchain_openai import ChatOpenAI
        
        print("\n" + "=" * 60)
        print("使用 OpenAI 模型创建 Agent...")
        print("=" * 60)
        
        # 创建 LLM
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,  # 设置为 0 以获得更确定的输出
        )
        
        # 创建 Agent
        agent = create_calculator_agent(llm)
        
        # 测试用例
        test_cases = [
            "计算 25 + 17",
            "100 减去 38 等于多少？",
            "7 乘以 8 是多少？",
            "144 除以 12 的结果",
            "先算 10 + 5，然后乘以 2",
            "计算 (15 - 3) * 4",
        ]
        
        print("\n" + "=" * 60)
        print("开始测试 Agent...")
        print("=" * 60)
        
        for i, query in enumerate(test_cases, 1):
            print(f"\n【测试 {i}】用户输入：{query}")
            print("-" * 40)
            
            # 调用 Agent
            response = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # 显示结果
            last_message = response["messages"][-1]
            print(f"Agent 回复：{last_message.content}")
            print("-" * 40)
        
        print("\n" + "=" * 60)
        print("OpenAI 模型测试完成！")
        print("=" * 60)
        
    except ImportError:
        print("\n[警告] 未安装 langchain-openai，跳过 OpenAI 模型测试")
        print("请运行：pip install langchain-openai")
    except Exception as e:
        print(f"\n[错误] OpenAI 模型测试失败：{e}")
        print("请检查 API Key 是否正确配置")
    
    # ========================================================================
    # 方式二：使用本地模型（如 Ollama）
    # ========================================================================
    try:
        from langchain_ollama import ChatOllama
        
        print("\n" + "=" * 60)
        print("使用 Ollama 本地模型创建 Agent...")
        print("=" * 60)
        
        # 创建本地 LLM
        local_llm = ChatOllama(
            model="llama3.2",  # 可以替换为你安装的模型
            temperature=0,
        )
        
        # 创建 Agent
        local_agent = create_calculator_agent(local_llm)
        
        # 简单测试
        test_query = "计算 42 + 58"
        print(f"\n用户输入：{test_query}")
        print("-" * 40)
        
        response = local_agent.invoke({
            "messages": [{"role": "user", "content": test_query}]
        })
        
        last_message = response["messages"][-1]
        print(f"Agent 回复：{last_message.content}")
        print("-" * 40)
        
        print("\n" + "=" * 60)
        print("本地模型测试完成！")
        print("=" * 60)
        
    except ImportError:
        print("\n[提示] 未安装 langchain-ollama，跳过本地模型测试")
        print("请运行：pip install langchain-ollama")
    except Exception as e:
        print(f"\n[错误] 本地模型测试失败：{e}")
        print("请确保已安装 Ollama 并下载了相应模型")
    
    # ========================================================================
    # 交互式模式
    # ========================================================================
    print("\n" + "=" * 60)
    print("交互式模式（输入 'quit' 或 'exit' 退出）")
    print("=" * 60)
    
    try:
        # 使用第一个可用的 LLM
        if 'agent' in locals():
            current_agent = agent
        elif 'local_agent' in locals():
            current_agent = local_agent
        else:
            print("[错误] 没有可用的 Agent，请检查配置")
            return
        
        while True:
            user_input = input("\n请输入计算问题：").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not user_input:
                continue
            
            # 调用 Agent
            response = current_agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            
            # 显示结果
            last_message = response["messages"][-1]
            print(f"结果：{last_message.content}")
            print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n\n程序中断，再见！")
    except Exception as e:
        print(f"\n[错误] {e}")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
