"""
模板使用示例
流式响应模板
"""

class StreamResponseTemplate:
    """流式响应模板"""
    
    def stream_response(self, data: dict):
        """生成流式响应"""
        yield f"data: {data}\n\n"
