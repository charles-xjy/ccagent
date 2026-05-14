"""
Baidu Search Agent - Main Entry Point
智能搜索代理应用入口
"""

import asyncio
from dotenv import load_dotenv
from loguru import logger
from config.settings import Settings
from api.routes import app
import uvicorn

# Load environment variables
load_dotenv()

async def main():
    """Application startup"""
    settings = Settings()
    
    logger.info("=" * 50)
    logger.info("Baidu Search Agent Starting...")
    logger.info(f"Search Timeout: {settings.search_timeout}s")
    logger.info("=" * 50)
    
    logger.info("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    asyncio.run(main())
