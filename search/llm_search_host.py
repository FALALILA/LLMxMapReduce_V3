#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Search Host Layer

这是MCP架构中的Host层，负责：
1. 提供高级业务接口
2. 协调搜索流程
3. 调用Client层与MCP服务器通信
4. 不暴露底层MCP工具实现细节
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .llm_search_mcp_client import create_mcp_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMSearchHost:
    """
    LLM搜索宿主层
    
    提供高级业务接口，隐藏MCP实现细节
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.mcp_client = None
        self._connected = False
        self.env_config = self._load_environment_config()
        
    def _load_environment_config(self):
        """加载环境配置文件"""
        try:
            config_paths = [
                "new/config/environment_config.json",
                "config/environment_config.json",
                os.path.join(os.path.dirname(__file__), "..", "..", "config", "environment_config.json")
            ]

            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)

            logger.warning("Environment config file not found, using defaults")
            return {
                "models": {"default_model": "gemini-2.5-flash", "default_infer_type": "OpenAI"},
                "search_settings": {"default_query_count": 30, "default_total_urls": 200, "default_top_n": 70}
            }
        except Exception as e:
            logger.error(f"Failed to load environment config: {e}")
            return {}
    
    async def connect(self):
        """连接到MCP服务器"""
        if self._connected:
            return
            
        try:
            # 准备MCP服务器配置
            server_config = self._prepare_server_config()
            
            # 创建MCP客户端
            self.mcp_client = await create_mcp_client(server_config)
            self._connected = True
            
            logger.info("Successfully connected to LLM Search MCP Server")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    def _prepare_server_config(self) -> Dict[str, Any]:
        """准备MCP服务器配置"""
        # 只设置必要的环境变量，避免系统环境变量冲突
        # 但保留一些关键的系统环境变量以确保Python正常运行
        env_vars = {
            "PYTHONPATH": ".",
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": os.environ.get("TEMP", ""),
            "TMP": os.environ.get("TMP", ""),
        }

        # 从配置文件设置API密钥
        api_keys = self.env_config.get("api_keys", {})

        openai_config = api_keys.get("openai", {})
        if openai_config.get("api_key"):
            env_vars["OPENAI_API_KEY"] = openai_config["api_key"]
            logger.info("✅ OPENAI_API_KEY 已传递给MCP服务器")
        if openai_config.get("base_url"):
            env_vars["OPENAI_BASE_URL"] = openai_config["base_url"]
            env_vars["OPENAI_API_BASE"] = openai_config["base_url"]  # 兼容性
            logger.info("✅ OPENAI_BASE_URL 已传递给MCP服务器")

        search_engines = api_keys.get("search_engines", {})
        if search_engines.get("serpapi_key"):
            env_vars["SERPAPI_KEY"] = search_engines["serpapi_key"]
            env_vars["SERP_API_KEY"] = search_engines["serpapi_key"]  # 兼容性
            logger.info(f"✅ SERPAPI_KEY 已传递给MCP服务器: {search_engines['serpapi_key'][:10]}...")
        if search_engines.get("bing_subscription_key"):
            env_vars["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = search_engines["bing_subscription_key"]
            logger.info("✅ BING_SEARCH_V7_SUBSCRIPTION_KEY 已传递给MCP服务器")

        # 添加Google API Key（如果有的话）
        if openai_config.get("api_key"):
            env_vars["GOOGLE_API_KEY"] = openai_config["api_key"]

        # 验证关键环境变量
        logger.info(f"🔍 环境变量验证:")
        logger.info(f"  - SERPAPI_KEY: {'已设置' if env_vars.get('SERPAPI_KEY') else '未设置'}")
        logger.info(f"  - OPENAI_API_KEY: {'已设置' if env_vars.get('OPENAI_API_KEY') else '未设置'}")
        logger.info(f"  - 环境变量总数: {len(env_vars)}")

        # 获取项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent

        # 使用模块导入方式启动，与配置文件保持一致
        return {
            "command": "python",
            "args": ["-m", "src.search.llm_search_mcp_server"],
            "env": env_vars,
            "cwd": str(project_root)
        }
    
    async def disconnect(self):
        """断开MCP连接"""
        if self.mcp_client:
            await self.mcp_client.disconnect()
            self.mcp_client = None
        self._connected = False
        logger.info("Disconnected from MCP server")
    
    async def search_literature(self, topic: str, description: str = "", top_n: int = 20) -> List[Dict[str, Any]]:
        """
        执行完整的文献搜索流程
        
        Args:
            topic: 研究主题
            description: 主题描述
            top_n: 返回结果数量
            
        Returns:
            文献搜索结果列表
        """
        if not self._connected:
            await self.connect()
            
        try:
            logger.info(f"Starting literature search for topic: '{topic}'")
            
            # 从配置获取默认参数
            search_settings = self.env_config.get("search_settings", {})
            query_count = search_settings.get("default_query_count", 30)
            
            # 步骤1: 生成搜索查询
            logger.info("Step 1: Generating search queries")
            query_result = await self.mcp_client.call_tool(
                "generate_search_queries",
                {
                    "topic": topic,
                    "description": description
                }
            )
            
            if not query_result or "query_file_path" not in query_result:
                raise ValueError("Failed to generate search queries")

            query_file_path = query_result["query_file_path"]
            query_count = query_result.get("query_count", 0)
            logger.info(f"Generated {query_count} search queries, saved to: {query_file_path}")

            # 步骤2: 执行网络搜索
            logger.info("Step 2: Performing web search")
            search_result = await self.mcp_client.call_tool(
                "web_search",
                {
                    "query_file_path": query_file_path,  # 传递文件路径而不是查询列表
                    "topic": topic,
                    "top_n": search_settings.get("default_total_urls", 200)
                }
            )
            
            if not search_result or "urls" not in search_result:
                raise ValueError("Failed to perform web search")
                
            urls = search_result["urls"]
            logger.info(f"Found {len(urls)} URLs from web search")
            
            # 步骤3: 爬取和分析内容
            logger.info("Step 3: Crawling and analyzing content")
            # 获取URL文件路径（从web_search结果中）
            url_file_path = search_result.get("url_file_path")
            crawl_result = await self.mcp_client.call_tool(
                "crawl_urls",
                {
                    "topic": topic,
                    "url_file_path": url_file_path,  # 传递文件路径而不是URL列表
                    "top_n": top_n,
                    "similarity_threshold": search_settings.get("default_similarity_threshold", 30)
                }
            )
            
            if not crawl_result:
                raise ValueError("Failed to crawl URLs")
            
            # 提取最终结果
            final_results = crawl_result.get("final_results", [])
            logger.info(f"Successfully retrieved {len(final_results)} literature papers")
            
            # 保存结果
            self._save_results(final_results, topic)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in literature search: {e}")
            raise
    
    def _save_results(self, results: List[Dict[str, Any]], topic: str) -> str:
        """保存搜索结果到文件"""
        try:
            # 创建保存目录
            save_dir = Path(__file__).parent.parent.parent / "test"
            save_dir.mkdir(exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')[:50]
            filename = f"literature_results_{safe_topic}_{timestamp}.json"
            filepath = save_dir / filename
            
            # 保存数据
            save_data = {
                "topic": topic,
                "timestamp": timestamp,
                "total_results": len(results),
                "results": results,
                "metadata": {
                    "generated_by": "llm_search_host.py",
                    "version": "1.0",
                    "description": f"Literature search results for topic: {topic}"
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# 便捷函数
async def search_literature(topic: str, description: str = "", top_n: int = 20) -> List[Dict[str, Any]]:
    """
    便捷的文献搜索函数
    
    Args:
        topic: 研究主题
        description: 主题描述
        top_n: 返回结果数量
        
    Returns:
        文献搜索结果列表
    """
    async with LLMSearchHost() as host:
        return await host.search_literature(topic, description, top_n)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_search_host.py <topic> [description] [top_n]")
        sys.exit(1)
    
    topic = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else ""
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    print(f"🔍 Starting literature search for: {topic}")
    if description:
        print(f"📝 Description: {description}")
    print(f"🎯 Target papers: {top_n}")
    print("-" * 50)
    
    try:
        results = asyncio.run(search_literature(topic, description, top_n))
        print(f"\n✅ Search completed! Found {len(results)} papers")
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
