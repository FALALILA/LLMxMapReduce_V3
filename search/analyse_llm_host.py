#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Interface with LLM Host

使用LLMHost进行智能任务处理的分析接口
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from request.wrapper import RequestWrapper
from .llm_host import LLMHost
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyseLLMHostInterface:
    """
    使用LLMHost的分析接口
    
    提供智能任务处理能力，让LLM自主选择和调用工具
    """
    
    def __init__(self,
                 base_dir: str = "new/test",
                 config_path: Optional[str] = None):

        self.base_dir = Path(base_dir)
        self.config_path = config_path
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = self.base_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        # 首先加载环境配置
        self._load_environment_config()

        # 确保env_config已设置
        # if not hasattr(self, 'env_config') or self.env_config is None:
        #     logger.error("env_config not properly loaded, using defaults")
        #     self.env_config = {
        #         "models": {"default_model": "gemini-2.5-flash", "default_infer_type": "OpenAI"},
        #         "analyse_settings": {"max_interaction_rounds": 3, "max_context_messages": 10}
        #     }

        # 从配置中获取参数，确保有默认值
        try:
            self.max_interaction_rounds = self.env_config.get("analyse_settings", {}).get("max_interaction_rounds", 3)
            self.llm_model = self.env_config.get("models", {}).get("default_model", "gemini-2.5-flash")
            self.llm_infer_type = self.env_config.get("models", {}).get("default_infer_type", "OpenAI")
        except Exception as e:
            logger.error(f"Failed to load environment config: {e}")
            # 设置默认值
            self.max_interaction_rounds = 3
            self.llm_model = "gemini-2.5-flash"
            self.llm_infer_type = "OpenAI"

        # 初始化记忆系统
        self.conversation_history = []  # 存储完整的对话历史

        # 初始化LLM宿主
        self.llm_host = LLMHost()

        # 初始化logger（修复Windows编码问题）
        self.logger = logging.getLogger(__name__)

        # 确保logger使用UTF-8编码，避免Windows GBK编码问题
        for handler in logging.root.handlers:
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass

# Memory功能已移除

        # 初始化组件（在加载配置之后）
        self._init_llm_components()

        self._load_config()

# _clear_memory_on_init方法已移除

    async def cleanup(self):
        """
        清理资源，关闭连接
        """
        try:
            if hasattr(self, 'llm_host') and self.llm_host:
                try:
                    await self.llm_host.disconnect()
                except Exception as disconnect_error:
                    # Log the error but don't let it stop the cleanup process
                    self.logger.warning(f"LLM host disconnect had issues (continuing cleanup): {disconnect_error}")
                finally:
                    # Always clear the host reference
                    self.llm_host = None
            self.logger.info("资源清理完成")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
            # Force cleanup
            if hasattr(self, 'llm_host'):
                self.llm_host = None

    def _load_environment_config(self):
        """加载环境配置文件"""
        try:
            # 尝试多个可能的统一配置文件路径
            config_paths = [
                "new/config/unified_config.json",
                "config/unified_config.json",
                os.path.join(os.path.dirname(__file__), "..", "..", "config", "unified_config.json")
            ]

            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.env_config = json.load(f)
                    logger.info(f"Environment config loaded from: {config_path}")

                    # 设置环境变量以确保API key正确传递
                    self._set_environment_variables()
                    return

            # 如果没有找到配置文件，使用默认配置
            logger.warning("Environment config file not found, using defaults")
            self.env_config = {
                "models": {
                    "default_model": "gemini-2.0-flash",
                    "default_infer_type": "OpenAI"
                },
                "analyse_settings": {
                    "max_interaction_rounds": 3,
                    "max_context_messages": 10,
                    "web_search_timeout": 30000.0,
                    "crawl_urls_timeout": 30000.0,
                    "mcp_tool_timeout": 30000.0
                }
            }
        except Exception as e:
            logger.error(f"Failed to load environment config: {e}")
            # 使用默认配置确保所有必需的字段都存在
            self.env_config = {
                "models": {
                    "default_model": "gemini-2.5-flash",
                    "default_infer_type": "OpenAI"
                },
                "analyse_settings": {
                    "max_interaction_rounds": 3,
                    "max_context_messages": 10,
                    "web_search_timeout": 30000.0,
                    "crawl_urls_timeout": 30000.0,
                    "mcp_tool_timeout": 30000.0
                }
            }

    def _set_environment_variables(self):
        """设置环境变量，确保API key正确传递"""
        try:
            api_keys = self.env_config.get("api_keys", {})

            # 设置OpenAI API配置
            openai_config = api_keys.get("openai", {})
            if openai_config.get("api_key"):
                os.environ["OPENAI_API_KEY"] = openai_config["api_key"]
                logger.info("✅ OPENAI_API_KEY 已设置")
            if openai_config.get("base_url"):
                os.environ["OPENAI_BASE_URL"] = openai_config["base_url"]
                logger.info("✅ OPENAI_BASE_URL 已设置")

            # 设置搜索引擎API密钥
            search_engines = api_keys.get("search_engines", {})
            if search_engines.get("serpapi_key"):
                os.environ["SERPAPI_KEY"] = search_engines["serpapi_key"]
                logger.info(f"✅ SERPAPI_KEY 已设置: {search_engines['serpapi_key'][:10]}...")

            if search_engines.get("bing_subscription_key"):
                os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = search_engines["bing_subscription_key"]
                logger.info("✅ BING_SEARCH_V7_SUBSCRIPTION_KEY 已设置")

        except Exception as e:
            logger.error(f"Failed to set environment variables: {e}")

        logger.info(f"AnalyseLLMHostInterface initialized:")
        logger.info(f"  - Base directory: {self.base_dir}")
        logger.info(f"  - Max interaction rounds: {self.max_interaction_rounds}")
        logger.info(f"  - LLM model: {self.llm_model}")
        logger.info(f"  - Using LLMHost for intelligent task processing")

    def _init_llm_components(self):
        try:
            self.llm_wrapper = RequestWrapper(
                model=self.llm_model,
                infer_type=self.llm_infer_type
            )
            logger.info("LLM wrapper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM wrapper: {e}")
            raise

    async def analyse(self, topic: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        执行智能任务分析 - 使用LLMHost进行智能工具选择
        
        Args:
            topic: 研究主题
            description: 主题描述
            
        Returns:
            分析结果
        """
        logger.info(f"Starting intelligent task analysis for topic: '{topic}'")

        # 记忆初始化
        self.conversation_history.clear()
        logger.info("=== 记忆初始化完成 ===")

        try:
            # 第一轮交互：主题扩写
            logger.info("=== 第一轮交互：主题扩写 ===")
            user_msg_1 = f"请扩写主题：{topic}。原始描述：{description or '无'}"
            expanded_topic = await self._llm_interaction_round_1(user_msg_1)
            logger.info(f"主题扩写完成: {expanded_topic[:100]}...")

            # 第二轮：使用LLMHost进行智能任务处理
            logger.info("=== 第二轮：智能任务处理 ===")
            
            # 构建任务描述
            task_description = f"执行文献搜索任务：{topic}"
            context = f"扩写后的主题描述：{expanded_topic}"

            # 使用LLMHost执行智能任务处理
            result = await self.llm_host.process_task(task_description, context)

            logger.info(f"✅ Intelligent analysis completed successfully")
            logger.info(f"Status: {result.get('status', 'unknown')}")
            logger.info(f"Rounds used: {result.get('rounds_used', 0)}")

            return result

        except Exception as e:
            logger.error(f"Error in intelligent task analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _llm_interaction_round_1(self, user_message: str) -> str:
        """第一轮交互：主题扩写"""
        # 添加用户消息到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 构建系统提示
        system_prompt = """你是一个专业的学术研究分析专家。用户会给你一个研究主题，请你扩写这个主题，提供详细的研究描述。

请从多个角度进行分析，生成一段专业且全面的描述，这个描述将用于后续的文献搜索。

只返回扩写后的主题描述，不要其他内容。"""

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.conversation_history

        # 调用LLM
        response = await self.llm_wrapper.async_request(messages)

        # 添加assistant响应到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _load_config(self):
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "search": {
                        "default_top_n": 20,
                        "default_engine": "google",
                        "similarity_threshold": 80
                    },
                    "interaction": {
                        "timeout_seconds": 0,  # 无超时限制
                        "auto_continue": False
                    }
                }
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            self.config = {}


# 便捷函数
async def analyse_with_llm_host(task: str, description: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷的智能分析函数
    
    Args:
        task: 研究主题
        description: 主题描述
        
    Returns:
        分析结果
    """
    analyser = AnalyseLLMHostInterface()
    try:
        return await analyser.analyse(task, description)
    finally:
        await analyser.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyse_llm_host.py <topic> [description]")
        sys.exit(1)

    topic = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"🤖 开始智能分析主题: {topic}")
    if description:
        print(f"📝 描述: {description}")
    print("-" * 50)

    try:
        import asyncio
        analysis_result = asyncio.run(analyse_with_llm_host(topic, description))
        print(f"\n✅ 智能分析完成！")
        print(f"状态: {analysis_result.get('status', 'unknown')}")
        print(f"使用轮数: {analysis_result.get('rounds_used', 0)}")
        print(f"结果: {analysis_result.get('result', 'No result')}")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
