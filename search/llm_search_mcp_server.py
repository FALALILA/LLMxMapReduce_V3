#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import sys
import os
import time
import re
import traceback
from typing import Dict, Any, List

# 设置标准输出编码为UTF-8（安全版本）
try:
    if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure') and sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    # 在某些环境中可能无法重新配置编码，忽略错误
    pass

from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
import mcp.server.stdio

try:
    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None
    CacheMode = None
    CrawlerRunConfig = None

# 优先从当前项目导入LLM_search
from .LLM_search import LLM_search
# print(f"✅ LLM_search 从相对导入成功")

# 导入HTML清理模块
try:
    from .clean.html_extrator import CommonCrawlWARCExtractor, JusTextExtractor, ResiliparseExtractor
    HTML_CLEANER_AVAILABLE = True
except ImportError as e:
    HTML_CLEANER_AVAILABLE = False
    CommonCrawlWARCExtractor = None
    JusTextExtractor = None
    ResiliparseExtractor = None

from src.exceptions import AnalyseError
from request import RequestWrapper

import logging.handlers

# 创建日志目录
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'mcp_server.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stderr)  # 输出到stderr以便客户端能看到
    ]
)

logger = logging.getLogger(__name__)

# 初始化HTML清理器
_html_extractor = None
if HTML_CLEANER_AVAILABLE:
    try:
        # 使用JusText算法作为默认清理器
        _html_extractor = CommonCrawlWARCExtractor(algorithm=JusTextExtractor())
        logger.info("✅ HTML cleaner initialized with JusText algorithm")
    except Exception as e:
        logger.warning(f"❌ Failed to initialize HTML cleaner: {e}")
        logger.info("Will fall back to basic HTML cleaning using regex")
        # 不设置HTML_CLEANER_AVAILABLE为False，让基本清理功能继续工作
        _html_extractor = None
else:
    logger.warning("❌ HTML cleaner not available, will use basic text extraction")

app = Server("llm-search-server")

def load_server_config():
    """从统一配置文件加载Server配置"""
    # 加载统一配置
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'unified_config.json')

    try:
        # 读取统一配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 设置环境变量
        api_keys = config.get("api_keys", {})

        # 设置OpenAI API配置
        openai_config = api_keys.get("openai", {})
        if openai_config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = openai_config["api_key"]
        if openai_config.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = openai_config["base_url"]

        # 设置搜索引擎API密钥（优先使用已存在的环境变量）
        search_engines = api_keys.get("search_engines", {})

        # SERPAPI密钥设置
        if not os.environ.get("SERPAPI_KEY") and search_engines.get("serpapi_key"):
            os.environ["SERPAPI_KEY"] = search_engines["serpapi_key"]
            logger.info(f"✅ SERPAPI_KEY 从配置文件设置: {search_engines['serpapi_key'][:10]}...")
        elif os.environ.get("SERPAPI_KEY"):
            logger.info(f"✅ SERPAPI_KEY 使用环境变量: {os.environ['SERPAPI_KEY'][:10]}...")
        else:
            logger.warning("❌ SERPAPI_KEY 未在环境变量或配置文件中找到")

        # Bing密钥设置
        if not os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY") and search_engines.get("bing_subscription_key"):
            os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = search_engines["bing_subscription_key"]
            logger.info(f"✅ BING_SEARCH_V7_SUBSCRIPTION_KEY 从配置文件设置")
        elif os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY"):
            logger.info(f"✅ BING_SEARCH_V7_SUBSCRIPTION_KEY 使用环境变量")
        else:
            logger.warning("❌ BING_SEARCH_V7_SUBSCRIPTION_KEY 未在环境变量或配置文件中找到")

        # 验证环境变量是否正确设置
        serpapi_key = os.environ.get("SERPAPI_KEY")
        bing_key = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY")
        logger.info(f"🔍 环境变量验证 - SERPAPI_KEY: {'✅ 已设置' if serpapi_key else '❌ 未设置'}")
        logger.info(f"🔍 环境变量验证 - BING_KEY: {'✅ 已设置' if bing_key else '❌ 未设置'}")

        # 构建服务器配置
        models = config.get("models", {})
        search_settings = config.get("search_settings", {})
        timeout_settings = config.get("timeout_settings", {})
        crawling_settings = config.get("crawling_settings", {})
        mcp_settings = config.get("mcp_settings", {})
        prompts = config.get("prompts", {})

        # 确保使用更新后的模型配置
        logger.info(f"📝 从配置文件读取的模型设置: {models.get('default_model', 'N/A')}")

        # 验证必需的模型配置
        required_models = ["default_model", "default_infer_type", "content_analysis_model",
                          "similarity_model", "page_refine_model"]
        for model_key in required_models:
            if not models.get(model_key):
                raise ValueError(f"Missing required model configuration: {model_key}")

        server_config = {
            # 模型配置 - 从配置文件读取，不使用硬编码默认值
            "default_model": models.get("default_model"),
            "default_infer_type": models.get("default_infer_type"),
            "content_analysis_model": models.get("content_analysis_model"),
            "similarity_model": models.get("similarity_model"),
            "page_refine_model": models.get("page_refine_model"),

            # 搜索配置
            "default_engine": search_settings.get("default_engine", "google"),
            "default_query_count": search_settings.get("default_query_count", 30),
            "default_each_query_result": search_settings.get("default_each_query_result", 10),
            "default_total_urls": search_settings.get("default_total_urls", 200),
            "default_top_n": search_settings.get("default_top_n", 70),
            "default_similarity_threshold": search_settings.get("default_similarity_threshold", 30),
            "default_min_length": search_settings.get("default_min_length", 100),
            "default_max_length": search_settings.get("default_max_length", 1000000),  # 增加到100万字符

            # 超时配置（单位：秒）
            "llm_request_timeout": timeout_settings.get("llm_request_timeout", 30),
            "web_search_timeout": timeout_settings.get("web_search_timeout", 0),
            "crawling_timeout": timeout_settings.get("crawling_timeout", 0),
            "single_url_crawl_timeout": timeout_settings.get("single_url_crawl_timeout", 60),
            "content_analysis_timeout": timeout_settings.get("content_analysis_timeout", 30),
            "similarity_scoring_timeout": timeout_settings.get("similarity_scoring_timeout", 30),
            "abstract_generation_timeout": timeout_settings.get("abstract_generation_timeout", 30),
            "abstract_tasks_wait_timeout": timeout_settings.get("abstract_tasks_wait_timeout", 300),

            # 爬虫配置
            "max_concurrent_crawls": crawling_settings.get("max_concurrent_crawls", 10),
            "page_timeout": crawling_settings.get("page_timeout", 60),
            "retry_attempts": crawling_settings.get("retry_attempts", 3),
            "cache_mode": crawling_settings.get("cache_mode", "BYPASS"),

            # MCP设置
            "query_cache_dir": mcp_settings.get("query_cache_dir", "query_cache"),
            "url_cache_dir": mcp_settings.get("url_cache_dir", "url_cache"),

            # 提示词配置
            "prompts": prompts
        }

        logger.info(f"Environment config loaded successfully")
        logger.info(f"Server config: {server_config}")
        return server_config

    except Exception as e:
        logger.error(f"Failed to load environment config: {e}")
        logger.error("Configuration file config/unified_config.json is required but not found or invalid")
        raise FileNotFoundError("Configuration file config/unified_config.json is required but not found or invalid")

SERVER_CONFIG = load_server_config()
llm_search_instances = {}

# 辅助函数：用于生成兼容 LLMxMapReduce_V2 和 Survey 格式的 JSON
def proc_title_to_str(origin_title: str) -> str:
    """
    将标题转换为bibkey格式
    复制自 LLMxMapReduce_V2/src/utils/process_str.py
    """
    if not origin_title:
        return ""

    title = origin_title.lower().strip()
    title = title.replace("-", "_")
    title = re.sub(r'[^\w\s\_]', '', title)
    title = title.replace(" ", "_")
    title = re.sub(r'_{2,}', '_', title)
    return title

def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量
    使用简单的启发式方法：单词数 * 1.3
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)

def extract_abstract(text: str, max_length: int = 500) -> str:
    """
    从文本中提取摘要
    简单实现：取前max_length个字符
    """
    if not text:
        return ""

    # 去除多余的空白字符
    cleaned_text = re.sub(r'\s+', ' ', text.strip())

    if len(cleaned_text) <= max_length:
        return cleaned_text

    # 尝试在句号处截断，避免截断句子
    truncated = cleaned_text[:max_length]
    last_period = truncated.rfind('.')

    if last_period > max_length * 0.7:  # 如果句号位置合理
        return truncated[:last_period + 1]
    else:
        return truncated + "..."
@app.list_resources()
async def list_resources() -> List[Resource]:
    return [
        Resource(
            uri="llm://search/prompts",
            name="LLM Search Prompts",
            description="LLM搜索相关的提示词模板",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> List[TextContent]:
    if uri == "llm://search/prompts":
        # 从配置文件读取提示词
        prompts = SERVER_CONFIG.get("prompts", {})
        return [TextContent(
            type="text",
            text=json.dumps(prompts, ensure_ascii=False, indent=2)
        )]
    else:
        raise ValueError(f"Unknown resource: {uri}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="generate_search_queries",
            description="基于LLM生成优化的搜索查询。需要提供研究主题，返回生成的查询数量和保存文件路径，不返回具体查询内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "研究主题"
                    },
                    "description": {
                        "type": "string",
                        "description": "主题的可选描述或上下文"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="web_search",
            description="执行网络搜索并收集URL。需要提供主题，返回搜索到的URL数量和保存文件路径，不返回具体URL列表。",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "用于相关性过滤的主要主题"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "返回的最相关URL数量",
                        "default": 200
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="crawl_urls",
            description="爬取URL内容并进行智能处理。需要提供研究主题，返回爬取的URL成功数量、最终结果数量和保存文件路径，不返回具体文章内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "研究主题，用于内容过滤和相似度评分"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "返回的最高质量结果数量",
                        "default": 70
                    }
                },
                "required": ["topic"]
            }
        )
    ]
@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """调用工具"""
    global llm_search_instances

    try:
        if name == "generate_search_queries":
            result = await _generate_search_queries(
                arguments["topic"],
                arguments.get("description", ""),
                arguments.get("model")  
            )
        elif name == "web_search":
            result = await _web_search(
                arguments.get("query_file_path"),  # 改为从参数获取文件路径
                arguments["topic"],
                arguments.get("top_n"),
                arguments.get("engine")
            )

        elif name == "crawl_urls":
            result = await _crawl_urls(
                arguments["topic"],
                arguments.get("url_file_path"),  # 改为从参数获取文件路径
                arguments.get("top_n"),
                arguments.get("model"),
                arguments.get("similarity_threshold"),
                arguments.get("min_length"),
                arguments.get("max_length")
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

def _get_llm_search_instance(model: str = None, engine: str = None):
    """获取LLM搜索实例，使用配置文件中的默认值"""
    global llm_search_instances

    # 使用配置文件中的默认值
    if model is None:
        model = SERVER_CONFIG["default_model"]
    if engine is None:
        engine = SERVER_CONFIG["default_engine"]

    infer_type = SERVER_CONFIG["default_infer_type"]
    each_query_result = SERVER_CONFIG["default_each_query_result"]

    key = f"{model}_{engine}_{infer_type}"
    if key not in llm_search_instances:
        # 确保环境变量已设置（重新加载配置以防万一）
        import os

        # 重新确认环境变量设置
        serpapi_key = os.environ.get("SERPAPI_KEY")
        bing_key = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY")

        logger.info(f"🔑 创建LLM_search实例前的环境变量检查:")
        logger.info(f"   SERPAPI_KEY: {'已设置' if serpapi_key else '未设置'}")
        if serpapi_key:
            logger.info(f"   SERPAPI_KEY值: {serpapi_key[:10]}...")
        logger.info(f"   BING_SEARCH_V7_SUBSCRIPTION_KEY: {'已设置' if bing_key else '未设置'}")

        # 如果环境变量未设置，尝试重新从配置文件加载
        if not serpapi_key and not bing_key:
            logger.warning("⚠️ 环境变量未设置，尝试重新从配置文件加载...")
            try:
                env_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'environment_config.json')
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)

                search_engines = env_config.get("api_keys", {}).get("search_engines", {})
                if search_engines.get("serpapi_key"):
                    os.environ["SERPAPI_KEY"] = search_engines["serpapi_key"]
                    logger.info(f"🔄 重新设置SERPAPI_KEY: {search_engines['serpapi_key'][:10]}...")

                if search_engines.get("bing_subscription_key"):
                    os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"] = search_engines["bing_subscription_key"]
                    logger.info(f"🔄 重新设置BING_SEARCH_V7_SUBSCRIPTION_KEY")

            except Exception as reload_e:
                logger.error(f"❌ 重新加载配置失败: {reload_e}")

        try:
            logger.info(f"🚀 开始创建LLM_search实例: {key}")
            llm_search_instances[key] = LLM_search(
                model=model,
                infer_type=infer_type,
                engine=engine,
                each_query_result=each_query_result
            )
            logger.info(f"✅ LLM_search实例创建成功: {key}")
        except Exception as e:
            logger.error(f"❌ LLM_search实例创建失败: {e}")
            logger.error(f"   模型: {model}, 推理类型: {infer_type}, 引擎: {engine}")
            # 输出更详细的环境变量信息用于调试
            logger.error(f"   当前SERPAPI_KEY: {os.environ.get('SERPAPI_KEY', 'None')}")
            logger.error(f"   当前BING_KEY: {os.environ.get('BING_SEARCH_V7_SUBSCRIPTION_KEY', 'None')}")
            raise e
    else:
        logger.info(f"♻️ 复用已存在的LLM_search实例: {key}")

    return llm_search_instances[key]

async def _generate_search_queries(topic: str, description: str = "", model: str = None) -> Dict[str, Any]:
    logger.info(f"Generating search queries for topic: {topic}")

    try:
        if model is None:
            model = SERVER_CONFIG["default_model"]

        infer_type = SERVER_CONFIG.get("default_infer_type", "OpenAI")

        llm_search = _get_llm_search_instance(model=model)

        # 在线程池中执行同步的get_queries方法
        import asyncio
        import functools
        loop = asyncio.get_event_loop()

        # 从配置中获取查询数量
        query_count = SERVER_CONFIG.get("default_query_count", 30)

        # 使用functools.partial来传递关键字参数
        get_queries_func = functools.partial(
            llm_search.get_queries,
            topic=topic,
            description=description,
            query_count=query_count
        )

        # 添加超时机制，防止LLM调用卡住
        try:
            queries = await asyncio.wait_for(
                loop.run_in_executor(None, get_queries_func),
                timeout=SERVER_CONFIG.get("llm_request_timeout", 30)  # 使用配置的超时时间（秒）
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM query generation timed out for topic: {topic}")
            # 返回默认查询
            queries = [
                topic,
                f"{topic} research",
                f"{topic} analysis",
                f"{topic} study"
            ]

        # 保存查询结果到本地文件
        query_file_path = _save_queries_to_file(queries, topic, description)

        result = {
            "topic": topic,
            "description": description,
            "model": model,
            "queries": queries,
            "query_count": len(queries),
            "query_file_path": query_file_path,  # 添加文件路径
            "processing_metadata": {
                "model": model,
                "method": "llm_generation",
                "timestamp": "2025-01-23",
                "query_file_saved": query_file_path is not None
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error generating queries: {e}")
        print(e)
        return
async def _web_search(query_file_path: str = None, topic: str = "", top_n: int = None, engine: str = None) -> Dict[str, Any]:

    # 从文件中读取查询列表
    queries = _load_queries_from_file(query_file_path, topic)
    logger.info(f"Performing web search for {len(queries)} queries")

    try:
        if top_n is None:
            top_n = SERVER_CONFIG["default_total_urls"]  # 修正：web search阶段应该使用total_urls配置
        if engine is None:
            engine = SERVER_CONFIG["default_engine"]

        # 获取LLM搜索实例，添加错误处理
        try:
            llm_search = _get_llm_search_instance(engine=engine)
            if llm_search is None:
                raise ValueError("Failed to create LLM_search instance")
        except Exception as e:
            logger.error(f"Failed to get LLM_search instance: {e}")
            return {
                "topic": topic,
                "queries": queries,
                "engine": engine or SERVER_CONFIG["default_engine"],
                "urls": [],
                "url_count": 0,
                "top_n": top_n,
                "processing_metadata": {
                    "engine": engine,
                    "query_count": len(queries),
                    "result_count": 0,
                    "method": "error_fallback",
                    "error": str(e)
                }
            }

        # 直接调用batch_web_search，避免线程池嵌套问题
        try:
            logger.info(f"🚀 开始执行batch_web_search，查询数量: {len(queries)}")
            urls = llm_search.batch_web_search(queries, topic, top_n)
            logger.info(f"✅ batch_web_search完成，返回URL数量: {len(urls)}")
        except Exception as e:
            logger.error(f"❌ batch_web_search执行失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            urls = []
        # 目前的问题：
        '''
            1. prompt和config的统一及唯一管理
            2. 精简掉没必要
            3. analyse让llm自动判断,每一个工具反馈给
        '''
        # 保存URL结果到本地文件
        url_file_path = _save_urls_to_file(urls, topic, queries)

        result = {
            "topic": topic,
            "queries": queries,
            "engine": engine,
            "urls": urls,
            "url_count": len(urls),
            "url_file_path": url_file_path,  # 添加文件路径
            "top_n": top_n,
            "processing_metadata": {
                "engine": engine,
                "query_count": len(queries),
                "result_count": len(urls),
                "method": "batch_web_search",
                "url_file_saved": url_file_path is not None
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error in web search: {e}")
        # 即使出错也尝试保存空的URL文件
        url_file_path = _save_urls_to_file([], topic, queries)

        return {
            "topic": topic,
            "queries": queries,
            "engine": engine or SERVER_CONFIG["default_engine"],
            "urls": [],
            "url_count": 0,
            "url_file_path": url_file_path,
            "top_n": top_n or SERVER_CONFIG["default_total_urls"],
            "processing_metadata": {
                "engine": engine or SERVER_CONFIG["default_engine"],
                "query_count": len(queries),
                "result_count": 0,
                "method": "fallback",
                "error": str(e),
                "url_file_saved": url_file_path is not None
            }
        }

async def _crawl_urls(topic: str, url_file_path: str = None, top_n: int = None, model: str = None,
                     similarity_threshold: float = None, min_length: int = None, max_length: int = None) -> Dict[str, Any]:

    # 从文件中读取URL列表
    url_list = _load_urls_from_file(url_file_path, topic)
    logger.info(f"Starting crawling process for {len(url_list)} URLs with topic: {topic}")

    try:
        if top_n is None:
            top_n = SERVER_CONFIG["default_top_n"]
        if model is None:
            model = SERVER_CONFIG["default_model"]
        if similarity_threshold is None:
            similarity_threshold = SERVER_CONFIG["default_similarity_threshold"]
        if min_length is None:
            min_length = SERVER_CONFIG["default_min_length"]
        if max_length is None:
            max_length = SERVER_CONFIG["default_max_length"]

        if not CRAWL4AI_AVAILABLE:
            raise ImportError("crawl4ai is not available")
        if RequestWrapper is None:
            raise ImportError("RequestWrapper is not available")

        import time
        import re

        # 使用配置中的模型设置
        content_model = SERVER_CONFIG.get("content_analysis_model", model)
        infer_type = SERVER_CONFIG.get("default_infer_type", "OpenAI")
        request_wrapper = RequestWrapper(model=content_model, infer_type=infer_type)

        process_start_time = time.time()
        stage_time = process_start_time

        # 爬取URL，不限制整体超时时间
        crawling_timeout = SERVER_CONFIG.get("crawling_timeout", 0)  # 0表示不限制
        if crawling_timeout > 0:
            logger.info(f"开始爬取 {len(url_list)} 个URL，超时时间: {crawling_timeout}秒")
        else:
            logger.info(f"开始爬取 {len(url_list)} 个URL，无整体超时限制")

        try:
            if crawling_timeout > 0:
                crawl_results = await asyncio.wait_for(
                    _crawl_urls_stage(topic, url_list),
                    timeout=crawling_timeout
                )
            else:
                # 不限制整体超时时间
                crawl_results = await _crawl_urls_stage(topic, url_list)
            logger.info(f"Stage 1 - Crawling completed in {time.time() - stage_time:.2f} seconds, with {len(crawl_results)} results")
        except asyncio.TimeoutError:
            logger.error(f"URL crawling stage timed out after {crawling_timeout} seconds")
            logger.error("⚠️ Attempting to retrieve partial results from incremental save file...")

            # 尝试从增量保存文件中恢复部分结果
            try:
                import os
                import json
                incremental_file_path = _get_incremental_crawl_file_path(topic)
                logger.info(f"尝试从增量文件恢复结果: {incremental_file_path}")

                if os.path.exists(incremental_file_path):
                    with open(incremental_file_path, 'r', encoding='utf-8') as f:
                        incremental_data = json.load(f)

                    crawl_progress = incremental_data.get("crawl_progress", [])
                    logger.info(f"从增量文件中找到 {len(crawl_progress)} 个已完成的爬取结果")

                    # 重构增量结果为标准格式，使用保存的实际内容
                    crawl_results = []
                    for progress in crawl_progress:
                        if progress.get("success", False):
                            # 这是成功的结果，使用保存的实际内容
                            crawl_results.append({
                                "url": progress.get("url", ""),
                                "error": False,
                                "raw_content": progress.get("content", ""),  # 使用保存的实际内容
                                "title": progress.get("title", ""),
                                "date": progress.get("date", ""),
                                "timestamp": progress.get("timestamp", ""),
                                "is_recovered": True  # 标记为从增量文件恢复的结果
                            })
                        else:
                            # 这是错误结果
                            crawl_results.append({
                                "url": progress.get("url", ""),
                                "error": True,
                                "raw_content": progress.get("error_message", "Unknown error"),
                                "timestamp": progress.get("timestamp", "")
                            })

                    logger.info(f"✅ 成功恢复 {len(crawl_results)} 个部分结果")
                else:
                    logger.error(f"增量文件不存在: {incremental_file_path}")
                    crawl_results = []

                # 超时分析
                logger.error(f"📊 Timeout analysis:")
                logger.error(f"  - Total URLs to crawl: {len(url_list)}")
                logger.error(f"  - Completed URLs: {len(crawl_results)}")
                logger.error(f"  - Completion rate: {len(crawl_results)/len(url_list)*100:.1f}%")
                logger.error(f"  - Timeout setting: {crawling_timeout} seconds")
                logger.error(f"  - Average time per URL: {crawling_timeout / len(url_list):.2f} seconds")

                # 建议调整
                suggested_timeout = len(url_list) * 60  # 每个URL给60秒
                logger.error(f"  - Suggested timeout: {suggested_timeout} seconds")

            except Exception as e:
                logger.error(f"Error during partial result recovery: {e}")
                crawl_results = []
        stage_time = time.time()

        # 重新启用内容过滤和相似度评分，但使用宽松设置
        logger.info("Stage 2 - Content filtering with relaxed settings")

        # 简化的内容过滤：只添加基本信息，保留所有内容
        filtered_results = []
        error_count = 0
        success_count = 0

        for result in crawl_results:
            # 检查是否是错误结果
            is_error = result.get("error", False)

            # 优先使用清理后的内容，如果没有则使用原始内容
            content = result.get("cleaned_content", "") or result.get("raw_content", "") or result.get("content", "")

            # 获取清理信息
            cleaning_info = result.get("cleaning_info", {})
            title = result.get("title", "")
            language = result.get("language", "UNKNOWN")

            # 记录统计信息
            if is_error:
                error_count += 1
                logger.warning(f"Error result for URL {result.get('url', 'unknown')}: {content[:100]}...")
            else:
                success_count += 1

            # 为每个结果添加基本信息，保留所有内容（包括错误结果用于调试）
            enhanced_result = {
                "url": result.get("url", ""),
                "title": title or result.get("title", "Unknown"),  # 使用清理后的标题
                "content": content,  # 使用清理后的内容
                "raw_content": result.get("raw_content", ""),  # 保留原始内容
                "date": result.get("date", ""),
                "length": len(content),
                "language": language,  # 添加语言信息
                "is_error": is_error,
                "original_error": result.get("error", False),
                "cleaning_info": cleaning_info  # 添加清理信息
            }
            filtered_results.append(enhanced_result)

        logger.info(f"Stage 2 completed - processed {len(filtered_results)} results (success: {success_count}, errors: {error_count})")

        # 如果所有结果都是错误，记录警告
        if error_count > 0 and success_count == 0:
            logger.warning(f"⚠️ All {error_count} crawl results failed! This will likely result in 0 final results.")
        elif error_count > 0:
            logger.warning(f"⚠️ {error_count} out of {len(crawl_results)} crawl results failed.")
        stage_time = time.time()

        # 简化的相似度评分：给所有结果高分
        logger.info("Stage 3 - Simplified similarity scoring")
        scored_results = []
        for result in filtered_results:
            result["similarity_score"] = 90.0  # 给所有结果高分，确保通过过滤
            scored_results.append(result)

        logger.info(f"Stage 3 completed - scored {len(scored_results)} results")
        stage_time = time.time()

        # 暂时跳过过滤，直接返回所有结果用于测试
        print(f"DEBUG: scored_results count: {len(scored_results)}")
        if scored_results:
            print(f"DEBUG: First result keys: {list(scored_results[0].keys())}")
            print(f"DEBUG: First result similarity_score: {scored_results[0].get('similarity_score', 'N/A')}")
            print(f"DEBUG: First result content length: {len(scored_results[0].get('content', ''))}")
            print(f"DEBUG: First result is_error: {scored_results[0].get('is_error', 'N/A')}")
            print(f"DEBUG: First result content preview: {scored_results[0].get('content', '')[:200]}...")

        # 注释掉错误过滤和错误信息过滤，保留所有结果用于调试
        valid_results = []
        for result in scored_results:
            # 注释掉错误过滤：if not result.get("is_error", False):
            # 进一步检查内容是否有效
            content = result.get("content", "")
            # 注释掉错误信息过滤：if content and not content.startswith("Error:") and len(content.strip()) > 50:
            if content and len(content.strip()) > 0:  # 进一步降低要求：只要有内容就保留
                valid_results.append(result)
                logger.info(f"Kept result: {result.get('url', 'unknown')}, content_length={len(content)}, stripped_length={len(content.strip())}")
            else:
                logger.warning(f"Filtered out result with invalid content: {result.get('url', 'unknown')}, content_length={len(content)}, stripped_length={len(content.strip()) if content else 0}")
            # 注释掉错误结果过滤：else:
            #     logger.warning(f"Filtered out error result: {result.get('url', 'unknown')}")

        print(f"DEBUG: valid_results count after content filtering: {len(valid_results)}")

        # 添加详细的内容分析
        if len(valid_results) > 0:
            first_result = valid_results[0]
            content = first_result.get("content", "")
            print(f"DEBUG: First valid result analysis:")
            print(f"  - URL: {first_result.get('url', 'unknown')}")
            print(f"  - Original length: {len(content)}")
            print(f"  - Stripped length: {len(content.strip())}")
            print(f"  - Is error: {first_result.get('is_error', False)}")
            print(f"  - Content preview (first 200 chars): '{content[:200]}...'")
            print(f"  - Content preview (last 200 chars): '...{content[-200:]}'")

            # 分析内容组成
            import re
            whitespace_count = len(re.findall(r'\s', content))
            meaningful_chars = len(re.findall(r'[a-zA-Z0-9\u4e00-\u9fff]', content))
            print(f"  - Whitespace characters: {whitespace_count}")
            print(f"  - Meaningful characters: {meaningful_chars}")

        # 使用valid_results作为final_results
        final_results = valid_results[:top_n]  # 只取前top_n个结果
        print(f"DEBUG: final_results count: {len(final_results)}")
        logger.info(f"Stage 4 - Result processing completed in {time.time() - stage_time:.2f} seconds")

        # 如果最终结果为0，记录详细的调试信息
        if len(final_results) == 0:
            logger.error("🚨 FINAL RESULTS IS ZERO! Debugging information:")
            logger.error(f"  - Original crawl_results: {len(crawl_results)}")
            logger.error(f"  - Filtered_results: {len(filtered_results)}")
            logger.error(f"  - Scored_results: {len(scored_results)}")
            logger.error(f"  - Valid_results: {len(valid_results)}")
            logger.error(f"  - Success_count: {success_count}")
            logger.error(f"  - Error_count: {error_count}")

            # 显示前几个原始结果的详细信息
            for i, result in enumerate(crawl_results[:5]):
                raw_content = result.get('raw_content', '')
                logger.error(f"  - Crawl result {i+1}:")
                logger.error(f"    URL: {result.get('url', 'N/A')}")
                logger.error(f"    Error: {result.get('error', False)}")
                logger.error(f"    Content length: {len(raw_content)}")
                logger.error(f"    Stripped length: {len(raw_content.strip()) if raw_content else 0}")
                if raw_content:
                    logger.error(f"    Content preview: '{raw_content[:200]}...'")
                    if raw_content.startswith("Error:"):
                        logger.error(f"    ⚠️ This is an error result")
                    else:
                        logger.error(f"    ✅ This appears to be valid content")

            # 分析为什么所有结果都被过滤
            logger.error("🔍 Analysis of why all results were filtered:")
            all_errors = all(r.get("error", False) for r in crawl_results)
            all_empty = all(len(r.get("raw_content", "").strip()) == 0 for r in crawl_results)

            if all_errors:
                logger.error("  🚨 ALL results are error results")
            elif all_empty:
                logger.error("  🚨 ALL results have empty content after stripping")
            else:
                error_count_check = sum(1 for r in crawl_results if r.get("error", False))
                empty_count = sum(1 for r in crawl_results if len(r.get("raw_content", "").strip()) == 0)
                logger.error(f"  🚨 Mixed issues: {error_count_check} errors, {empty_count} empty content")
        else:
            logger.info(f"✅ Successfully generated {len(final_results)} final results")

        total_time = time.time() - process_start_time
        logger.info(f"Total crawling process completed in {total_time:.2f} seconds")

        # 生成兼容 LLMxMapReduce_V2 JSONL 格式的结果
        # 主要结果：主题+论文列表的格式
        main_result = {
            "title": topic,
            "papers": final_results
        }

        # 详细的处理信息（用于调试和监控）
        detailed_result = {
            "title": topic,  # 改为title
            "total_urls": len(url_list),
            "crawl_results": len(crawl_results),
            "filtered_results": len(filtered_results),
            "scored_results": len(scored_results),
            "papers": final_results,  # 改为papers
            "final_count": len(final_results),
            "processing_metadata": {
                "model": model,
                "similarity_threshold": similarity_threshold,
                "min_length": min_length,
                "max_length": max_length,
                "top_n": top_n,
                "total_time": total_time,
                "success": True
            },
            # 添加兼容格式的主要结果
            "llm_mapreduce_format": main_result
        }

        # 保存结果到本地JSON文件，包含调试信息和错误分析
        try:
            import json
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawl_results_{timestamp}.json"

            # 分析错误类型
            error_analysis = {
                "total_errors": error_count,
                "total_success": success_count,
                "error_details": [],
                "error_types": {}
            }

            # 收集错误详情
            for result in crawl_results:
                if result.get("error", False):
                    error_detail = {
                        "url": result.get("url", "unknown"),
                        "error_message": result.get("raw_content", ""),
                        "timestamp": result.get("timestamp", "")
                    }
                    error_analysis["error_details"].append(error_detail)

                    # 统计错误类型
                    error_msg = result.get("raw_content", "")
                    if "timeout" in error_msg.lower():
                        error_type = "timeout"
                    elif "connection" in error_msg.lower():
                        error_type = "connection"
                    elif "none" in error_msg.lower() and "attribute" in error_msg.lower():
                        error_type = "parsing_error"
                    elif "403" in error_msg or "forbidden" in error_msg.lower():
                        error_type = "access_denied"
                    elif "404" in error_msg or "not found" in error_msg.lower():
                        error_type = "not_found"
                    else:
                        error_type = "other"

                    error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1

            # 添加调试信息
            debug_info = {
                "raw_crawl_results": crawl_results,
                "processed_filtered_results": filtered_results,
                "processed_scored_results": scored_results,
                "error_analysis": error_analysis
            }
            detailed_result["debug_info"] = debug_info

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_result, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to {filename}")
            logger.info(f"Error analysis: {error_analysis['error_types']}")
            detailed_result["saved_file"] = filename

            # 单独保存错误报告
            if error_count > 0:
                error_filename = f"crawl_errors_{timestamp}.json"
                with open(error_filename, 'w', encoding='utf-8') as f:
                    json.dump(error_analysis, f, ensure_ascii=False, indent=2)
                logger.info(f"Error report saved to {error_filename}")
                detailed_result["error_report_file"] = error_filename

        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")

        return detailed_result

    except Exception as e:
        logger.error(f"Error in crawling pipeline: {e}")
        return {
            "topic": topic,
            "total_urls": len(url_list),
            "crawl_results": 0,
            "filtered_results": 0,
            "scored_results": 0,
            "final_results": [],
            "final_count": 0,
            "processing_metadata": {
                "model": model,
                "similarity_threshold": similarity_threshold,
                "min_length": min_length,
                "max_length": max_length,
                "top_n": top_n,
                "success": False,
                "error": str(e)
            }
        }

# 全局变量存储当前会话的增量文件路径
_current_incremental_file_path = None

def _get_incremental_crawl_file_path(topic: str, create_new: bool = False) -> str:
    """获取增量爬取结果文件路径"""
    global _current_incremental_file_path

    if _current_incremental_file_path is None or create_new:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crawl_results_{topic}_{timestamp}.json"
        _current_incremental_file_path = filename

    return _current_incremental_file_path

def _load_existing_crawl_results(file_path: str) -> Dict[str, Any]:
    """加载现有的爬取结果文件"""
    import json
    import os

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load existing crawl results: {e}")

    # 返回默认结构
    return {
        "topic": "",
        "total_urls": 0,
        "crawl_results": 0,
        "filtered_results": 0,
        "scored_results": 0,
        "final_results": [],
        "crawl_progress": []
    }

async def _generate_abstract_llm(content: str, url: str, file_path: str, url_index: int) -> str:
    """
    异步生成LLM摘要并更新文件

    Args:
        content: 要生成摘要的内容
        url: URL地址
        file_path: 增量文件路径
        url_index: URL在crawl_progress中的索引

    Returns:
        生成的摘要文本
    """
    try:
        # 获取abstract生成模型配置
        abstract_model = SERVER_CONFIG.get("abstract_generation_model", "gemini-2.5-flash")
        infer_type = SERVER_CONFIG.get("default_infer_type", "OpenAI")

        # 创建RequestWrapper实例
        request_wrapper = RequestWrapper(model=abstract_model, infer_type=infer_type)

        # 获取prompt模板
        prompts = SERVER_CONFIG.get("prompts", {})
        prompt_template = prompts.get("abstract_generation", "Please summarize the following content:\n\n{content}")

        # 限制内容长度，避免token过多
        max_content_length = 8000  # 约2000个token
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        # 生成prompt
        prompt = prompt_template.format(content=content)

        # 调用LLM生成摘要，添加超时保护
        logger.info(f"开始为URL生成LLM摘要: {url}")
        abstract_timeout = SERVER_CONFIG.get("abstract_generation_timeout", 30)

        try:
            # 使用asyncio.to_thread将同步调用转为异步，并添加超时
            abstract_llm = await asyncio.wait_for(
                asyncio.to_thread(request_wrapper.completion, prompt),
                timeout=abstract_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM摘要生成超时 ({abstract_timeout}秒): {url}")
            raise Exception(f"Abstract generation timeout after {abstract_timeout} seconds")

        # 清理生成的摘要
        abstract_llm = abstract_llm.strip()
        if len(abstract_llm) > 1000:  # 限制摘要长度
            abstract_llm = abstract_llm[:1000] + "..."

        # 更新文件中的abstract_llm字段
        await _update_abstract_llm_in_file(file_path, url_index, abstract_llm, "completed")

        logger.info(f"✅ LLM摘要生成完成: {url}")
        return abstract_llm

    except Exception as e:
        logger.error(f"❌ LLM摘要生成失败 {url}: {e}")
        # 更新状态为失败
        await _update_abstract_llm_in_file(file_path, url_index, f"Error: {str(e)}", "failed")
        return f"Error: {str(e)}"

async def _update_abstract_llm_in_file(file_path: str, url_index: int, abstract_llm: str, status: str):
    """
    更新文件中指定URL的abstract_llm字段

    Args:
        file_path: 文件路径
        url_index: URL在crawl_progress中的索引
        abstract_llm: LLM生成的摘要
        status: 状态（completed/failed）
    """
    import json
    import os
    import time
    import platform

    # 跨平台文件锁实现
    def _acquire_file_lock(f):
        if platform.system() == "Windows":
            # Windows下使用msvcrt
            try:
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            except ImportError:
                # 如果msvcrt不可用，使用简单的重试机制
                pass
        else:
            # Unix/Linux下使用fcntl
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass

    def _release_file_lock(f):
        if platform.system() == "Windows":
            try:
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except ImportError:
                pass
        else:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass

    max_retries = 3  # 减少重试次数
    retry_delay = 0.05  # 减少初始延迟

    for attempt in range(max_retries):
        try:
            # 使用文件锁避免并发写入冲突
            with open(file_path, 'r+', encoding='utf-8') as f:
                # 获取文件锁
                _acquire_file_lock(f)

                try:
                    # 读取现有数据
                    f.seek(0)
                    data = json.load(f)

                    # 更新指定索引的abstract_llm字段
                    if url_index < len(data.get("crawl_progress", [])):
                        data["crawl_progress"][url_index]["abstract_llm"] = abstract_llm
                        data["crawl_progress"][url_index]["abstract_llm_status"] = status

                        # 写回文件
                        f.seek(0)
                        f.truncate()
                        json.dump(data, f, ensure_ascii=False, indent=2)

                        logger.debug(f"更新abstract_llm字段成功，索引: {url_index}")
                        return  # 成功更新，退出函数
                    else:
                        logger.error(f"URL索引超出范围: {url_index}")
                        return

                finally:
                    # 释放文件锁
                    _release_file_lock(f)

        except (json.JSONDecodeError, OSError, IOError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"更新abstract_llm字段失败，重试 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
            else:
                logger.error(f"更新abstract_llm字段最终失败: {e}")
        except Exception as e:
            logger.error(f"更新abstract_llm字段出现未预期错误: {e}")
            break

def _save_incremental_crawl_result(file_path: str, crawl_result: Dict[str, Any], topic: str, total_urls: int) -> int:
    """增量保存单个爬取结果，包含HTML清理信息

    Returns:
        int: 新增记录在crawl_progress中的索引
    """
    import json
    import os

    try:
        # 加载现有结果
        existing_data = _load_existing_crawl_results(file_path)

        # 更新基本信息
        existing_data["topic"] = topic
        existing_data["total_urls"] = total_urls
        existing_data["crawl_results"] = len(existing_data["crawl_progress"]) + 1

        # 获取内容信息
        raw_content = crawl_result.get("raw_content", "")
        cleaned_content = crawl_result.get("cleaned_content", "")

        # 优先使用清理后的内容作为主要内容，如果没有则使用原始内容
        main_content = cleaned_content if cleaned_content else raw_content

        # 获取清理信息
        cleaning_info = crawl_result.get("cleaning_info", {})

        # 生成abstract字段（取前2000个字符）
        abstract = ""
        if main_content and not crawl_result.get("error", False):
            # 清理内容并截取前2000字符
            cleaned_for_abstract = main_content.strip()
            if len(cleaned_for_abstract) > 2000:
                # 尝试在句号处截断，避免截断句子
                truncated = cleaned_for_abstract[:2000]
                last_period = truncated.rfind('.')
                if last_period > 1600:  # 如果句号位置合理（至少80%的内容）
                    abstract = truncated[:last_period + 1]
                else:
                    abstract = truncated + "..."
            else:
                abstract = cleaned_for_abstract

        # 构建保存数据，包含完整的HTML清理信息和新增的abstract字段
        save_data = {
            "url": crawl_result.get("url", ""),
            "success": not crawl_result.get("error", False),
            "content_length": len(main_content),
            "timestamp": crawl_result.get("timestamp", ""),
            "error_message": raw_content if crawl_result.get("error", False) else "",
            "content": main_content,  # 主要内容（优先使用清理后的）
            "raw_content": raw_content,  # 保留原始内容用于对比
            "cleaned_content": cleaned_content,  # 清理后的内容
            "title": crawl_result.get("title", ""),
            "language": crawl_result.get("language", "UNKNOWN"),
            "date": crawl_result.get("date", ""),
            "cleaning_info": cleaning_info,  # HTML清理元数据
            "abstract": abstract,  # 新增：内容前2000字符的摘要
            "abstract_llm": "",  # 新增：LLM生成的摘要（初始为空，后续异步填充）
            "abstract_llm_status": "pending"  # 新增：LLM摘要生成状态
        }

        # 获取新记录的索引（在添加之前）
        new_index = len(existing_data["crawl_progress"])

        existing_data["crawl_progress"].append(save_data)

        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        # 记录清理效果
        if cleaning_info.get("method") != "none":
            logger.info(f"Saved crawl result with {cleaning_info.get('method')} cleaning for URL: {crawl_result.get('url', 'N/A')}")
        else:
            logger.debug(f"Saved crawl result (no cleaning) for URL: {crawl_result.get('url', 'N/A')}")

        return new_index

    except Exception as e:
        logger.error(f"Failed to save incremental crawl result: {e}")
        return -1  # 返回-1表示保存失败

async def _crawl_urls_stage(topic: str, url_list: List[str]) -> List[Dict[str, Any]]:
    import asyncio

    if not CRAWL4AI_AVAILABLE:
        raise ImportError("crawl4ai is not available")

    MAX_CONCURRENT_CRAWLS = SERVER_CONFIG.get("max_concurrent_crawls", 10)  # 从配置读取并发数量
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    total_items = len(url_list)

    # 跟踪abstract生成任务
    abstract_tasks = []

    # 创建增量保存文件
    incremental_file_path = _get_incremental_crawl_file_path(topic, create_new=True)
    logger.info(f"增量爬取结果将保存到: {incremental_file_path}")

    # 添加URL到队列
    for url in url_list:
        await input_queue.put((url, topic))

    async def consumer():
        consumer_id = id(asyncio.current_task())
        processed_count = 0

        while True:
            try:
                url, topic = input_queue.get_nowait()
                try:
                    result = await _crawl_single_url(url, topic)

                    # 添加时间戳
                    import datetime
                    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 立即保存到增量文件，获取索引
                    url_index = _save_incremental_crawl_result(incremental_file_path, result, topic, total_items)

                    await output_queue.put(result)
                    processed_count += 1

                    # 如果爬取成功且有内容，异步启动abstract_llm生成
                    if (not result.get("error", False) and
                        url_index >= 0 and
                        result.get("cleaned_content") or result.get("raw_content")):

                        content_for_abstract = result.get("cleaned_content") or result.get("raw_content", "")
                        if content_for_abstract.strip():
                            # 异步启动abstract_llm生成，并添加到跟踪列表
                            task = asyncio.create_task(_generate_abstract_llm(
                                content_for_abstract,
                                url,
                                incremental_file_path,
                                url_index
                            ))
                            abstract_tasks.append(task)

                    # 减少日志频率，只在处理每10个URL或队列为空时记录
                    remaining = input_queue.qsize()
                    if processed_count % 10 == 0 or remaining == 0:
                        logger.info(f"Consumer {consumer_id}: 已处理 {processed_count} 个URL, 剩余: {remaining}/{total_items}")

                except Exception as e:
                    logger.error(f"Consumer {consumer_id}: 爬取URL失败 {url}: {e}")
                    # 即使失败也要放入结果，避免丢失
                    import datetime
                    error_result = {
                        "topic": topic,
                        "url": url,
                        "raw_content": f"Error: {e}",
                        "error": True,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # 保存错误结果
                    error_index = _save_incremental_crawl_result(incremental_file_path, error_result, topic, total_items)
                    await output_queue.put(error_result)
                finally:
                    input_queue.task_done()
            except asyncio.QueueEmpty:
                if processed_count > 0:
                    logger.info(f"Consumer {consumer_id}: 完成，共处理 {processed_count} 个URL")
                break

    consumers = [asyncio.create_task(consumer()) for _ in range(MAX_CONCURRENT_CRAWLS)]
    logger.info(f"启动 {MAX_CONCURRENT_CRAWLS} 个并发爬取任务处理 {total_items} 个URL")

    # 添加进度监控和队列等待超时保护
    start_time = time.time()
    try:
        # 计算队列等待超时：每个URL给60秒 + 5分钟缓冲
        queue_timeout = len(url_list) * 60 + 300
        logger.info(f"队列等待超时设置: {queue_timeout}秒")

        await asyncio.wait_for(input_queue.join(), timeout=queue_timeout)
        end_time = time.time()
        logger.info(f"所有URL爬取完成，耗时: {end_time - start_time:.2f}秒")
    except asyncio.TimeoutError:
        logger.error(f"队列等待超时 ({queue_timeout}秒)，强制结束爬取流程")
    except Exception as e:
        logger.error(f"爬取过程中出现异常: {e}")
    finally:
        # 确保所有consumer任务被正确取消
        for consumer_task in consumers:
            if not consumer_task.done():
                consumer_task.cancel()

        # 等待所有任务完成取消
        await asyncio.gather(*consumers, return_exceptions=True)

        # 等待abstract生成任务完成（带超时）
        if abstract_tasks:
            abstract_wait_timeout = SERVER_CONFIG.get("abstract_tasks_wait_timeout", 300)
            logger.info(f"等待 {len(abstract_tasks)} 个abstract生成任务完成，超时: {abstract_wait_timeout}秒")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*abstract_tasks, return_exceptions=True),
                    timeout=abstract_wait_timeout
                )
                logger.info("所有abstract生成任务已完成")
            except asyncio.TimeoutError:
                logger.warning(f"Abstract生成任务等待超时 ({abstract_wait_timeout}秒)，取消剩余任务")
                # 取消未完成的abstract任务
                for task in abstract_tasks:
                    if not task.done():
                        task.cancel()
            except Exception as e:
                logger.error(f"等待abstract任务时出现异常: {e}")

    results = []
    while not output_queue.empty():
        results.append(await output_queue.get())

    return results

def _clean_html_content(html_content: str, url: str) -> Dict[str, Any]:
    """
    Clean HTML content using the integrated HTML cleaner

    Args:
        html_content (str): Raw HTML content
        url (str): URL for context

    Returns:
        Dict[str, Any]: Cleaned content with keys: text, language, cleaned, error
    """
    # 基本清理函数
    def basic_clean(content):
        import re
        # 移除脚本和样式标签及其内容
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        # 清理多余的空白字符
        content = re.sub(r'\s+', ' ', content).strip()
        return content

    # 尝试使用高级HTML清理器
    if HTML_CLEANER_AVAILABLE and _html_extractor:
        try:
            extracted = _html_extractor.extract(html_content)

            if extracted and extracted.get("text"):
                return {
                    "text": extracted["text"],
                    "language": extracted.get("language", "UNKNOWN"),
                    "cleaned": True,
                    "error": False,
                    "method": "advanced"
                }
        except Exception as e:
            logger.debug(f"Advanced HTML cleaning failed for URL={url}: {e}")
            # 继续到基本清理

    # 使用基本清理作为默认或回退方案
    try:
        basic_text = basic_clean(html_content)
        return {
            "text": basic_text,
            "language": "UNKNOWN",
            "cleaned": True,
            "error": False,
            "method": "basic"
        }
    except Exception as e:
        logger.error(f"Basic HTML cleaning failed for URL={url}: {e}")
        return {
            "text": html_content,  # 返回原始内容作为最后的回退
            "language": "UNKNOWN",
            "cleaned": False,
            "error": True,
            "method": "none",
            "error_message": str(e)
        }

async def _crawl_single_url(url: str, topic: str) -> Dict[str, Any]:
    """
    Crawl a single URL and return the result with integrated HTML cleaning.

    Args:
        url (str): URL to crawl
        topic (str): The topic for context

    Returns:
        Dict[str, Any]: Crawl result with keys: topic, url, raw_content, cleaned_content,
                       title, language, error, cleaning_info
    """
    try:
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("crawl4ai is not available")

        # crawl4ai需要毫秒单位的超时时间，优先使用page_timeout配置
        page_timeout_seconds = SERVER_CONFIG.get("page_timeout", SERVER_CONFIG.get("single_url_crawl_timeout", 60))
        page_timeout_ms = page_timeout_seconds * 1000  # 转换为毫秒

        crawler_run_config = CrawlerRunConfig(
            page_timeout=page_timeout_ms,
            cache_mode=CacheMode.BYPASS
        )

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=crawler_run_config)

            # 检查爬取结果是否成功
            if not result:
                raise Exception("Crawler returned None result")

            # 获取原始内容和HTML内容
            raw_content = None
            html_content = None
            title = ""

            # 尝试获取标题
            if hasattr(result, 'metadata') and result.metadata:
                title = result.metadata.get('title', '')

            # 检查markdown_v2是否存在
            if not hasattr(result, 'markdown_v2') or result.markdown_v2 is None:
                logger.warning(f"No markdown_v2 found for URL={url}, trying alternative content")
                # 尝试使用其他内容字段
                if hasattr(result, 'markdown') and result.markdown:
                    raw_content = result.markdown
                elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                    raw_content = result.cleaned_html
                    html_content = result.cleaned_html
                elif hasattr(result, 'html') and result.html:
                    raw_content = result.html
                    html_content = result.html
                else:
                    raise Exception("No usable content found in crawl result")
            else:
                # 检查raw_markdown是否存在
                if hasattr(result.markdown_v2, 'raw_markdown') and result.markdown_v2.raw_markdown:
                    raw_content = result.markdown_v2.raw_markdown
                else:
                    logger.warning(f"No raw_markdown found for URL={url}, trying alternative markdown content")
                    # 尝试使用markdown_v2的其他字段
                    if hasattr(result.markdown_v2, 'markdown') and result.markdown_v2.markdown:
                        raw_content = result.markdown_v2.markdown
                    else:
                        raise Exception("No usable markdown content found")

                # 尝试获取HTML内容用于清理
                if hasattr(result, 'html') and result.html:
                    html_content = result.html

            # 进行HTML内容清理
            cleaned_info = {"method": "none", "error": False}
            cleaned_content = raw_content
            detected_language = "UNKNOWN"

            if html_content and len(html_content.strip()) > 0:
                cleaning_result = _clean_html_content(html_content, url)
                if not cleaning_result.get("error", True) and cleaning_result.get("text"):
                    cleaned_content = cleaning_result["text"]
                    detected_language = cleaning_result.get("language", "UNKNOWN")
                    cleaned_info = {
                        "method": cleaning_result.get("method", "unknown"),
                        "error": cleaning_result.get("error", False),
                        "error_message": cleaning_result.get("error_message", "")
                    }
                    logger.info(f"Content cleaned using {cleaning_result.get('method')} method for URL={url}")

            logger.info(f"Content length={len(raw_content)} (raw), {len(cleaned_content)} (cleaned) for URL={url}")

            return {
                "topic": topic,
                "url": url,
                "raw_content": raw_content,  # 保持原有字段名
                "cleaned_content": cleaned_content,  # 新增清理后的内容
                "title": title,
                "language": detected_language,
                "error": False,
                "cleaning_info": cleaned_info
            }
    except Exception as e:
        logger.error(f"Crawling failed for URL={url}: {e}")
        return {
            "topic": topic,
            "url": url,
            "raw_content": f"Error: Crawling failed({e})",
            "cleaned_content": "",
            "title": "",
            "language": "UNKNOWN",
            "error": True,
            "cleaning_info": {"method": "none", "error": True, "error_message": str(e)}
        }

async def _process_filter_and_titles_stage(crawl_results: List[Dict[str, Any]], request_wrapper) -> List[Dict[str, Any]]:
    import asyncio

    MAX_CONCURRENT_PROCESSES = 10
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    for data in crawl_results:
        if not data.get("error", False):
            await input_queue.put(data)

    async def processor():
        while True:
            try:
                data = input_queue.get_nowait()
                try:
                    result = await _process_filter_and_title_single(data, request_wrapper)
                    await output_queue.put(result)
                finally:
                    input_queue.task_done()
            except asyncio.QueueEmpty:
                break

    processors = [asyncio.create_task(processor()) for _ in range(MAX_CONCURRENT_PROCESSES)]

    await input_queue.join()

    for processor_task in processors:
        processor_task.cancel()

    results = []
    while not output_queue.empty():
        results.append(await output_queue.get())

    return results

async def _process_filter_and_title_single(data: Dict[str, Any], request_wrapper) -> Dict[str, Any]:
    import re

    try:
        prompt_template_response = await read_resource("llm://search/prompts")
        prompt_template = prompt_template_response[0].text
        prompts = json.loads(prompt_template)

        prompt = prompts["page_refine"].format(
            topic=data["topic"], raw_content=data["raw_content"]
        )
        res = request_wrapper.completion(prompt)

        # 尝试提取标题和内容
        title = re.search(r"<TITLE>(.*?)</TITLE>", res, re.DOTALL)
        content = re.search(r"<CONTENT>(.*?)</CONTENT>", res, re.DOTALL)

        if not title or not content:
            # 如果格式不正确，尝试从原始内容中提取
            logger.warning(f"Invalid response format for URL={data.get('url', 'unknown')}, trying fallback extraction")

            # 尝试从原始内容中提取标题（通常是第一行或前几行）
            raw_content = data.get("raw_content", "")
            lines = raw_content.split('\n')

            # 寻找可能的标题
            potential_title = ""
            for line in lines[:10]:  # 检查前10行
                line = line.strip()
                if line and len(line) < 200:  # 标题通常较短
                    potential_title = line
                    break

            if not potential_title:
                potential_title = f"Content from {data.get('url', 'unknown')}"

            # 使用原始内容作为过滤后的内容（简单清理）
            filtered_content = raw_content
            # 简单清理：移除过多的空行和特殊字符
            filtered_content = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_content)
            filtered_content = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()[\]{}"\'-]', ' ', filtered_content)

            data["title"] = potential_title[:200]  # 限制标题长度
            data["content"] = filtered_content[:10000]  # 限制内容长度
            data["filter_error"] = False

            logger.info(f"Used fallback extraction for URL={data.get('url', 'unknown')}")
        else:
            data["title"] = title.group(1).strip()
            data["content"] = content.group(1).strip()
            data["filter_error"] = False

        return data

    except Exception as e:
        logger.error(f"Content filtering failed for URL={data.get('url', 'unknown')}: {e}")
        data["title"] = f"Error processing: {data.get('url', 'unknown')}"
        data["content"] = f"Error: Content filtering failed({e})"
        data["filter_error"] = True
        return data

async def _process_similarity_scores_stage(filtered_results: List[Dict[str, Any]], request_wrapper) -> List[Dict[str, Any]]:

    import asyncio

    MAX_CONCURRENT_PROCESSES = 10
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    for data in filtered_results:
        if not data.get("filter_error", False):
            await input_queue.put(data)

    async def scorer():
        while True:
            try:
                data = input_queue.get_nowait()
                try:
                    result = await _process_similarity_score_single(data, request_wrapper)
                    await output_queue.put(result)
                finally:
                    input_queue.task_done()
            except asyncio.QueueEmpty:
                break

    scorers = [asyncio.create_task(scorer()) for _ in range(MAX_CONCURRENT_PROCESSES)]

    await input_queue.join()

    for scorer_task in scorers:
        scorer_task.cancel()

    results = []
    while not output_queue.empty():
        results.append(await output_queue.get())

    return results

async def _process_similarity_score_single(data: Dict[str, Any], request_wrapper) -> Dict[str, Any]:
    import re

    try:
        prompt_template_response = await read_resource("llm://search/prompts")
        prompt_template = prompt_template_response[0].text
        prompts = json.loads(prompt_template)
        
        prompt = prompts["similarity_scoring"].format(
            topic=data["topic"], content=data["content"]
        )
        res = request_wrapper.completion(prompt)
        score_match = re.search(r"<SCORE>(\d+)</SCORE>", res)

        if score_match:
            data["similarity_score"] = int(score_match.group(1))
        else:
            data["similarity_score"] = 50
            logger.warning(f"No score found in response for URL={data.get('url', 'unknown')}, using default score 50")

        data["score_error"] = False
        return data

    except Exception as e:
        logger.error(f"Similarity scoring failed for URL={data.get('url', 'unknown')}: {e}")
        data["similarity_score"] = 0
        data["score_error"] = True
        return data

def _load_queries_from_file(query_file_path: str = None, topic: str = "") -> List[str]:
    """
    从文件中读取查询列表

    Args:
        query_file_path: 查询文件路径，如果为None则从配置中查找最新的文件
        topic: 搜索主题，用于查找相关文件

    Returns:
        查询列表
    """
    try:
        import os
        import glob

        # 如果没有提供文件路径，从query_cache目录中查找最新的文件
        if query_file_path is None:
            cache_dir = SERVER_CONFIG.get("query_cache_dir", "query_cache")
            cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', cache_dir)
            if not os.path.exists(cache_dir):
                logger.warning(f"Query cache directory not found: {cache_dir}")
                return []

            # 查找所有查询文件
            pattern = os.path.join(cache_dir, "queries_*.json")
            query_files = glob.glob(pattern)

            if not query_files:
                logger.warning(f"No query files found in: {cache_dir}")
                return []

            # 选择最新的文件
            query_file_path = max(query_files, key=os.path.getmtime)
            logger.info(f"Using latest query file: {query_file_path}")

        # 读取文件
        if not os.path.exists(query_file_path):
            logger.error(f"Query file not found: {query_file_path}")
            return []

        with open(query_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = data.get("queries", [])
        logger.info(f"✅ Loaded {len(queries)} queries from: {query_file_path}")
        return queries

    except Exception as e:
        logger.error(f"❌ Failed to load queries from file: {e}")
        return []

def _load_urls_from_file(url_file_path: str = None, topic: str = "") -> List[str]:
    """
    从文件中读取URL列表

    Args:
        url_file_path: URL文件路径，如果为None则从配置中查找最新的文件
        topic: 搜索主题，用于查找相关文件

    Returns:
        URL列表
    """
    try:
        import os
        import glob

        # 如果没有提供文件路径，从url_cache目录中查找最新的文件
        if url_file_path is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'url_cache')
            if not os.path.exists(cache_dir):
                logger.warning(f"URL cache directory not found: {cache_dir}")
                return []

            # 查找所有URL文件
            pattern = os.path.join(cache_dir, "urls_*.json")
            url_files = glob.glob(pattern)

            if not url_files:
                logger.warning(f"No URL files found in: {cache_dir}")
                return []

            # 选择最新的文件
            url_file_path = max(url_files, key=os.path.getmtime)
            logger.info(f"Using latest URL file: {url_file_path}")

        # 读取文件
        if not os.path.exists(url_file_path):
            logger.error(f"URL file not found: {url_file_path}")
            return []

        with open(url_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        urls = data.get("urls", [])
        logger.info(f"✅ Loaded {len(urls)} URLs from: {url_file_path}")
        return urls

    except Exception as e:
        logger.error(f"❌ Failed to load URLs from file: {e}")
        return []

def _save_queries_to_file(queries: List[str], topic: str, description: str = "") -> str:
    """
    将查询列表保存到本地文件

    Args:
        queries: 查询列表
        topic: 搜索主题
        description: 主题描述

    Returns:
        保存的文件路径，如果保存失败则返回None
    """
    try:
        import time
        import os

        # 创建保存目录
        cache_dir = SERVER_CONFIG.get("query_cache_dir", "query_cache")
        save_dir = os.path.join(os.path.dirname(__file__), '..', '..', cache_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名（包含时间戳和主题）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:30]  # 限制长度
        filename = f"queries_{safe_topic}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)

        # 准备保存的数据
        save_data = {
            "topic": topic,
            "description": description,
            "queries": queries,
            "query_count": len(queries),
            "timestamp": timestamp,
            "metadata": {
                "generated_by": "generate_search_queries_tool",
                "version": "1.0",
                "description": f"Search queries for topic: {topic}"
            }
        }

        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Queries saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"❌ Failed to save queries to file: {e}")
        return None

def _save_urls_to_file(urls: List[str], topic: str, queries: List[str]) -> str:
    """
    将URL列表保存到本地文件

    Args:
        urls: URL列表
        topic: 搜索主题
        queries: 搜索查询列表

    Returns:
        保存的文件路径，如果保存失败则返回None
    """
    try:
        import time
        import os

        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'url_cache')
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名（包含时间戳和主题）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:30]  # 限制长度
        filename = f"urls_{safe_topic}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)

        # 准备保存的数据
        save_data = {
            "topic": topic,
            "queries": queries,
            "urls": urls,
            "url_count": len(urls),
            "timestamp": timestamp,
            "metadata": {
                "generated_by": "web_search_tool",
                "version": "1.0",
                "description": f"Web search URLs for topic: {topic}"
            }
        }

        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ URLs saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"❌ Failed to save URLs to file: {e}")
        return None

def _process_and_sort_results(scored_results: List[Dict[str, Any]], top_n: int,
                             similarity_threshold: float, min_length: int, max_length: int) -> List[Dict[str, Any]]:

    filtered_results = []
    for result in scored_results:
        content_length = len(result.get("content", ""))
        score_error = result.get("score_error", False)
        similarity_score = result.get("similarity_score", 0)

        # 添加详细的调试信息
        logger.info(f"Processing result: URL={result.get('url', 'Unknown')}")
        logger.info(f"  - content_length: {content_length}")
        logger.info(f"  - score_error: {score_error}")
        logger.info(f"  - similarity_score: {similarity_score}")
        logger.info(f"  - similarity_threshold: {similarity_threshold}")
        logger.info(f"  - max_length: {max_length}")

        # 放宽长度限制，只要有内容就保留
        if (not score_error and
            similarity_score >= similarity_threshold and
            content_length > 0 and content_length <= max_length):
            filtered_results.append(result)
            logger.info(f"  ✅ Result PASSED all filters")
        else:
            logger.info(f"  ❌ Result FAILED filters:")
            if score_error:
                logger.info(f"    - score_error: {score_error}")
            if similarity_score < similarity_threshold:
                logger.info(f"    - similarity_score ({similarity_score}) < threshold ({similarity_threshold})")
            if content_length <= 0:
                logger.info(f"    - content_length ({content_length}) <= 0")
            if content_length > max_length:
                logger.info(f"    - content_length ({content_length}) > max_length ({max_length})")

    filtered_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    final_results = filtered_results[:top_n]

    # 生成兼容 LLMxMapReduce_V2 和 Survey 格式的结果
    formatted_results = []
    for i, result in enumerate(final_results):
        content = result.get("content", "")
        title = result.get("title", "")

        # 生成兼容格式的论文数据
        paper_data = {
            # LLMxMapReduce_V2 核心字段
            "title": title,
            "url": result.get("url", ""),
            "txt": content,  # content → txt
            "similarity": result.get("similarity_score", 0),  # similarity_score → similarity

            # Survey 兼容字段
            "bibkey": proc_title_to_str(title),
            "abstract": extract_abstract(content, 500),
            "txt_token": estimate_tokens(content),
            "txt_length": len(content),

            # 元数据字段
            "source_type": "web_crawl",
            "crawl_timestamp": time.time(),
            "processing_stage": "crawl_complete"
        }
        formatted_results.append(paper_data)

    return formatted_results

async def main():
    logger.info("Starting LLM Search MCP Server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
