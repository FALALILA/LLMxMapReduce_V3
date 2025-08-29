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
from .llm_search_host import LLMSearchHost
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyseInterface:
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
        if not hasattr(self, 'env_config') or self.env_config is None:
            logger.error("env_config not properly loaded! Configuration file is required")
            raise FileNotFoundError("Configuration file config/unified_config.json is required but not found or invalid")

        # 从配置中获取参数
        self.max_interaction_rounds = self.env_config.get("analyse_settings", {}).get("max_interaction_rounds", 3)
        self.llm_model = self.env_config.get("models", {}).get("default_model", "gemini-2.5-flash")
        self.llm_infer_type = self.env_config.get("models", {}).get("default_infer_type", "OpenAI")

        # 初始化记忆系统
        self.search_memory = []  # 存储搜索过程的记忆
        self.conversation_history = []  # 存储完整的对话历史

        # 初始化LLM搜索宿主
        self.llm_search_host = LLMSearchHost()

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
            if hasattr(self, 'llm_search_host') and self.llm_search_host:
                await self.llm_search_host.disconnect()
            self.logger.info("资源清理完成")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")

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

            # 如果没有找到配置文件，抛出错误
            logger.error("Environment config file not found! Please ensure config/unified_config.json exists")
            raise FileNotFoundError("Configuration file config/unified_config.json is required but not found")
        except Exception as e:
            logger.error(f"Failed to load environment config: {e}")
            self.env_config = {}

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

        logger.info(f"AnalyseInterface initialized:")
        logger.info(f"  - Base directory: {self.base_dir}")
        logger.info(f"  - Max interaction rounds: {self.max_interaction_rounds}")
        logger.info(f"  - LLM model: {self.llm_model}")
        logger.info(f"  - Search engine: MCP Client")

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









    def _format_available_tools(self, tools: List[Dict]) -> str:
        """格式化可用工具信息"""
        formatted = []
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")

            # 格式化输入参数
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            params_info = []
            for prop, prop_info in properties.items():
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")
                is_required = prop in required
                req_marker = " (必需)" if is_required else " (可选)"
                params_info.append(f"{prop}: {prop_type}{req_marker} - {prop_desc}")

            params_str = "; ".join(params_info) if params_info else "无参数"
            formatted.append(f"- {name}: {description}\n  参数: {params_str}")

        return "\n".join(formatted)

    async def analyse(self, topic: str, description: Optional[str] = None,
                      top_n: int = 20) -> List[Dict[str, Any]]:
        """
        执行完整的文献分析工作流 - 简化版本，直接调用llm_search_host

        Args:
            topic: 研究主题
            description: 主题描述
            top_n: 返回结果数量
            **kwargs: 其他参数

        Returns:
            分析结果列表
        """
        logger.info(f"Starting literature analysis for topic: '{topic}'")

        # 记忆初始化
        self.conversation_history.clear()
        logger.info("=== 记忆初始化完成 ===")

        try:
            # 第一轮交互：主题扩写
            logger.info("=== 第一轮交互：主题扩写 ===")
            user_msg_1 = f"请扩写主题：{topic}。原始描述：{description or '无'}"
            expanded_topic = await self._llm_interaction_round_1(user_msg_1)
            logger.info(f"主题扩写完成: {expanded_topic[:100]}...")

            # 第二轮：直接执行文献搜索
            logger.info("=== 第二轮：执行文献搜索 ===")

            # 使用llm_search_host执行完整的搜索流程
            results = await self.llm_search_host.search_literature(
                topic=topic,
                description=expanded_topic,
                top_n=top_n
            )

            logger.info(f"✅ Analysis completed successfully. Retrieved {len(results)} papers")

            return results

        except Exception as e:
            logger.error(f"Error in literature analysis: {e}")
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

        # 从配置读取系统提示
        prompts = self.env_config.get("prompts", {})
        system_prompt = prompts.get("analyse_topic_expansion",
            "你是一个专业的学术研究分析专家。用户会给你一个研究主题，请你扩写这个主题，提供详细的研究描述。\n\n请从多个角度进行分析，生成一段专业且全面的描述，这个描述将用于后续的文献搜索。\n\n只返回扩写后的主题描述，不要其他内容。")

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







    def _save_results_to_json(self, results: List[Dict[str, Any]], topic: str) -> str:
        """保存结果到JSON文件"""
        try:
            # 创建test目录（如果不存在）
            test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test')
            os.makedirs(test_dir, exist_ok=True)

            # 生成文件名（包含时间戳和主题）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')[:50]  # 限制长度
            filename = f"literature_results_{safe_topic}_{timestamp}.json"
            filepath = os.path.join(test_dir, filename)

            # 准备保存的数据
            save_data = {
                "topic": topic,
                "timestamp": timestamp,
                "total_results": len(results),
                "results": results,
                "metadata": {
                    "generated_by": "analyse.py",
                    "version": "1.0",
                    "description": f"Literature search results for topic: {topic}"
                }
            }

            # 保存到JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 结果已保存到: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ 保存结果到JSON文件失败: {e}")
            return ""

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



    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            if response.strip().startswith('{'):
                return json.loads(response.strip())

            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())

            brace_pattern = r'\{.*\}'
            match = re.search(brace_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None

    def _create_task_directory(self, task: str) -> Path:

        safe_task_name = "".join(c for c in task if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_task_name = safe_task_name.replace(' ', '_')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir_name = f"{safe_task_name}_{timestamp}"

        task_dir = self.base_dir / task_dir_name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "papers").mkdir(exist_ok=True)
        (task_dir / "analysis").mkdir(exist_ok=True)
        (task_dir / "logs").mkdir(exist_ok=True)

        return task_dir
    
    async def _interactive_refinement(self, expanded_topic: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting interactive refinement (max {self.max_interaction_rounds} rounds)")

        current_analysis = expanded_topic.copy()

        for round_num in range(self.max_interaction_rounds):
            logger.info(f"--- Interaction Round {round_num + 1}/{self.max_interaction_rounds} ---")

            self._display_analysis(current_analysis, round_num + 1)

            user_feedback = self._get_user_feedback(round_num + 1)

            if user_feedback.get('satisfied', False):
                logger.info("User satisfied with current analysis, proceeding to literature search")
                break

            if user_feedback.get('feedback_text'):
                logger.info("Refining analysis based on user feedback")
                current_analysis = await self._refine_with_feedback(current_analysis, user_feedback)
            else:
                logger.info("No specific feedback provided, keeping current analysis")

        logger.info("Interactive refinement completed")
        return current_analysis

    def _display_analysis(self, analysis: Dict[str, Any], round_num: int):
        """展示当前分析结果给用户"""
        print(f"\n{'='*60}")
        print(f"第 {round_num} 轮分析结果")
        print(f"{'='*60}")
        print(f"\n🎯 我对这个topic的分析结果如下：{analysis.get('description', 'N/A')}")
        

    def _get_user_feedback(self, round_num: int) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"请提供第 {round_num} 轮反馈")
        print(f"{'='*60}")

        try:
            satisfied_input = input("\n您是否满意当前的分析结果？(y/n/回车继续): ").strip().lower()

            if satisfied_input in ['y', 'yes', '是', 'ok']:
                return {'satisfied': True}

            if satisfied_input in ['', 'continue', '继续']:
                if round_num >= self.max_interaction_rounds:
                    return {'satisfied': True}
                else:
                    return {'satisfied': False}

            # 获取具体反馈
            print("\n请提供具体的改进建议")
            # print("- 需要调整的概念定义")
            # print("- 需要添加或删除的子领域")
            # print("- 需要补充的研究问题")
            # print("- 其他任何改进意见")
            # print("\n输入您的反馈（回车结束）：")

            feedback_text = input().strip()

            return {
                'satisfied': False,
                'feedback_text': feedback_text,
                'round': round_num
            }

        except KeyboardInterrupt:
            logger.info("User interrupted, proceeding with current analysis")
            return {'satisfied': True}
        except Exception as e:
            logger.warning(f"Error getting user feedback: {e}")
            return {'satisfied': True}





async def analyse(task: str, description: Optional[str] = None, top_n: int = 20) -> List[Dict[str, Any]]:
    analyser = AnalyseInterface()
    return await analyser.analyse(task, description, top_n)


if __name__ == "__main__":
    import sys

    topic = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else None
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print(f"🔍 开始分析主题: {topic}")
    if description:
        print(f"📝 描述: {description}")
    print(f"🎯 目标文献数量: {top_n}")
    print("-" * 50)

    try:
        import asyncio
        literature_results = asyncio.run(analyse(topic, description, top_n))
        print(f"\n✅ 分析完成！检索到 {len(literature_results)} 篇文献")
        print("📁 文献已保存在 llm_search_server 指定的目录中")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
