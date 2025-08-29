import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPClient:

    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.session = None
        self.stdio_context = None
        self._connected = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with improved error handling"""
        try:
            await self.disconnect()
        except Exception as e:
            logger.warning(f"Error during context manager exit (this is often normal during shutdown): {e}")
        return False

    async def connect(self):
        """连接到MCP服务器"""
        try:
            if self._connected:
                logger.warning("Already connected to MCP server")
                return

            logger.info("Connecting to MCP Server...")

            # 准备环境变量 - 继承当前进程的关键环境变量以避免Python初始化错误
            env_vars = self.server_config.get("env", {})

            # 从当前进程继承关键的系统环境变量
            env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "TEMP": os.environ.get("TEMP", ""),
                "TMP": os.environ.get("TMP", ""),
                "USERPROFILE": os.environ.get("USERPROFILE", ""),
                "HOMEDRIVE": os.environ.get("HOMEDRIVE", ""),
                "HOMEPATH": os.environ.get("HOMEPATH", ""),
                "COMPUTERNAME": os.environ.get("COMPUTERNAME", ""),
                "USERNAME": os.environ.get("USERNAME", ""),
                "USERDOMAIN": os.environ.get("USERDOMAIN", ""),
            }

            # 添加配置中的环境变量
            if env_vars:
                for k, v in env_vars.items():
                    if v:  # 只包含非空值
                        env[k] = v

            # 移除空值
            env = {k: v for k, v in env.items() if v}
            logger.info(f"Setting environment variables: {list(env.keys())}")

            # 使用StdioServerParameters创建服务器参数
            server_params = StdioServerParameters(
                command=self.server_config["command"],
                args=self.server_config.get("args", []),
                env=env
            )

            # 使用正确的context manager方式连接
            self.stdio_context = stdio_client(server_params)
            read_stream, write_stream = await self.stdio_context.__aenter__()

            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            self._connected = True
            logger.info("Successfully connected to MCP server")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self._connected = False
            raise

    async def disconnect(self):
        """
        Safely disconnect from MCP server with improved error handling
        """
        try:
            if not self._connected:
                return

            logger.info("Starting MCP client disconnect process")

            # Step 1: Clear session reference safely
            if self.session:
                try:
                    # Just clear the reference, don't try to manually manage the context
                    logger.debug("Clearing session reference")
                    self.session = None
                except Exception as e:
                    logger.warning(f"Error clearing session: {e}")
                    self.session = None

            # Step 2: Clear stdio context reference safely
            if self.stdio_context:
                try:
                    # Just clear the reference, don't try to manually manage the context
                    logger.debug("Clearing stdio context reference")
                    self.stdio_context = None
                except Exception as e:
                    logger.warning(f"Error clearing stdio context: {e}")
                    self.stdio_context = None

            # Step 3: Mark as disconnected
            self._connected = False
            logger.info("MCP client disconnected successfully")

        except Exception as e:
            logger.error(f"Error during MCP client disconnect: {e}")
            # Force disconnection even if there were errors
            self._connected = False
            self.session = None
            self.stdio_context = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出可用的工具"""
        try:
            if not self._connected or not self.session:
                raise RuntimeError("Not connected to MCP server")

            tools = await self.session.list_tools()
            logger.debug(f"Available tools: {[tool.name for tool in tools.tools]}")

            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools.tools
            ]

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self._connected or not self.session:
                raise RuntimeError("Not connected to MCP server")

            logger.debug(f"Calling tool: {tool_name} with arguments: {arguments}")

            # 根据文档，直接调用session.call_tool
            result = await self.session.call_tool(tool_name, arguments)

            if result.content and len(result.content) > 0:
                content = result.content[0]
                if isinstance(content, TextContent):
                    return json.loads(content.text)

            raise ValueError(f"No valid response received from tool {tool_name}")

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        try:
            if not self._connected or not self.session:
                raise RuntimeError("Not connected to MCP server")

            resources = await self.session.list_resources()
            logger.debug(f"Available resources: {[resource.uri for resource in resources.resources]}")

            return [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                }
                for resource in resources.resources
            ]

        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            raise

    async def read_resource(self, uri: str) -> str:
        try:
            if not self._connected or not self.session:
                raise RuntimeError("Not connected to MCP server")

            result = await self.session.read_resource(uri)
            return result.contents[0].text if result.contents else ""

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise

    @property
    def is_connected(self) -> bool:
        return self._connected

async def create_mcp_client(server_config: Dict[str, Any]) -> MCPClient:

    try:
        client = MCPClient(server_config)
        await client.connect()
        return client

    except Exception as e:
        logger.error(f"Failed to create MCP client: {e}")
        raise

async def create_mcp_client_from_config(config_path: str = "config/llm_search_mcp_config.json",
                                       server_name: str = "llm_search_mcp") -> MCPClient:
    try:
        # 确保使用绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.abspath("."), config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 修正配置路径，使用mcpServers而不是servers
        server_config = config["mcpServers"][server_name]
        return await create_mcp_client(server_config)

    except Exception as e:
        logger.error(f"Failed to create MCP client from config: {e}")
        raise

async def example_usage():
    """示例用法"""
    client = None
    try:

        client = await create_mcp_client_from_config()

        tools = await client.list_tools()
        print("Available tools:", [tool["name"] for tool in tools])

        result = await client.call_tool(
            "generate_search_queries",
            {
                "topic": "machine learning optimization",
                "description": "Research on optimization techniques in machine learning"
            }
        )

        print("Tool result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"Example usage failed: {e}")
    finally:
        if client:
            await client.disconnect()

if __name__ == "__main__":
    asyncio.run(example_usage())
