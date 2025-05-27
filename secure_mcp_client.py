"""
Secure MCP Client with built-in authentication and authorization.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from auth_system import auth_system, require_auth, Permission, User
from mcp_inspector import mcp_inspector, log_tool_call
from durable_objects_agent import durable_manager, with_durable_state

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    user: User
    session_id: str
    agent_id: Optional[str] = None

class SecureMCPClient:
    """Secure MCP Client with authentication and Durable Objects integration."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_permissions: Dict[str, Permission] = {}
        self.register_cloudflare_tools()

    def register_tool(self, name: str, func: Callable, permission: Permission = None):
        """Register a tool with optional permission requirement."""
        self.tools[name] = func
        if permission:
            self.tool_permissions[name] = permission

    def register_cloudflare_tools(self):
        """Register Cloudflare MCP tools with appropriate permissions."""

        # Workers tools
        self.register_tool("workers_list", self.workers_list, Permission.READ_WORKERS)
        self.register_tool("workers_get_worker", self.workers_get_worker, Permission.READ_WORKERS)
        self.register_tool("workers_get_worker_code", self.workers_get_worker_code, Permission.READ_WORKERS)

        # KV tools
        self.register_tool("kv_namespaces_list", self.kv_namespaces_list, Permission.READ_KV)
        self.register_tool("kv_namespace_create", self.kv_namespace_create, Permission.WRITE_KV)
        self.register_tool("kv_namespace_delete", self.kv_namespace_delete, Permission.WRITE_KV)
        self.register_tool("kv_namespace_get", self.kv_namespace_get, Permission.READ_KV)
        self.register_tool("kv_namespace_update", self.kv_namespace_update, Permission.WRITE_KV)

        # R2 tools
        self.register_tool("r2_buckets_list", self.r2_buckets_list, Permission.READ_R2)
        self.register_tool("r2_bucket_create", self.r2_bucket_create, Permission.WRITE_R2)
        self.register_tool("r2_bucket_get", self.r2_bucket_get, Permission.READ_R2)
        self.register_tool("r2_bucket_delete", self.r2_bucket_delete, Permission.WRITE_R2)

        # D1 tools
        self.register_tool("d1_databases_list", self.d1_databases_list, Permission.READ_D1)
        self.register_tool("d1_database_create", self.d1_database_create, Permission.WRITE_D1)
        self.register_tool("d1_database_get", self.d1_database_get, Permission.READ_D1)
        self.register_tool("d1_database_delete", self.d1_database_delete, Permission.WRITE_D1)
        self.register_tool("d1_database_query", self.d1_database_query, Permission.WRITE_D1)

        # Observability tools
        self.register_tool("query_worker_observability", self.query_worker_observability, Permission.READ_ANALYTICS)
        self.register_tool("observability_keys", self.observability_keys, Permission.READ_ANALYTICS)
        self.register_tool("observability_values", self.observability_values, Permission.READ_ANALYTICS)

        # Agent management tools
        self.register_tool("create_agent", self.create_agent)
        self.register_tool("get_agent_info", self.get_agent_info)
        self.register_tool("list_user_agents", self.list_user_agents)
        self.register_tool("terminate_agent", self.terminate_agent)

    async def call_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Call a tool with authentication and authorization checks."""

        # Log connection if new session
        mcp_inspector.log_connection_opened(
            session_id=tool_call.session_id,
            user_id=tool_call.user.user_id,
            metadata={"tool_call": tool_call.tool_name}
        )

        try:
            # Check if tool exists
            if tool_call.tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_call.tool_name}' not found")

            # Check permissions
            required_permission = self.tool_permissions.get(tool_call.tool_name)
            if required_permission and not auth_system.check_permission(tool_call.user, required_permission):
                raise PermissionError(f"Missing permission: {required_permission.value}")

            # Check tool-specific access
            if not auth_system.check_tool_access(tool_call.user, tool_call.tool_name):
                raise PermissionError("Tool access denied")

            # Log successful auth check
            mcp_inspector.log_auth_check(
                session_id=tool_call.session_id,
                user_id=tool_call.user.user_id,
                tool_name=tool_call.tool_name,
                success=True
            )

            # Log tool call
            mcp_inspector.log_tool_call(
                session_id=tool_call.session_id,
                tool_name=tool_call.tool_name,
                parameters=tool_call.parameters,
                user_id=tool_call.user.user_id
            )

            # Prepare parameters with auth info
            params = tool_call.parameters.copy()
            params.update({
                "authenticated_user": tool_call.user,
                "session_id": tool_call.session_id,
                "api_key": tool_call.user.api_key
            })

            if tool_call.agent_id:
                params["agent_id"] = tool_call.agent_id

            # Call the tool
            tool_func = self.tools[tool_call.tool_name]
            result = await tool_func(**params)

            # Log successful result
            mcp_inspector.log_tool_result(
                session_id=tool_call.session_id,
                tool_name=tool_call.tool_name,
                result={"success": True, "data": result},
                user_id=tool_call.user.user_id
            )

            return {"success": True, "result": result}

        except Exception as e:
            # Log error
            mcp_inspector.log_error(
                session_id=tool_call.session_id,
                error=str(e),
                tool_name=tool_call.tool_name,
                user_id=tool_call.user.user_id
            )

            return {"success": False, "error": str(e)}

    # Cloudflare Workers tools
    @log_tool_call("workers_list")
    @require_auth(Permission.READ_WORKERS)
    async def workers_list(self, **kwargs) -> Dict[str, Any]:
        """List Cloudflare Workers."""
        # Simulate API call to Cloudflare
        return {
            "workers": [
                {"name": "keyboss-electric-production", "id": {"tag": "7461493fec7d42fb8a9ac40205b0b4a1"}},
                {"name": "3d-marketplace-app", "id": {"tag": "6562df6f70cd49e9aa4de2b050021168"}},
                {"name": "marketplace-worker", "id": {"tag": "1f0c703fdfed42e683ee47c28bb1ba93"}},
                {"name": "keyboss-worker", "id": {"tag": "687fcb04074048ab9ddb4393fe3e1799"}},
                {"name": "keyboss-electric", "id": {"tag": "3bd319ef4ad44340b1dd7e5b30237f36"}}
            ],
            "count": 5
        }

    @log_tool_call("workers_get_worker")
    @require_auth(Permission.READ_WORKERS)
    async def workers_get_worker(self, scriptName: str, **kwargs) -> Dict[str, Any]:
        """Get details of a specific worker."""
        return {
            "name": scriptName,
            "id": {"tag": f"worker_{hash(scriptName)}"},
            "created_on": "2025-05-25T14:03:23.426912Z",
            "modified_on": "2025-05-25T15:04:26.879324Z",
            "deployment_id": f"deployment_{hash(scriptName)}"
        }

    @log_tool_call("workers_get_worker_code")
    @require_auth(Permission.READ_WORKERS)
    async def workers_get_worker_code(self, scriptName: str, **kwargs) -> Dict[str, Any]:
        """Get worker source code."""
        return {
            "script": f"// Worker code for {scriptName}\nexport default {{\n  async fetch(request) {{\n    return new Response('Hello from {scriptName}!');\n  }}\n}}",
            "metadata": {"main_module": "index.js"}
        }

    # KV tools
    @log_tool_call("kv_namespaces_list")
    @require_auth(Permission.READ_KV)
    async def kv_namespaces_list(self, **kwargs) -> Dict[str, Any]:
        """List KV namespaces."""
        return {
            "namespaces": [
                {"id": "066a23ba8a0243a99ca04902385cfb12", "title": "emerging-tech-kv"},
                {"id": "17e56de306f642f9add04cae75b6d3d5", "title": "admin_dashboard_tokens"},
                {"id": "2a5ea865be8246d2ad9980f295931352", "title": "keyboss_kv"},
                {"id": "3d5f7712807c403a93c41f0dfd6401d4", "title": "3d-marketplace-kv"},
                {"id": "9711cfa19bd04ce4afbd8b28bd051f7b", "title": "marketplace-kv"}
            ],
            "count": 5
        }

    @log_tool_call("kv_namespace_create")
    @require_auth(Permission.WRITE_KV)
    async def kv_namespace_create(self, title: str, **kwargs) -> Dict[str, Any]:
        """Create a new KV namespace."""
        return {
            "success": True,
            "namespace": {
                "id": f"new_kv_{hash(title)}",
                "title": title,
                "created_at": datetime.utcnow().isoformat()
            }
        }

    # Agent management tools
    @log_tool_call("create_agent")
    @with_durable_state("agent_id")
    async def create_agent(self, agent_type: str, configuration: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Create a new agent instance."""
        user = kwargs["authenticated_user"]
        session_id = kwargs["session_id"]

        agent_id = await durable_manager.create_agent(
            user_id=user.user_id,
            session_id=session_id,
            agent_type=agent_type,
            configuration=configuration or {}
        )

        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "user_id": user.user_id,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat()
        }

    @log_tool_call("get_agent_info")
    @with_durable_state()
    async def get_agent_info(self, agent_obj, **kwargs) -> Dict[str, Any]:
        """Get agent information."""
        return await agent_obj.get_agent_info()

    @log_tool_call("list_user_agents")
    async def list_user_agents(self, **kwargs) -> Dict[str, Any]:
        """List all agents for the authenticated user."""
        user = kwargs["authenticated_user"]
        agents = await durable_manager.list_user_agents(user.user_id)
        return {"agents": agents, "count": len(agents)}

    @log_tool_call("terminate_agent")
    @with_durable_state()
    async def terminate_agent(self, agent_obj, **kwargs) -> Dict[str, Any]:
        """Terminate an agent."""
        success = await agent_obj.terminate_agent()
        return {"success": success, "message": "Agent terminated" if success else "Failed to terminate agent"}

    # Placeholder implementations for other tools
    async def r2_buckets_list(self, **kwargs): return {"buckets": [], "count": 0}
    async def r2_bucket_create(self, **kwargs): return {"success": True}
    async def r2_bucket_get(self, **kwargs): return {"bucket": {}}
    async def r2_bucket_delete(self, **kwargs): return {"success": True}
    async def d1_databases_list(self, **kwargs): return {"result": [], "result_info": {"count": 0}}
    async def d1_database_create(self, **kwargs): return {"success": True}
    async def d1_database_get(self, **kwargs): return {"database": {}}
    async def d1_database_delete(self, **kwargs): return {"success": True}
    async def d1_database_query(self, **kwargs): return {"results": []}
    async def query_worker_observability(self, **kwargs): return {"data": []}
    async def observability_keys(self, **kwargs): return {"keys": []}
    async def observability_values(self, **kwargs): return {"values": []}
    async def kv_namespace_get(self, **kwargs): return {"namespace": {}}
    async def kv_namespace_update(self, **kwargs): return {"success": True}
    async def kv_namespace_delete(self, **kwargs): return {"success": True}

# Global secure MCP client
secure_mcp_client = SecureMCPClient()
