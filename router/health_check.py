"""
Health Check System - Verify models and tools are available before routing.

Critical for production:
- Don't route to unloaded models
- Don't route to unavailable tools
- Automatic fallback when things fail
"""

import subprocess
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from enum import Enum
import urllib.request
import urllib.error


# ============================================================================
# Model Health Check
# ============================================================================

@dataclass
class ModelStatus:
    """Status of a model"""
    name: str
    available: bool
    last_checked: float = 0
    error: Optional[str] = None


class ModelHealthCheck:
    """
    Check if models are actually loaded in LM Studio before routing.
    Caches results to avoid spamming the API.
    """

    CACHE_TTL_SECONDS = 30  # Don't check more than once per 30 seconds

    def __init__(self, lmstudio_url: str = "http://localhost:1234/v1"):
        self.lmstudio_url = lmstudio_url
        self._cache: Dict[str, ModelStatus] = {}
        self._last_full_check: float = 0
        self._available_models: Set[str] = set()

    def check_lmstudio_models(self, force: bool = False) -> List[str]:
        """
        GET http://localhost:1234/v1/models
        Returns list of loaded model names.
        Caches for 30 seconds.
        """
        now = time.time()

        # Use cache if recent enough
        if not force and (now - self._last_full_check) < self.CACHE_TTL_SECONDS:
            return list(self._available_models)

        try:
            url = f"{self.lmstudio_url}/models"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

            # LM Studio returns {"data": [{"id": "model-name", ...}, ...]}
            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id:
                    models.append(model_id)

            self._available_models = set(models)
            self._last_full_check = now

            # Update cache
            for model in models:
                self._cache[model] = ModelStatus(
                    name=model,
                    available=True,
                    last_checked=now
                )

            return models

        except urllib.error.URLError as e:
            # LM Studio not running
            self._available_models = set()
            return []
        except Exception as e:
            print(f"Warning: Error checking LM Studio models: {e}")
            return list(self._available_models)  # Return cached

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is loaded.
        Uses fuzzy matching - "command-r-35b" matches "commandr-35b-cpt-sft-i1"
        """
        # Refresh cache if stale
        models = self.check_lmstudio_models()

        # Exact match
        if model_name in self._available_models:
            return True

        # Normalize for comparison (remove hyphens, lowercase)
        def normalize(s):
            return s.lower().replace("-", "").replace("_", "")

        model_norm = normalize(model_name)
        for loaded in self._available_models:
            loaded_norm = normalize(loaded)
            # Check substring match on normalized strings
            if model_norm in loaded_norm or loaded_norm in model_norm:
                return True
            # Also check key identifiers (e.g., "commandr" and "35b")
            model_parts = [p for p in model_name.lower().replace("-", " ").split() if len(p) > 2]
            if all(p in loaded_norm for p in model_parts):
                return True

        return False

    def get_available_model(self, model_name: str) -> Optional[str]:
        """
        Get the actual model name that's loaded (handles version differences).
        Returns None if not available.
        """
        models = self.check_lmstudio_models()

        # Exact match first
        if model_name in self._available_models:
            return model_name

        # Normalize for comparison (remove hyphens, lowercase)
        def normalize(s):
            return s.lower().replace("-", "").replace("_", "")

        model_norm = normalize(model_name)
        for loaded in self._available_models:
            loaded_norm = normalize(loaded)
            # Check substring match on normalized strings
            if model_norm in loaded_norm or loaded_norm in model_norm:
                return loaded
            # Also check key identifiers (e.g., "commandr" and "35b")
            model_parts = [p for p in model_name.lower().replace("-", " ").split() if len(p) > 2]
            if all(p in loaded_norm for p in model_parts):
                return loaded

        return None

    def is_lmstudio_running(self) -> bool:
        """Quick check if LM Studio API is responding"""
        try:
            url = f"{self.lmstudio_url}/models"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except:
            return False


# ============================================================================
# MCP / Tool Health Check
# ============================================================================

class ToolStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class MCPToolStatus:
    """Status of MCP tools"""
    playwright_available: bool = False
    docker_running: bool = False
    browser_tools: List[str] = field(default_factory=list)
    last_checked: float = 0
    error: Optional[str] = None


class MCPHealthCheck:
    """
    Check if MCP tools (especially Playwright) are available.
    """

    CACHE_TTL_SECONDS = 60  # Tools don't change as often

    def __init__(self):
        self._status = MCPToolStatus()
        self._last_check: float = 0

    def check_playwright_docker(self) -> bool:
        """
        Check if Playwright Docker container is running.
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=playwright", "--format", "{{.Names}}"],
                capture_output=True,
                timeout=5
            )
            output = result.stdout.decode().strip()
            return "playwright" in output.lower() or "mcp" in output.lower()
        except FileNotFoundError:
            # Docker not installed
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Warning: Error checking Docker: {e}")
            return False

    def check_mcp_server(self, port: int = 3000) -> bool:
        """
        Check if MCP server is responding on expected port.
        """
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except:
            return False

    def check_tools(self, force: bool = False) -> MCPToolStatus:
        """
        Full check of MCP tool availability.
        """
        now = time.time()

        if not force and (now - self._last_check) < self.CACHE_TTL_SECONDS:
            return self._status

        # Check Docker
        docker_running = self.check_playwright_docker()

        # Check MCP server (try common ports)
        mcp_running = False
        for port in [3000, 8080, 3001]:
            if self.check_mcp_server(port):
                mcp_running = True
                break

        self._status = MCPToolStatus(
            playwright_available=docker_running or mcp_running,
            docker_running=docker_running,
            browser_tools=[
                "browser_navigate", "browser_click", "browser_type",
                "browser_screenshot", "browser_snapshot"
            ] if docker_running else [],
            last_checked=now
        )
        self._last_check = now

        return self._status

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        status = self.check_tools()

        if tool_name.startswith("browser_"):
            return status.playwright_available

        # Other tools assumed available
        return True

    def can_execute_browser_tasks(self) -> bool:
        """Quick check for browser automation capability"""
        status = self.check_tools()
        return status.playwright_available


# ============================================================================
# Fallback Chains
# ============================================================================

FALLBACK_CHAINS: Dict[str, List[Optional[str]]] = {
    "tool_use": [
        "gpt-oss-20b",        # Primary - fast, 128K context, great at tools
        "command-r-35b",      # Fallback 1 - tool orchestration
        "qwen3-coder-30b",    # Fallback 2 - can do tools
    ],
    "browser_automation": [
        "gpt-oss-20b",        # Primary - fast browser automation
        "command-r-35b",      # Fallback 1
        "qwen3-coder-30b",    # Fallback 2
    ],
    "code_generation": [
        "qwen3-coder-30b",    # Primary - code specialist
        "gpt-oss-20b",        # Fallback 1 - can do code
        "command-r-35b",      # Fallback 2
    ],
    "vision": [
        "qwen3-vl-8b",        # Primary - only vision model
        None,                  # No fallback - this is the only vision model
    ],
    "reasoning": [
        "gpt-oss-20b",        # Primary - fast, great reasoning
        "ministral-14b-reasoning",  # Fallback 1 - fast reasoning
        "command-r-35b",      # Fallback 2
    ],
    "math_reasoning": [
        "ministral-14b-reasoning",  # Primary - fast math
        "gpt-oss-20b",        # Fallback 1
        "command-r-35b",      # Fallback 2
    ],
    "complex_reasoning": [
        "gpt-oss-20b",        # Primary - 128K context, fast
        "command-r-35b",      # Fallback 1
        "ministral-14b-reasoning",  # Fallback 2
    ],
    "simple_chat": [
        "gemma-3-1b",         # Primary - ultra fast
        "gemma-3n-e4b",       # Fallback 1
        "gpt-oss-20b",        # Fallback 2
    ],
    "general_chat": [
        "gemma-3n-e4b",       # Primary - fast chat
        "gemma-3-1b",         # Fallback 1
        "ministral-14b-reasoning",  # Fallback 2
        "gpt-oss-20b",        # Fallback 3
    ],
}


class FallbackResolver:
    """
    Resolve fallback models when primary is unavailable.
    """

    def __init__(self, health_check: ModelHealthCheck):
        self.health_check = health_check

    def get_available_model(
        self,
        task_type: str,
        preferred_model: str
    ) -> Optional[str]:
        """
        Get the best available model for a task type.

        1. Try preferred model first
        2. If unavailable, walk the fallback chain
        3. Return None if nothing available
        """
        # Check if preferred is available
        if self.health_check.is_model_available(preferred_model):
            return self.health_check.get_available_model(preferred_model)

        # Get fallback chain for this task type
        chain = FALLBACK_CHAINS.get(task_type, [])

        for fallback in chain:
            if fallback is None:
                # Explicit "no fallback" marker
                return None

            if fallback == preferred_model:
                # Already tried this one
                continue

            if self.health_check.is_model_available(fallback):
                return self.health_check.get_available_model(fallback)

        return None

    def get_fallback_chain(self, task_type: str) -> List[str]:
        """Get the fallback chain for a task type"""
        return [m for m in FALLBACK_CHAINS.get(task_type, []) if m is not None]


# ============================================================================
# Combined Health Checker
# ============================================================================

class SystemHealthCheck:
    """
    Combined health check for the entire routing system.
    """

    def __init__(self, lmstudio_url: str = "http://localhost:1234/v1"):
        self.model_health = ModelHealthCheck(lmstudio_url)
        self.mcp_health = MCPHealthCheck()
        self.fallback_resolver = FallbackResolver(self.model_health)

    def full_check(self) -> Dict[str, any]:
        """Run full system health check"""
        models = self.model_health.check_lmstudio_models(force=True)
        tools = self.mcp_health.check_tools(force=True)

        return {
            "lmstudio_running": self.model_health.is_lmstudio_running(),
            "models_loaded": models,
            "model_count": len(models),
            "playwright_available": tools.playwright_available,
            "docker_running": tools.docker_running,
            "browser_tools": tools.browser_tools,
            "ready_for_routing": len(models) > 0,
            "ready_for_browser": tools.playwright_available and len(models) > 0,
        }

    def get_routing_recommendation(self, task_type: str) -> Dict[str, any]:
        """
        Get recommendation for routing a task type.
        Includes available models and warnings.
        """
        chain = FALLBACK_CHAINS.get(task_type, [])
        available = []
        unavailable = []

        for model in chain:
            if model is None:
                continue
            if self.model_health.is_model_available(model):
                available.append(model)
            else:
                unavailable.append(model)

        # Check tools for browser tasks
        needs_browser = task_type in ["tool_use", "browser_automation"]
        browser_ok = self.mcp_health.can_execute_browser_tasks() if needs_browser else True

        return {
            "task_type": task_type,
            "recommended_model": available[0] if available else None,
            "available_models": available,
            "unavailable_models": unavailable,
            "needs_browser": needs_browser,
            "browser_available": browser_ok,
            "can_execute": len(available) > 0 and (browser_ok or not needs_browser),
            "warnings": self._get_warnings(task_type, available, unavailable, browser_ok)
        }

    def _get_warnings(
        self,
        task_type: str,
        available: List[str],
        unavailable: List[str],
        browser_ok: bool
    ) -> List[str]:
        """Generate warnings for routing decision"""
        warnings = []

        if not available:
            warnings.append(f"No models available for {task_type}")

        if unavailable and available:
            warnings.append(f"Primary model(s) unavailable: {', '.join(unavailable[:2])}")

        if task_type in ["tool_use", "browser_automation"] and not browser_ok:
            warnings.append("Playwright/browser tools not available")

        if task_type == "vision" and not available:
            warnings.append("Vision model (qwen3-vl-8b) not loaded - cannot process images")

        return warnings


# ============================================================================
# Quick helpers
# ============================================================================

_default_health: Optional[SystemHealthCheck] = None


def get_health_checker() -> SystemHealthCheck:
    """Get or create default health checker"""
    global _default_health
    if _default_health is None:
        _default_health = SystemHealthCheck()
    return _default_health


def quick_health_check() -> Dict[str, any]:
    """Quick one-liner for system health"""
    return get_health_checker().full_check()


def is_ready() -> bool:
    """Quick check if system is ready for routing"""
    health = quick_health_check()
    return health["ready_for_routing"]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60)

    health = SystemHealthCheck()
    status = health.full_check()

    print(f"\nLM Studio: {'Running' if status['lmstudio_running'] else 'NOT RUNNING'}")
    print(f"Models loaded: {status['model_count']}")
    if status['models_loaded']:
        for model in status['models_loaded']:
            print(f"  - {model}")

    print(f"\nPlaywright: {'Available' if status['playwright_available'] else 'NOT AVAILABLE'}")
    print(f"Docker: {'Running' if status['docker_running'] else 'Not running'}")

    print(f"\nReady for routing: {'YES' if status['ready_for_routing'] else 'NO'}")
    print(f"Ready for browser: {'YES' if status['ready_for_browser'] else 'NO'}")

    # Test recommendations
    print("\n" + "-" * 40)
    print("ROUTING RECOMMENDATIONS")
    print("-" * 40)

    for task_type in ["tool_use", "code_generation", "vision", "simple_chat"]:
        rec = health.get_routing_recommendation(task_type)
        print(f"\n{task_type}:")
        print(f"  Recommended: {rec['recommended_model'] or 'NONE'}")
        print(f"  Available: {rec['available_models']}")
        print(f"  Can execute: {'YES' if rec['can_execute'] else 'NO'}")
        if rec['warnings']:
            for w in rec['warnings']:
                print(f"  Warning: {w}")
