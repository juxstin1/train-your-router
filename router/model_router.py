"""
MoE-Style Model Router for Episode Intelligence / Aura Platform
Routes requests to the optimal model based on task type and context.

Model Fleet:
- command-r-35b (20GB) - Tool orchestration champion, 128K context
- qwen3-coder-30b (19GB) - Code generation specialist, 32K context
- ministral-14b-reasoning (9GB) - Fast reasoning, math
- qwen3-vl-8b (6GB) - Vision only
- gpt-oss-20b (12GB) - General chat/reasoning
- gemma-3n-e4b (4GB) - Fast chat
- gemma-3-1b (720MB) - Ultra-fast simple queries
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import time


class ModelID(Enum):
    COMMAND_R_35B = "command-r-35b"
    QWEN3_CODER_30B = "qwen3-coder-30b"
    MINISTRAL_14B = "ministral-14b-reasoning"
    QWEN3_VL_8B = "qwen3-vl-8b"
    GPT_OSS_20B = "gpt-oss-20b"
    GEMMA_3N_E4B = "gemma-3n-e4b"
    GEMMA_3_1B = "gemma-3-1b"


@dataclass
class ModelInfo:
    id: ModelID
    size_gb: float
    context_length: int
    strengths: List[str]
    weaknesses: List[str]
    speed: str  # "ultra-fast", "fast", "medium", "slow"


MODEL_REGISTRY: Dict[ModelID, ModelInfo] = {
    ModelID.COMMAND_R_35B: ModelInfo(
        id=ModelID.COMMAND_R_35B,
        size_gb=20,
        context_length=128000,
        strengths=["tool_use", "browser_automation", "multi_step_workflows", "orchestration"],
        weaknesses=["slow_for_simple_tasks"],
        speed="medium"
    ),
    ModelID.QWEN3_CODER_30B: ModelInfo(
        id=ModelID.QWEN3_CODER_30B,
        size_gb=19,
        context_length=32000,
        strengths=["code_generation", "fcpxml", "scripting", "tool_use"],
        weaknesses=["not_for_chat"],
        speed="medium"
    ),
    ModelID.MINISTRAL_14B: ModelInfo(
        id=ModelID.MINISTRAL_14B,
        size_gb=9,
        context_length=32000,
        strengths=["reasoning", "math", "fast_analysis"],
        weaknesses=["tool_execution", "overthinks", "loops"],
        speed="fast"
    ),
    ModelID.QWEN3_VL_8B: ModelInfo(
        id=ModelID.QWEN3_VL_8B,
        size_gb=6,
        context_length=32000,
        strengths=["vision", "image_analysis", "ocr"],
        weaknesses=["tool_use", "everything_else"],
        speed="fast"
    ),
    ModelID.GPT_OSS_20B: ModelInfo(
        id=ModelID.GPT_OSS_20B,
        size_gb=12,
        context_length=128000,  # Massive context window
        strengths=["tool_use", "browser_automation", "complex_reasoning", "analysis", "long_context"],
        weaknesses=[],  # Fast and capable
        speed="fast"
    ),
    ModelID.GEMMA_3N_E4B: ModelInfo(
        id=ModelID.GEMMA_3N_E4B,
        size_gb=4,
        context_length=8000,
        strengths=["fast_chat", "simple_responses"],
        weaknesses=["complex_tasks", "tool_use", "reasoning"],
        speed="fast"
    ),
    ModelID.GEMMA_3_1B: ModelInfo(
        id=ModelID.GEMMA_3_1B,
        size_gb=0.72,
        context_length=8000,
        strengths=["ultra_fast", "simple_queries", "routing"],
        weaknesses=["everything_complex"],
        speed="ultra-fast"
    ),
}


@dataclass
class RequestContext:
    """Context for routing decisions"""
    request: str
    has_image: bool = False
    has_code: bool = False
    conversation_history: Optional[List[str]] = None
    available_tools: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "context": {
                "has_image": self.has_image,
                "has_code": self.has_code,
                "conversation_history": self.conversation_history or [],
                "available_tools": self.available_tools or []
            }
        }


@dataclass
class RoutingDecision:
    """Result of a routing decision"""
    model: ModelID
    task_type: str
    confidence: float
    reasoning: str
    tools_required: List[str]
    fallback_model: Optional[ModelID] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "task_type": self.task_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "tools_required": self.tools_required,
            "fallback_model": self.fallback_model.value if self.fallback_model else None
        }


class ModelRouter:
    """
    Rules-based router for model selection.
    Phase 1: Keyword matching + heuristics
    Phase 2: Fine-tune gemma-3-1b on collected routing decisions
    """

    # Trigger patterns for each task type
    TOOL_USE_TRIGGERS = [
        r"\bgoto\b", r"\bnavigate\b", r"\bbrowse\b", r"\bscreenshot\b",
        r"\bclick\b", r"\btype\b", r"\bfill\b", r"\bsubmit\b",
        r"\bopen\s+(?:the\s+)?(?:website|page|url|site)\b",
        r"\bgo\s+to\b", r"\bvisit\b", r"\bcheck\s+(?:the\s+)?(?:website|page)\b"
    ]

    CODE_TRIGGERS = [
        r"\bwrite\s+(?:a\s+)?(?:code|function|script|program)\b",
        r"\brefactor\b", r"\bdebug\b", r"\bfunction\b", r"\bclass\b",
        r"\bimplement\b", r"\bcreate\s+(?:a\s+)?(?:script|function)\b",
        r"\bfix\s+(?:this|the)\s+(?:code|bug|error)\b",
        r"\bpython\b", r"\bjavascript\b", r"\btypescript\b",
        r"\bfcpxml\b", r"\bparse\b", r"\bapi\b"
    ]

    VISION_TRIGGERS = [
        r"\bimage\b", r"\bpicture\b", r"\bphoto\b",
        r"\bwhat(?:'s| is)\s+in\s+(?:this|the)\b",
        r"\bdescribe\s+(?:this|the)\s+(?:image|picture|photo)\b",
        r"\bcan\s+you\s+see\b", r"\blook\s+at\b",
        r"\bocr\b", r"\bread\s+(?:the\s+)?text\b"
    ]

    REASONING_TRIGGERS = [
        r"\bexplain\b", r"\banalyze\b", r"\bcompare\b",
        r"\bwhy\b", r"\bhow\s+does\b", r"\bwhat\s+(?:causes|makes)\b",
        r"\bbreak\s+down\b", r"\bstep\s+by\s+step\b",
        r"\breason(?:ing)?\b", r"\bthink\s+through\b"
    ]

    MATH_TRIGGERS = [
        r"\bcalculate\b", r"\bsolve\b", r"\bmath\b",
        r"\bequation\b", r"\bformula\b", r"\bnumber\b",
        r"\d+\s*[\+\-\*\/]\s*\d+", r"\bsum\b", r"\baverage\b"
    ]

    SIMPLE_CHAT_TRIGGERS = [
        r"^hello\b", r"^hi\b", r"^hey\b",
        r"^what(?:'s| is)\s+(?:your|the)\s+(?:name|time|date)\b",
        r"^how\s+are\s+you\b", r"^thanks?\b", r"^bye\b"
    ]

    def __init__(self, log_decisions: bool = True):
        self.log_decisions = log_decisions
        self.decision_log: List[Dict[str, Any]] = []

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern"""
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _count_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match"""
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                count += 1
        return count

    def _estimate_complexity(self, request: str) -> str:
        """Estimate request complexity: simple, medium, complex"""
        word_count = len(request.split())

        if word_count <= 5:
            return "simple"
        elif word_count <= 20:
            return "medium"
        else:
            return "complex"

    def route(self, context: RequestContext) -> RoutingDecision:
        """
        Route a request to the optimal model.
        Returns RoutingDecision with model selection and reasoning.
        """
        request = context.request
        complexity = self._estimate_complexity(request)

        # Priority 1: Vision tasks (if image present)
        if context.has_image or self._matches_any(request, self.VISION_TRIGGERS):
            decision = RoutingDecision(
                model=ModelID.QWEN3_VL_8B,
                task_type="vision",
                confidence=0.9 if context.has_image else 0.7,
                reasoning="Vision task detected - routing to vision specialist",
                tools_required=[],
                fallback_model=ModelID.GPT_OSS_20B
            )
            self._log(context, decision)
            return decision

        # Priority 2: Tool use / Browser automation
        tool_matches = self._count_matches(request, self.TOOL_USE_TRIGGERS)
        if tool_matches > 0:
            tools = self._detect_required_tools(request)
            decision = RoutingDecision(
                model=ModelID.GPT_OSS_20B,
                task_type="tool_use",
                confidence=min(0.95, 0.7 + (tool_matches * 0.1)),
                reasoning=f"Tool use detected ({tool_matches} triggers) - routing to GPT-OSS (fast, 128K context)",
                tools_required=tools,
                fallback_model=ModelID.COMMAND_R_35B
            )
            self._log(context, decision)
            return decision

        # Priority 3: Code generation
        code_matches = self._count_matches(request, self.CODE_TRIGGERS)
        if code_matches > 0 or context.has_code:
            decision = RoutingDecision(
                model=ModelID.QWEN3_CODER_30B,
                task_type="code_generation",
                confidence=min(0.95, 0.7 + (code_matches * 0.1)),
                reasoning=f"Code task detected ({code_matches} triggers) - routing to code specialist",
                tools_required=[],
                fallback_model=ModelID.COMMAND_R_35B
            )
            self._log(context, decision)
            return decision

        # Priority 4: Math / quick reasoning
        if self._matches_any(request, self.MATH_TRIGGERS):
            decision = RoutingDecision(
                model=ModelID.MINISTRAL_14B,
                task_type="math_reasoning",
                confidence=0.8,
                reasoning="Math/calculation task - routing to fast reasoning model",
                tools_required=[],
                fallback_model=ModelID.GPT_OSS_20B
            )
            self._log(context, decision)
            return decision

        # Priority 5: Complex reasoning
        if self._matches_any(request, self.REASONING_TRIGGERS) and complexity == "complex":
            decision = RoutingDecision(
                model=ModelID.GPT_OSS_20B,
                task_type="complex_reasoning",
                confidence=0.75,
                reasoning="Complex reasoning task - using GPT-OSS (128K context, fast)",
                tools_required=[],
                fallback_model=ModelID.COMMAND_R_35B
            )
            self._log(context, decision)
            return decision

        # Priority 6: Medium reasoning
        if self._matches_any(request, self.REASONING_TRIGGERS):
            decision = RoutingDecision(
                model=ModelID.MINISTRAL_14B,
                task_type="reasoning",
                confidence=0.7,
                reasoning="Reasoning task - routing to fast reasoning model",
                tools_required=[],
                fallback_model=ModelID.GPT_OSS_20B
            )
            self._log(context, decision)
            return decision

        # Priority 7: Simple chat
        if self._matches_any(request, self.SIMPLE_CHAT_TRIGGERS) or complexity == "simple":
            decision = RoutingDecision(
                model=ModelID.GEMMA_3_1B,
                task_type="simple_chat",
                confidence=0.85,
                reasoning="Simple query - routing to ultra-fast model",
                tools_required=[],
                fallback_model=ModelID.GEMMA_3N_E4B
            )
            self._log(context, decision)
            return decision

        # Default: Medium complexity general chat
        decision = RoutingDecision(
            model=ModelID.GEMMA_3N_E4B,
            task_type="general_chat",
            confidence=0.6,
            reasoning="General query - routing to fast chat model",
            tools_required=[],
            fallback_model=ModelID.MINISTRAL_14B
        )
        self._log(context, decision)
        return decision

    def _detect_required_tools(self, request: str) -> List[str]:
        """Detect which tools might be required"""
        tools = []
        request_lower = request.lower()

        if any(t in request_lower for t in ["goto", "navigate", "browse", "visit", "open", "go to"]):
            tools.append("browser_navigate")
        if "screenshot" in request_lower:
            tools.append("browser_screenshot")
        if any(t in request_lower for t in ["click", "press", "tap"]):
            tools.append("browser_click")
        if any(t in request_lower for t in ["type", "fill", "enter", "input"]):
            tools.append("browser_type")
        if any(t in request_lower for t in ["search", "find", "look for"]):
            tools.append("browser_type")  # Usually involves typing in search
        if any(t in request_lower for t in ["scroll", "down", "up"]):
            tools.append("browser_scroll")
        if "wait" in request_lower:
            tools.append("browser_wait")

        return tools

    def _log(self, context: RequestContext, decision: RoutingDecision):
        """Log decision for dataset collection"""
        if self.log_decisions:
            entry = {
                "timestamp": time.time(),
                "input": context.to_dict(),
                "output": decision.to_dict()
            }
            self.decision_log.append(entry)

    def export_decisions(self, filepath: str):
        """Export logged decisions to JSONL for training"""
        with open(filepath, 'w') as f:
            for entry in self.decision_log:
                f.write(json.dumps(entry) + '\n')
        print(f"Exported {len(self.decision_log)} decisions to {filepath}")

    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics"""
        stats = {}
        for entry in self.decision_log:
            model = entry["output"]["model"]
            stats[model] = stats.get(model, 0) + 1
        return stats


# Convenience function for quick routing
def route_request(request: str, has_image: bool = False, has_code: bool = False) -> RoutingDecision:
    """Quick routing without context object"""
    router = ModelRouter(log_decisions=False)
    context = RequestContext(request=request, has_image=has_image, has_code=has_code)
    return router.route(context)


if __name__ == "__main__":
    # Test the router
    router = ModelRouter()

    test_cases = [
        "goto spencers.com and find a beanie",
        "take a screenshot of the page",
        "write a python function to parse JSON",
        "what's in this image?",
        "explain how transformers work step by step",
        "calculate 15% of 89.50",
        "hello",
        "refactor this code to use async/await",
        "browse to amazon and search for headphones",
        "debug this function that's throwing an error",
    ]

    print("=" * 60)
    print("MODEL ROUTER TEST")
    print("=" * 60)

    for request in test_cases:
        context = RequestContext(request=request)
        decision = router.route(context)
        print(f"\nRequest: {request}")
        print(f"  -> Model: {decision.model.value}")
        print(f"  -> Type: {decision.task_type}")
        print(f"  -> Confidence: {decision.confidence:.2f}")
        print(f"  -> Tools: {decision.tools_required}")

    print("\n" + "=" * 60)
    print("ROUTING STATS:")
    print(router.get_stats())
