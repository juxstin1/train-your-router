"""
Session Tracker - Multi-turn conversation tracking.
Tracks context across turns for better routing decisions.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from core_logger import (
    RoutingLogger, RoutingLog, RequestContext,
    get_logger
)


@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    turn_number: int
    timestamp: str
    request: str
    model_used: str
    task_type: str
    success: bool
    has_image: bool = False
    has_code: bool = False
    tool_calls: List[str] = field(default_factory=list)


@dataclass
class Session:
    """Represents a conversation session"""
    session_id: str
    started_at: str
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_model(self) -> Optional[str]:
        if self.turns:
            return self.turns[-1].model_used
        return None

    @property
    def models_used(self) -> List[str]:
        return list(set(t.model_used for t in self.turns))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turn_count": self.turn_count,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "timestamp": t.timestamp,
                    "request": t.request,
                    "model_used": t.model_used,
                    "task_type": t.task_type,
                    "success": t.success,
                    "has_image": t.has_image,
                    "has_code": t.has_code,
                    "tool_calls": t.tool_calls
                }
                for t in self.turns
            ],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            turns=[
                ConversationTurn(
                    turn_number=t["turn_number"],
                    timestamp=t["timestamp"],
                    request=t["request"],
                    model_used=t["model_used"],
                    task_type=t["task_type"],
                    success=t["success"],
                    has_image=t.get("has_image", False),
                    has_code=t.get("has_code", False),
                    tool_calls=t.get("tool_calls", [])
                )
                for t in data.get("turns", [])
            ],
            metadata=data.get("metadata", {})
        )


class SessionTracker:
    """
    Tracks conversation sessions for context-aware routing.

    Features:
    - Automatic session management
    - Context building for routing decisions
    - Session persistence
    - Multi-turn analysis
    """

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.logger = logger or get_logger()
        self._session = Session(
            session_id=self.logger.session_id,
            started_at=datetime.now().isoformat()
        )

    @property
    def session(self) -> Session:
        return self._session

    @property
    def session_id(self) -> str:
        return self._session.session_id

    def add_turn(self, log: RoutingLog):
        """Add a turn from a routing log"""
        turn = ConversationTurn(
            turn_number=self._session.turn_count + 1,
            timestamp=log.timestamp,
            request=log.request.text,
            model_used=log.routing.model_chosen,
            task_type=log.routing.task_type,
            success=log.execution.success,
            has_image=log.request.has_image,
            has_code=log.request.has_code_block,
            tool_calls=log.execution.tool_calls
        )
        self._session.turns.append(turn)

    def get_context(self) -> RequestContext:
        """
        Build context for the next routing decision.
        Uses conversation history to inform routing.
        """
        return RequestContext(
            conversation_turns=self._session.turn_count,
            previous_model=self._session.last_model,
            available_tools=self._get_available_tools()
        )

    def _get_available_tools(self) -> List[str]:
        """Get tools that have been used in this session"""
        tools = set()
        for turn in self._session.turns:
            tools.update(turn.tool_calls)
        return list(tools)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            "session_id": self.session_id,
            "turn_count": self._session.turn_count,
            "models_used": self._session.models_used,
            "task_types": list(set(t.task_type for t in self._session.turns)),
            "total_tool_calls": sum(len(t.tool_calls) for t in self._session.turns),
            "success_rate": (
                sum(1 for t in self._session.turns if t.success) / self._session.turn_count
                if self._session.turn_count > 0 else 1.0
            )
        }

    def should_stick_with_model(self) -> Optional[str]:
        """
        Determine if we should stick with the current model.
        Returns model name if yes, None if routing should proceed normally.

        Heuristics:
        - If last 3 turns used same model successfully, stick with it
        - If in middle of multi-step workflow, stick with current
        - If current model has tool context, prefer keeping it
        """
        if self._session.turn_count < 2:
            return None

        recent = self._session.turns[-3:] if len(self._session.turns) >= 3 else self._session.turns

        # Check if all recent turns used same model successfully
        models = [t.model_used for t in recent]
        successes = [t.success for t in recent]

        if len(set(models)) == 1 and all(successes):
            # Same model, all successful
            last_turn = self._session.turns[-1]

            # If there were tool calls, we're likely in a workflow
            if last_turn.tool_calls:
                return models[0]

        return None

    def get_recent_requests(self, n: int = 5) -> List[str]:
        """Get recent request texts for context"""
        return [t.request for t in self._session.turns[-n:]]

    def save_session(self, path: Optional[Path] = None):
        """Save session to file"""
        if path is None:
            path = Path(f"logs/sessions/{self.session_id}_summary.json")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._session.to_dict(), f, indent=2)

    def new_session(self):
        """Start a new session"""
        self._session = Session(
            session_id=str(uuid.uuid4())[:8],
            started_at=datetime.now().isoformat()
        )


# ============================================================================
# Context-aware routing helper
# ============================================================================

class ContextAwareRouter:
    """
    Wraps the base router with session context.
    Provides smarter routing based on conversation history.
    """

    def __init__(self, base_router, tracker: Optional[SessionTracker] = None):
        self.router = base_router
        self.tracker = tracker or SessionTracker()

    def route(self, request: str, has_image: bool = False, has_code: bool = False):
        """
        Route with context awareness.
        """
        from model_router import RequestContext as RouterRequestContext

        # Check if we should stick with current model
        sticky_model = self.tracker.should_stick_with_model()

        # Get context
        context = self.tracker.get_context()

        # Build router context
        router_context = RouterRequestContext(
            request=request,
            has_image=has_image,
            has_code=has_code,
            conversation_history=self.tracker.get_recent_requests(),
            available_tools=context.available_tools
        )

        # Get routing decision
        decision = self.router.route(router_context)

        # Apply sticky model if applicable
        if sticky_model and decision.confidence < 0.9:
            # If not highly confident, prefer sticking with current model
            from model_router import ModelID
            try:
                decision.model = ModelID(sticky_model)
                decision.reasoning = f"Continuing with {sticky_model} for workflow continuity"
            except ValueError:
                pass

        return decision


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from core_logger import quick_log

    print("=" * 50)
    print("SESSION TRACKER TEST")
    print("=" * 50)

    tracker = SessionTracker()

    # Simulate a multi-turn conversation
    requests = [
        ("goto amazon.com", "command-r-35b", "browser_automation", ["browser_navigate"]),
        ("search for headphones", "command-r-35b", "browser_automation", ["browser_type", "browser_click"]),
        ("screenshot the results", "command-r-35b", "tool_use", ["browser_screenshot"]),
        ("which one is cheapest?", "command-r-35b", "reasoning", []),
    ]

    print(f"\nSession ID: {tracker.session_id}")
    print("\nSimulating conversation:")

    for request, model, task_type, tools in requests:
        log = quick_log(
            request=request,
            model=model,
            task_type=task_type,
            tools=tools
        )
        tracker.add_turn(log)
        print(f"  Turn {tracker.session.turn_count}: {request[:30]}... -> {model}")

    print("\n" + "-" * 50)
    print("Conversation Summary:")
    summary = tracker.get_conversation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "-" * 50)
    print("Context for next request:")
    ctx = tracker.get_context()
    print(f"  Turns: {ctx.conversation_turns}")
    print(f"  Previous model: {ctx.previous_model}")
    print(f"  Available tools: {ctx.available_tools}")

    print("\n" + "-" * 50)
    sticky = tracker.should_stick_with_model()
    print(f"Should stick with model: {sticky or 'No (route normally)'}")

    # Save session
    tracker.save_session()
    print(f"\nSession saved to logs/sessions/{tracker.session_id}_summary.json")
