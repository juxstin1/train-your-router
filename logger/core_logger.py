"""
Core Logging Infrastructure for Model Router
Dead simple JSONL logging - automatic, fast, non-intrusive.
"""

import json
import uuid
import time
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading


# ============================================================================
# Data Classes - Match the spec exactly
# ============================================================================

@dataclass
class RequestContext:
    """Context about the request"""
    conversation_turns: int = 0
    previous_model: Optional[str] = None
    available_tools: List[str] = field(default_factory=list)


@dataclass
class RequestData:
    """The incoming request"""
    text: str
    length: int = 0
    has_image: bool = False
    has_code_block: bool = False
    context: RequestContext = field(default_factory=RequestContext)

    def __post_init__(self):
        self.length = len(self.text)
        # Auto-detect code blocks
        if "```" in self.text or self.text.count("    ") > 2:
            self.has_code_block = True


@dataclass
class RoutingData:
    """The routing decision"""
    task_type: str
    confidence: float
    model_chosen: str
    alternatives: List[str] = field(default_factory=list)
    reasoning: str = ""
    decision_time_ms: float = 0.0


@dataclass
class ExecutionData:
    """Execution results"""
    model_loaded: bool = False
    response_time_ms: float = 0.0
    tokens_generated: int = 0
    tool_calls: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class FeedbackData:
    """User feedback"""
    user_rating: Optional[int] = None  # 1-5 or 1 (thumbs up) / -1 (thumbs down)
    correct_model: Optional[str] = None  # If user says "wrong model"
    notes: Optional[str] = None


@dataclass
class FeatureData:
    """Features extracted for analysis"""
    keywords_matched: List[str] = field(default_factory=list)
    task_indicators: Dict[str, float] = field(default_factory=dict)
    model_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingLog:
    """Complete routing log entry"""
    timestamp: str
    session_id: str
    request: RequestData
    routing: RoutingData
    execution: ExecutionData = field(default_factory=ExecutionData)
    feedback: FeedbackData = field(default_factory=FeedbackData)
    features: FeatureData = field(default_factory=FeatureData)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "request": {
                "text": self.request.text,
                "length": self.request.length,
                "has_image": self.request.has_image,
                "has_code_block": self.request.has_code_block,
                "context": asdict(self.request.context)
            },
            "routing": asdict(self.routing),
            "execution": asdict(self.execution),
            "feedback": asdict(self.feedback),
            "features": asdict(self.features)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingLog":
        """Create from dictionary"""
        return cls(
            timestamp=data["timestamp"],
            session_id=data["session_id"],
            request=RequestData(
                text=data["request"]["text"],
                length=data["request"]["length"],
                has_image=data["request"]["has_image"],
                has_code_block=data["request"]["has_code_block"],
                context=RequestContext(**data["request"]["context"])
            ),
            routing=RoutingData(**data["routing"]),
            execution=ExecutionData(**data["execution"]),
            feedback=FeedbackData(**data["feedback"]),
            features=FeatureData(**data["features"])
        )


# ============================================================================
# Core Logger
# ============================================================================

class RoutingLogger:
    """
    Core logging class - handles all file I/O.
    Thread-safe, automatic file rotation by date.
    """

    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
        self._lock = threading.Lock()
        self._session_id = str(uuid.uuid4())[:8]
        self._write_active_session()

    def _setup_directories(self):
        """Create directory structure"""
        (self.base_dir / "routing").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "routing" / "analytics").mkdir(exist_ok=True)
        (self.base_dir / "sessions").mkdir(exist_ok=True)
        (self.base_dir / "research" / "insights").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "research" / "suggestions").mkdir(exist_ok=True)
        (self.base_dir / "research" / "patterns").mkdir(exist_ok=True)

    def _write_active_session(self):
        """Write current session ID to active.txt"""
        active_file = self.base_dir / "sessions" / "active.txt"
        with open(active_file, 'w') as f:
            f.write(f"{self._session_id}\n{datetime.now().isoformat()}\n")

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_daily_log_path(self) -> Path:
        """Get path for today's log file"""
        today = date.today().isoformat()
        return self.base_dir / "routing" / f"{today}.jsonl"

    def _get_session_log_path(self) -> Path:
        """Get path for session log file"""
        return self.base_dir / "sessions" / f"{self._session_id}.jsonl"

    def log(self, entry: RoutingLog):
        """Log a routing decision - thread-safe"""
        with self._lock:
            data = entry.to_dict()
            line = json.dumps(data, ensure_ascii=False) + '\n'

            # Write to daily log
            daily_path = self._get_daily_log_path()
            with open(daily_path, 'a', encoding='utf-8') as f:
                f.write(line)

            # Write to session log
            session_path = self._get_session_log_path()
            with open(session_path, 'a', encoding='utf-8') as f:
                f.write(line)

    def update_feedback(self, timestamp: str, feedback: FeedbackData):
        """Update feedback for a logged entry (by timestamp)"""
        # This is a simplified implementation - in production you might use SQLite
        # For now, we append feedback as a separate entry with reference
        feedback_entry = {
            "type": "feedback_update",
            "reference_timestamp": timestamp,
            "feedback": asdict(feedback),
            "updated_at": datetime.now().isoformat()
        }
        with self._lock:
            daily_path = self._get_daily_log_path()
            with open(daily_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry) + '\n')

    def read_daily_logs(self, date_str: Optional[str] = None) -> List[RoutingLog]:
        """Read all logs for a specific date (default: today)"""
        if date_str is None:
            date_str = date.today().isoformat()

        log_path = self.base_dir / "routing" / f"{date_str}.jsonl"
        if not log_path.exists():
            return []

        logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Skip feedback updates
                if data.get("type") == "feedback_update":
                    continue
                logs.append(RoutingLog.from_dict(data))

        return logs

    def read_session_logs(self, session_id: Optional[str] = None) -> List[RoutingLog]:
        """Read all logs for a session"""
        if session_id is None:
            session_id = self._session_id

        log_path = self.base_dir / "sessions" / f"{session_id}.jsonl"
        if not log_path.exists():
            return []

        logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("type") == "feedback_update":
                    continue
                logs.append(RoutingLog.from_dict(data))

        return logs

    def get_recent_logs(self, n: int = 10) -> List[RoutingLog]:
        """Get the n most recent logs from current session"""
        logs = self.read_session_logs()
        return logs[-n:] if logs else []


# ============================================================================
# Logging Context Manager - For timing
# ============================================================================

class LoggingContext:
    """
    Context manager for automatic timing and logging.

    Usage:
        logger = RoutingLogger()
        with LoggingContext(logger, request_text, routing_decision) as ctx:
            # Execute model
            response = model.generate(...)
            ctx.set_execution(tokens=100, tool_calls=["browser_navigate"])
        # Automatically logged on exit
    """

    def __init__(
        self,
        logger: RoutingLogger,
        request_text: str,
        routing: RoutingData,
        has_image: bool = False,
        has_code: bool = False,
        context: Optional[RequestContext] = None
    ):
        self.logger = logger
        self.request = RequestData(
            text=request_text,
            has_image=has_image,
            has_code_block=has_code,
            context=context or RequestContext()
        )
        self.routing = routing
        self.execution = ExecutionData()
        self.features = FeatureData()
        self._start_time = None
        self._log_entry = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate response time
        if self._start_time:
            self.execution.response_time_ms = (time.perf_counter() - self._start_time) * 1000

        # Set error if exception occurred
        if exc_type:
            self.execution.success = False
            self.execution.error = str(exc_val)

        # Create and save log entry
        self._log_entry = RoutingLog(
            timestamp=datetime.now().isoformat(),
            session_id=self.logger.session_id,
            request=self.request,
            routing=self.routing,
            execution=self.execution,
            features=self.features
        )
        self.logger.log(self._log_entry)

        return False  # Don't suppress exceptions

    def set_execution(
        self,
        model_loaded: bool = True,
        tokens: int = 0,
        tool_calls: Optional[List[str]] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Set execution details"""
        self.execution.model_loaded = model_loaded
        self.execution.tokens_generated = tokens
        self.execution.tool_calls = tool_calls or []
        self.execution.success = success
        self.execution.error = error

    def set_features(
        self,
        keywords: Optional[List[str]] = None,
        task_indicators: Optional[Dict[str, float]] = None,
        model_scores: Optional[Dict[str, float]] = None
    ):
        """Set feature extraction data"""
        self.features.keywords_matched = keywords or []
        self.features.task_indicators = task_indicators or {}
        self.features.model_scores = model_scores or {}

    @property
    def log_entry(self) -> Optional[RoutingLog]:
        """Get the log entry after context exits"""
        return self._log_entry


# ============================================================================
# Quick logging function - simplest possible interface
# ============================================================================

_default_logger: Optional[RoutingLogger] = None


def get_logger(base_dir: str = "logs") -> RoutingLogger:
    """Get or create default logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = RoutingLogger(base_dir)
    return _default_logger


def quick_log(
    request: str,
    model: str,
    task_type: str,
    confidence: float = 0.8,
    tools: Optional[List[str]] = None,
    success: bool = True,
    decision_time_ms: float = 0.0,
    response_time_ms: float = 0.0,
    tokens: int = 0
):
    """
    Quick one-liner logging for simple cases.

    Usage:
        quick_log("goto google.com", "command-r-35b", "tool_use", tools=["browser_navigate"])
    """
    logger = get_logger()

    entry = RoutingLog(
        timestamp=datetime.now().isoformat(),
        session_id=logger.session_id,
        request=RequestData(text=request),
        routing=RoutingData(
            task_type=task_type,
            confidence=confidence,
            model_chosen=model,
            decision_time_ms=decision_time_ms
        ),
        execution=ExecutionData(
            model_loaded=True,
            response_time_ms=response_time_ms,
            tokens_generated=tokens,
            tool_calls=tools or [],
            success=success
        )
    )

    logger.log(entry)
    return entry


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test the logger
    logger = RoutingLogger(base_dir="../logs")
    print(f"Session ID: {logger.session_id}")

    # Test quick_log
    quick_log(
        request="goto spencers.com and find a beanie",
        model="command-r-35b",
        task_type="browser_automation",
        confidence=0.92,
        tools=["browser_navigate", "browser_type", "browser_click"],
        decision_time_ms=8.5,
        response_time_ms=2340.0,
        tokens=156
    )

    # Test context manager
    routing = RoutingData(
        task_type="code_generation",
        confidence=0.88,
        model_chosen="qwen3-coder-30b",
        reasoning="Code task detected"
    )

    with LoggingContext(logger, "write a python function", routing) as ctx:
        # Simulate model execution
        time.sleep(0.1)
        ctx.set_execution(tokens=50, tool_calls=[])
        ctx.set_features(keywords=["write", "function", "python"])

    # Read back logs
    logs = logger.get_recent_logs(5)
    print(f"\nRecent logs ({len(logs)}):")
    for log in logs:
        print(f"  [{log.routing.model_chosen}] {log.request.text[:40]}...")

    print("\nLogging test complete!")
