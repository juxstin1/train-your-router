"""
Logged Model Router - Router with integrated logging, health checks, and feedback.

Features:
- Model health check - verify model loaded in LM Studio before routing
- Fallback chains - automatic fallback if primary unavailable
- MCP tool check - verify Playwright running before tool tasks
- Low confidence prompt - ask user if confidence < 70%
- Live stats - session accuracy in real-time
- Observer mode - lightweight model watches logs and generates insights
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

# Add logger to path
sys.path.insert(0, str(Path(__file__).parent.parent / "logger"))

from model_router import ModelRouter, RequestContext, RoutingDecision, ModelID
from health_check import (
    SystemHealthCheck,
    ModelHealthCheck,
    MCPHealthCheck,
    FallbackResolver,
    FALLBACK_CHAINS,
    get_health_checker,
)

from core_logger import (
    RoutingLogger,
    RoutingLog,
    RequestData,
    RoutingData,
    ExecutionData,
    FeedbackData,
    FeatureData,
    get_logger,
)
from feedback_prompt import FeedbackCollector, inline_feedback
from session_tracker import SessionTracker
from log_observer import RealtimeObserver, ObserverMode


# ============================================================================
# ANSI Colors
# ============================================================================

class Colors:
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RouterConfig:
    """Configuration for the logged router"""
    log_dir: str = "logs"
    lmstudio_url: str = "http://localhost:1234/v1"

    # Feedback settings
    collect_feedback: bool = True
    feedback_timeout: float = 2.0

    # Confidence threshold - prompt user below this
    confidence_threshold: float = 0.70
    prompt_on_low_confidence: bool = True

    # Health check settings
    check_model_health: bool = True
    check_tool_health: bool = True
    auto_fallback: bool = True

    # Warn user about issues
    warn_on_fallback: bool = True
    warn_on_tool_unavailable: bool = True

    # Observer settings
    observer_mode: str = "off"  # "off", "passive", "active"
    observer_model: str = "gemma-3-1b"


# ============================================================================
# Live Stats Tracker
# ============================================================================

class LiveStats:
    """
    Track session statistics in real-time.
    """

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self.by_model: Dict[str, Dict[str, int]] = {}

    def record(
        self,
        model: str,
        task_type: str,
        success: bool,
        user_feedback: Optional[int] = None
    ):
        """Record a request result"""
        self.requests.append({
            "model": model,
            "task_type": task_type,
            "success": success,
            "feedback": user_feedback,
            "timestamp": time.time()
        })

        # Update by-model stats
        if model not in self.by_model:
            self.by_model[model] = {"total": 0, "thumbs_up": 0, "thumbs_down": 0}

        self.by_model[model]["total"] += 1
        if user_feedback == 1:
            self.by_model[model]["thumbs_up"] += 1
        elif user_feedback == -1:
            self.by_model[model]["thumbs_down"] += 1

    def update_feedback(self, feedback: int):
        """Update feedback for the last request"""
        if self.requests:
            self.requests[-1]["feedback"] = feedback
            model = self.requests[-1]["model"]
            if feedback == 1:
                self.by_model[model]["thumbs_up"] += 1
            elif feedback == -1:
                self.by_model[model]["thumbs_down"] += 1

    def get_summary(self, last_n: int = 15) -> Dict[str, Any]:
        """Get summary of recent requests"""
        recent = self.requests[-last_n:] if self.requests else []

        total = len(recent)
        with_feedback = [r for r in recent if r["feedback"] is not None]
        thumbs_up = sum(1 for r in with_feedback if r["feedback"] == 1)
        thumbs_down = sum(1 for r in with_feedback if r["feedback"] == -1)

        return {
            "total_requests": total,
            "with_feedback": len(with_feedback),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "accuracy": thumbs_up / len(with_feedback) if with_feedback else 1.0,
            "by_model": self.by_model.copy()
        }

    def print_dashboard(self, last_n: int = 15):
        """Print a nice dashboard to terminal"""
        summary = self.get_summary(last_n)

        print(f"\n{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}Session Stats (Last {last_n} requests){Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")

        for model, stats in sorted(self.by_model.items(), key=lambda x: x[1]["total"], reverse=True):
            total = stats["total"]
            up = stats["thumbs_up"]
            down = stats["thumbs_down"]
            rated = up + down

            if rated > 0:
                acc = up / rated
                acc_str = f"{acc:.0%}"
                if acc >= 0.9:
                    acc_color = Colors.GREEN
                    status = "✨"
                elif acc >= 0.7:
                    acc_color = Colors.YELLOW
                    status = ""
                else:
                    acc_color = Colors.RED
                    status = "⚠️"
            else:
                acc_str = "N/A"
                acc_color = Colors.GRAY
                status = ""

            print(f"├─ {model}: {total} requests ", end="")
            if rated > 0:
                print(f"({up} 👍, {down} 👎) = {acc_color}{acc_str}{Colors.RESET} {status}")
            else:
                print(f"{Colors.GRAY}(no feedback){Colors.RESET}")

        # Overall
        if summary["with_feedback"] > 0:
            overall_acc = summary["accuracy"]
            if overall_acc >= 0.8:
                overall_color = Colors.GREEN
            elif overall_acc >= 0.6:
                overall_color = Colors.YELLOW
            else:
                overall_color = Colors.RED
            print(f"└─ {Colors.BOLD}Overall: {overall_color}{overall_acc:.0%}{Colors.RESET} accuracy")
        else:
            print(f"└─ {Colors.GRAY}No feedback recorded yet{Colors.RESET}")

        print(f"{Colors.CYAN}{'═' * 50}{Colors.RESET}\n")


# ============================================================================
# Main Router Class
# ============================================================================

class LoggedRouter:
    """
    Model router with integrated logging, health checks, fallbacks, and feedback.

    Usage:
        router = LoggedRouter()

        # Route a request (automatically logged, health-checked, with fallback)
        decision, warnings = router.route("goto amazon.com")

        # After model responds
        router.complete_request(success=True, tokens=100)

        # Collect feedback
        router.collect_feedback()

        # Show stats
        router.show_stats()
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

        # Core components
        self.base_router = ModelRouter(log_decisions=False)
        self.logger = RoutingLogger(base_dir=self.config.log_dir)
        self.feedback_collector = FeedbackCollector(self.logger)
        self.session_tracker = SessionTracker(self.logger)

        # Health checking
        self.health = SystemHealthCheck(self.config.lmstudio_url)

        # Live stats
        self.stats = LiveStats()

        # Observer (AI watching logs)
        self.observer = RealtimeObserver(
            mode=self.config.observer_mode,
            lmstudio_url=self.config.lmstudio_url
        )

        # State
        self._last_log: Optional[RoutingLog] = None
        self._last_decision: Optional[RoutingDecision] = None
        self._last_warnings: List[str] = []
        self._decision_start_time: float = 0

    @property
    def session_id(self) -> str:
        return self.logger.session_id

    def route(
        self,
        request: str,
        has_image: bool = False,
        has_code: bool = False,
        interactive: bool = True
    ) -> Tuple[RoutingDecision, List[str]]:
        """
        Route a request to the optimal model.

        Returns:
            Tuple of (RoutingDecision, list of warnings)

        Steps:
            1. Get initial routing decision
            2. Check model health, apply fallback if needed
            3. Check tool availability for browser tasks
            4. Prompt user if low confidence
            5. Log and return
        """
        self._decision_start_time = time.perf_counter()
        warnings = []

        # Build context with session info
        session_ctx = self.session_tracker.get_context()
        context = RequestContext(
            request=request,
            has_image=has_image,
            has_code=has_code,
            conversation_history=self.session_tracker.get_recent_requests(),
            available_tools=session_ctx.available_tools
        )

        # Get initial routing decision
        decision = self.base_router.route(context)
        original_model = decision.model.value

        # Step 1: Check model health and apply fallback
        if self.config.check_model_health:
            decision, model_warnings = self._apply_health_check(decision)
            warnings.extend(model_warnings)

        # Step 2: Check tool availability for browser tasks
        if self.config.check_tool_health and decision.task_type in ["tool_use", "browser_automation"]:
            tool_warnings = self._check_tool_availability(decision)
            warnings.extend(tool_warnings)

        # Step 3: Handle low confidence
        if (self.config.prompt_on_low_confidence and
            interactive and
            decision.confidence < self.config.confidence_threshold):
            decision = self._prompt_low_confidence(decision, context)

        # Calculate decision time
        decision_time = (time.perf_counter() - self._decision_start_time) * 1000

        # Extract features
        keywords = self._extract_matched_keywords(request)

        # Create log entry
        self._last_log = RoutingLog(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            session_id=self.session_id,
            request=RequestData(
                text=request,
                has_image=has_image,
                has_code_block=has_code,
            ),
            routing=RoutingData(
                task_type=decision.task_type,
                confidence=decision.confidence,
                model_chosen=decision.model.value,
                alternatives=[original_model] if original_model != decision.model.value else [],
                reasoning=decision.reasoning,
                decision_time_ms=decision_time
            ),
            features=FeatureData(
                keywords_matched=keywords,
                task_indicators=self._get_task_indicators(request),
                model_scores={}
            )
        )

        self._last_decision = decision
        self._last_warnings = warnings

        # Show warnings
        if warnings and self.config.warn_on_fallback:
            for w in warnings:
                print(f"{Colors.YELLOW}⚠️  {w}{Colors.RESET}")

        return decision, warnings

    def _apply_health_check(
        self,
        decision: RoutingDecision
    ) -> Tuple[RoutingDecision, List[str]]:
        """Apply health check and fallback logic"""
        warnings = []
        original_model = decision.model.value

        # Check if model is available
        if not self.health.model_health.is_model_available(original_model):
            if self.config.auto_fallback:
                # Try fallback chain
                fallback = self.health.fallback_resolver.get_available_model(
                    decision.task_type,
                    original_model
                )

                if fallback:
                    warnings.append(
                        f"{original_model} unavailable, using fallback: {fallback}"
                    )
                    try:
                        # Update decision with fallback model
                        for model_id in ModelID:
                            if model_id.value == fallback or fallback in model_id.value:
                                decision.model = model_id
                                decision.reasoning = f"[FALLBACK] {decision.reasoning}"
                                break
                    except:
                        pass
                else:
                    warnings.append(
                        f"{original_model} unavailable and no fallback available!"
                    )
            else:
                warnings.append(f"{original_model} may not be loaded in LM Studio")

        return decision, warnings

    def _check_tool_availability(self, decision: RoutingDecision) -> List[str]:
        """Check if required tools are available"""
        warnings = []

        if not self.health.mcp_health.can_execute_browser_tasks():
            warnings.append(
                "Playwright/browser tools not available - browser commands may fail"
            )

            # Optionally downgrade to non-tool task
            if self.config.warn_on_tool_unavailable:
                print(f"\n{Colors.RED}⚠️  Browser tools unavailable!{Colors.RESET}")
                print(f"{Colors.GRAY}   Run: docker start playwright-mcp{Colors.RESET}")

        return warnings

    def _prompt_low_confidence(
        self,
        decision: RoutingDecision,
        context: RequestContext
    ) -> RoutingDecision:
        """Prompt user when confidence is low"""
        print(f"\n{Colors.YELLOW}⚠️  Low confidence ({decision.confidence:.0%}){Colors.RESET}")
        print(f"   Suggested: {Colors.BOLD}{decision.model.value}{Colors.RESET}")
        print(f"   Task type: {decision.task_type}")

        # Get alternatives from fallback chain
        chain = FALLBACK_CHAINS.get(decision.task_type, [])
        alternatives = [m for m in chain if m and m != decision.model.value][:3]

        if alternatives:
            print(f"\n   Alternatives:")
            for i, alt in enumerate(alternatives, 1):
                available = "✓" if self.health.model_health.is_model_available(alt) else "✗"
                print(f"     {i}. {alt} [{available}]")

        print(f"\n   {Colors.GRAY}[Enter=use suggested, 1-{len(alternatives)}=alternative, m=manual]{Colors.RESET}")

        try:
            choice = input("   Choice: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return decision

        if not choice:
            return decision  # Use suggested

        if choice.isdigit() and 1 <= int(choice) <= len(alternatives):
            alt_model = alternatives[int(choice) - 1]
            for model_id in ModelID:
                if model_id.value == alt_model:
                    decision.model = model_id
                    decision.confidence = 1.0  # User chose explicitly
                    decision.reasoning = f"[USER SELECTED] {alt_model}"
                    break

        elif choice == "m":
            # Manual model selection
            print(f"\n   Available models:")
            available = self.health.model_health.check_lmstudio_models()
            for i, model in enumerate(available, 1):
                print(f"     {i}. {model}")

            try:
                num = input("   Select: ").strip()
                if num.isdigit() and 1 <= int(num) <= len(available):
                    selected = available[int(num) - 1]
                    for model_id in ModelID:
                        if model_id.value in selected or selected in model_id.value:
                            decision.model = model_id
                            decision.confidence = 1.0
                            decision.reasoning = f"[USER MANUAL] {selected}"
                            break
            except:
                pass

        return decision

    def complete_request(
        self,
        success: bool = True,
        tokens: int = 0,
        tool_calls: Optional[List[str]] = None,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Call after the model has responded to complete the log entry.
        """
        if not self._last_log:
            return

        # Update execution data
        self._last_log.execution = ExecutionData(
            model_loaded=True,
            response_time_ms=response_time_ms or 0,
            tokens_generated=tokens,
            tool_calls=tool_calls or [],
            success=success,
            error=error
        )

        # Save log
        self.logger.log(self._last_log)

        # Update session tracker
        self.session_tracker.add_turn(self._last_log)

        # Record in live stats
        self.stats.record(
            model=self._last_log.routing.model_chosen,
            task_type=self._last_log.routing.task_type,
            success=success
        )

        # Notify observer that routing is complete
        if self.observer.enabled:
            self.observer.on_routing_complete(
                request=self._last_log.request.text,
                model_chosen=self._last_log.routing.model_chosen,
                task_type=self._last_log.routing.task_type,
                confidence=self._last_log.routing.confidence,
                tools_used=tool_calls or [],
                success=success
            )

    def collect_feedback(self, timeout: Optional[float] = None) -> Optional[int]:
        """
        Prompt for user feedback on the last routing decision.
        Returns: 1 (thumbs up), -1 (thumbs down), or None (skipped)
        """
        if not self._last_log or not self.config.collect_feedback:
            return None

        if not sys.stdout.isatty():
            return None  # Not interactive

        feedback = self.feedback_collector.prompt(
            self._last_log,
            timeout=timeout or self.config.feedback_timeout
        )

        # Update live stats with feedback
        user_rating = None
        corrected_to = None

        if feedback:
            user_rating = feedback.user_rating
            corrected_to = feedback.correct_model
            if user_rating:
                self.stats.update_feedback(user_rating)

        # Trigger observer with feedback
        if self.observer.enabled:
            self.observer.on_feedback_received(
                user_feedback=user_rating,
                corrected_to=corrected_to,
                print_observation=True
            )

        return user_rating

    def record_success(self):
        """Record that the routing was correct"""
        if self._last_log:
            self.feedback_collector.quick_thumbs(self._last_log, is_correct=True)
            self.stats.update_feedback(1)

    def record_failure(self, correct_model: Optional[str] = None):
        """Record that the routing was wrong"""
        if self._last_log:
            if correct_model:
                self.feedback_collector.record_model_switch(
                    self._last_log,
                    switched_to=correct_model
                )
            else:
                self.feedback_collector.quick_thumbs(self._last_log, is_correct=False)
            self.stats.update_feedback(-1)

    def show_stats(self, last_n: int = 15):
        """Show live statistics dashboard"""
        self.stats.print_dashboard(last_n)

        # Show observer stats if enabled
        if self.observer.enabled:
            obs_stats = self.observer.observer.get_stats()
            print(f"{Colors.CYAN}Observer Stats:{Colors.RESET}")
            print(f"  Mode: {obs_stats['mode']}")
            print(f"  Observations: {obs_stats['total_observations']}")
            print(f"  Alerts: {obs_stats['alerts_raised']}")
            print(f"  Patterns: {obs_stats['patterns_detected']}")
            print(f"  High-value data: {obs_stats['high_value_count']}")
            print()

    def show_observer_insights(self):
        """Show recent observer insights"""
        if not self.observer.enabled:
            print(f"{Colors.YELLOW}Observer is disabled. Enable with --observer-active or --observer-passive{Colors.RESET}")
            return

        print(self.observer.get_insights_summary())

    def check_health(self) -> Dict[str, Any]:
        """Run full system health check"""
        return self.health.full_check()

    def show_health(self):
        """Print health status"""
        status = self.health.full_check()

        print(f"\n{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}System Health{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")

        # LM Studio
        if status['lmstudio_running']:
            print(f"├─ LM Studio: {Colors.GREEN}Running{Colors.RESET}")
            print(f"│  Models loaded: {status['model_count']}")
            for model in status['models_loaded'][:5]:
                print(f"│    • {model}")
            if len(status['models_loaded']) > 5:
                print(f"│    ... and {len(status['models_loaded']) - 5} more")
        else:
            print(f"├─ LM Studio: {Colors.RED}NOT RUNNING{Colors.RESET}")

        # Playwright
        if status['playwright_available']:
            print(f"├─ Playwright: {Colors.GREEN}Available{Colors.RESET}")
        else:
            print(f"├─ Playwright: {Colors.RED}NOT AVAILABLE{Colors.RESET}")
            print(f"│  {Colors.GRAY}Run: docker start playwright-mcp{Colors.RESET}")

        # Overall
        if status['ready_for_routing']:
            print(f"└─ Status: {Colors.GREEN}Ready for routing{Colors.RESET}")
        else:
            print(f"└─ Status: {Colors.RED}NOT READY{Colors.RESET}")

        print(f"{Colors.CYAN}{'═' * 50}{Colors.RESET}\n")

    def show_models(self):
        """Print available models with task recommendations"""
        models = self.health.model_health.check_lmstudio_models()

        print(f"\n{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}Available Models{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")

        if models:
            for model in models:
                # Find which tasks this model is good for
                good_for = []
                for task, chain in FALLBACK_CHAINS.items():
                    if chain and chain[0] and model in chain[0]:
                        good_for.append(task)

                print(f"• {model}")
                if good_for:
                    print(f"  {Colors.GRAY}Primary for: {', '.join(good_for)}{Colors.RESET}")
        else:
            print(f"{Colors.RED}No models loaded!{Colors.RESET}")
            print(f"{Colors.GRAY}Start LM Studio and load a model.{Colors.RESET}")

        print(f"{Colors.CYAN}{'═' * 50}{Colors.RESET}\n")

    def _extract_matched_keywords(self, request: str) -> List[str]:
        """Extract keywords that matched routing rules"""
        keywords = []
        request_lower = request.lower()

        trigger_sets = [
            ("tool_use", self.base_router.TOOL_USE_TRIGGERS),
            ("code", self.base_router.CODE_TRIGGERS),
            ("vision", self.base_router.VISION_TRIGGERS),
            ("reasoning", self.base_router.REASONING_TRIGGERS),
            ("math", self.base_router.MATH_TRIGGERS),
            ("simple_chat", self.base_router.SIMPLE_CHAT_TRIGGERS),
        ]

        import re
        for category, triggers in trigger_sets:
            for pattern in triggers:
                match = re.search(pattern, request_lower, re.IGNORECASE)
                if match:
                    keywords.append(f"{category}:{match.group()}")

        return keywords

    def _get_task_indicators(self, request: str) -> dict:
        """Get task type indicator scores"""
        indicators = {}

        tool_score = self.base_router._count_matches(request, self.base_router.TOOL_USE_TRIGGERS)
        code_score = self.base_router._count_matches(request, self.base_router.CODE_TRIGGERS)
        vision_score = self.base_router._count_matches(request, self.base_router.VISION_TRIGGERS)
        reasoning_score = self.base_router._count_matches(request, self.base_router.REASONING_TRIGGERS)
        math_score = self.base_router._count_matches(request, self.base_router.MATH_TRIGGERS)
        chat_score = self.base_router._count_matches(request, self.base_router.SIMPLE_CHAT_TRIGGERS)

        total = max(tool_score + code_score + vision_score + reasoning_score + math_score + chat_score, 1)

        indicators["tool_use"] = tool_score / total if total > 0 else 0
        indicators["code"] = code_score / total if total > 0 else 0
        indicators["vision"] = vision_score / total if total > 0 else 0
        indicators["reasoning"] = reasoning_score / total if total > 0 else 0
        indicators["math"] = math_score / total if total > 0 else 0
        indicators["chat"] = chat_score / total if total > 0 else 0

        return indicators

    def get_session_summary(self) -> dict:
        """Get summary of current session"""
        return self.session_tracker.get_conversation_summary()

    def export_session(self, path: Optional[str] = None):
        """Export session data"""
        self.session_tracker.save_session(Path(path) if path else None)


# ============================================================================
# Quick helpers
# ============================================================================

_default_router: Optional[LoggedRouter] = None


def get_router(config: Optional[RouterConfig] = None) -> LoggedRouter:
    """Get or create default logged router"""
    global _default_router
    if _default_router is None:
        _default_router = LoggedRouter(config)
    return _default_router


def route(request: str, has_image: bool = False, has_code: bool = False) -> Tuple[RoutingDecision, List[str]]:
    """Quick route with logging"""
    router = get_router()
    return router.route(request, has_image, has_code)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LOGGED ROUTER TEST")
    print("=" * 60)

    config = RouterConfig(
        log_dir="../logs",
        collect_feedback=True,
        feedback_timeout=0,  # No timeout for testing
        prompt_on_low_confidence=True,
        confidence_threshold=0.70
    )

    router = LoggedRouter(config)

    # Show health first
    router.show_health()
    router.show_models()

    test_cases = [
        ("goto amazon.com", False, False),
        ("write a python function", False, True),
        ("what's in this image?", True, False),
        ("hello", False, False),
        ("maybe do something with code or not", False, False),  # Low confidence
    ]

    for request, has_image, has_code in test_cases:
        print(f"\n{'='*50}")
        print(f"Request: {request}")

        # Route
        decision, warnings = router.route(request, has_image, has_code)
        print(f"  → Model: {decision.model.value}")
        print(f"  → Type: {decision.task_type}")
        print(f"  → Confidence: {decision.confidence:.0%}")

        # Simulate model response
        router.complete_request(
            success=True,
            tokens=50,
            tool_calls=decision.tools_required,
            response_time_ms=1500
        )

        # Collect feedback
        router.collect_feedback()

    # Show stats
    router.show_stats()

    # Export session
    router.export_session()
    print(f"\nSession exported!")
