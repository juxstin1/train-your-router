"""
Feedback Collection - Quick thumbs up/down after responses.
Non-intrusive, fast, optional.
"""

import sys
import select
from typing import Optional, Tuple
from datetime import datetime

from core_logger import (
    RoutingLogger, RoutingLog, FeedbackData,
    get_logger
)


# ANSI colors for terminal output
class Colors:
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def format_feedback_prompt(log: RoutingLog) -> str:
    """Format the feedback prompt line"""
    model = log.routing.model_chosen
    task = log.routing.task_type
    conf = log.routing.confidence

    # Color code confidence
    if conf >= 0.85:
        conf_color = Colors.GREEN
    elif conf >= 0.70:
        conf_color = Colors.YELLOW
    else:
        conf_color = Colors.RED

    return (
        f"{Colors.GRAY}["
        f"{Colors.BLUE}{model}{Colors.GRAY} | "
        f"{task} | "
        f"{conf_color}{conf:.0%}{Colors.GRAY}] "
        f"{Colors.RESET}"
    )


class FeedbackCollector:
    """
    Collects feedback after model responses.

    Usage:
        collector = FeedbackCollector(logger)

        # After model responds
        feedback = collector.prompt(log_entry)
        # User sees: [command-r-35b | tool_use | 92%] 👍 👎 💬 [Enter=OK]
        # User can press: y/n/c or just Enter to skip
    """

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.logger = logger or get_logger()
        self._last_log: Optional[RoutingLog] = None

    def prompt(
        self,
        log: RoutingLog,
        timeout: float = 3.0,
        auto_skip: bool = True
    ) -> Optional[FeedbackData]:
        """
        Show feedback prompt and collect response.

        Args:
            log: The routing log entry to get feedback for
            timeout: Seconds to wait for input (0 = no timeout)
            auto_skip: If True, treat timeout as implicit OK

        Returns:
            FeedbackData if feedback given, None if skipped
        """
        self._last_log = log

        # Show prompt
        prompt_text = format_feedback_prompt(log)
        sys.stdout.write(f"{prompt_text}")
        sys.stdout.write(f"{Colors.GRAY}[y=👍 n=👎 c=comment Enter=OK]{Colors.RESET} ")
        sys.stdout.flush()

        # Get input
        try:
            if timeout > 0:
                response = self._get_input_with_timeout(timeout)
            else:
                response = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return None

        # Process response
        feedback = self._process_response(response)

        if feedback:
            # Save feedback
            self.logger.update_feedback(log.timestamp, feedback)
            self._show_feedback_confirmation(feedback)

        return feedback

    def _get_input_with_timeout(self, timeout: float) -> str:
        """Get input with timeout (Unix only - falls back to no timeout on Windows)"""
        try:
            # Try Unix select-based timeout
            if sys.platform != 'win32':
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
                if ready:
                    return sys.stdin.readline().strip().lower()
                else:
                    print()  # New line after timeout
                    return ""
            else:
                # Windows - just use regular input (no timeout)
                return input().strip().lower()
        except Exception:
            return input().strip().lower()

    def _process_response(self, response: str) -> Optional[FeedbackData]:
        """Process user response into FeedbackData"""
        if not response or response == "":
            # Enter = implicit OK, no explicit feedback recorded
            return None

        feedback = FeedbackData()

        if response in ("y", "yes", "1", "+", "good"):
            feedback.user_rating = 1  # Thumbs up
            return feedback

        elif response in ("n", "no", "0", "-", "bad", "wrong"):
            feedback.user_rating = -1  # Thumbs down
            # Ask for correct model
            correct = self._ask_correct_model()
            if correct:
                feedback.correct_model = correct
            return feedback

        elif response in ("c", "comment", "note"):
            # Get comment
            note = self._ask_for_comment()
            if note:
                feedback.notes = note
            return feedback

        elif response.startswith("m:") or response.startswith("model:"):
            # Quick model correction: "m:command-r-35b"
            model = response.split(":", 1)[1].strip()
            feedback.user_rating = -1
            feedback.correct_model = model
            return feedback

        return None

    def _ask_correct_model(self) -> Optional[str]:
        """Ask user which model should have been used"""
        print(f"\n{Colors.YELLOW}Which model should have handled this?{Colors.RESET}")
        print("  1. gpt-oss-20b      → tool use, browser, complex reasoning")
        print("  2. qwen3-coder-30b  → code generation, FCPXML, scripting")
        print("  3. ministral-14b    → fast reasoning, math")
        print("  4. qwen3-vl-8b      → vision (images only)")
        print("  5. gemma-3n-e4b     → general chat")
        print("  6. gemma-3-1b       → simple queries")
        print("  7. command-r-35b    → tool orchestration (fallback)")

        try:
            choice = input(f"{Colors.GRAY}Select [1-7]: {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        model_map = {
            "1": "gpt-oss-20b",
            "2": "qwen3-coder-30b",
            "3": "ministral-14b-reasoning",
            "4": "qwen3-vl-8b",
            "5": "gemma-3n-e4b",
            "6": "gemma-3-1b",
            "7": "command-r-35b"
        }

        return model_map.get(choice, choice if choice else None)

    def _ask_for_comment(self) -> Optional[str]:
        """Ask user for a comment"""
        try:
            note = input(f"{Colors.GRAY}Note: {Colors.RESET}").strip()
            return note if note else None
        except (KeyboardInterrupt, EOFError):
            return None

    def _show_feedback_confirmation(self, feedback: FeedbackData):
        """Show brief confirmation of recorded feedback"""
        if feedback.user_rating == 1:
            print(f"{Colors.GREEN}✓ Recorded: correct routing{Colors.RESET}")
        elif feedback.user_rating == -1:
            if feedback.correct_model:
                print(f"{Colors.RED}✗ Recorded: should be {feedback.correct_model}{Colors.RESET}")
            else:
                print(f"{Colors.RED}✗ Recorded: wrong model{Colors.RESET}")
        elif feedback.notes:
            print(f"{Colors.BLUE}📝 Note recorded{Colors.RESET}")

    def quick_thumbs(self, log: RoutingLog, is_correct: bool) -> FeedbackData:
        """
        Programmatic thumbs up/down without prompting.

        Usage:
            # If user manually switched models
            collector.quick_thumbs(log, is_correct=False)
        """
        feedback = FeedbackData(user_rating=1 if is_correct else -1)
        self.logger.update_feedback(log.timestamp, feedback)
        return feedback

    def record_model_switch(
        self,
        log: RoutingLog,
        switched_to: str,
        reason: Optional[str] = None
    ) -> FeedbackData:
        """
        Record when user manually switches to a different model.
        This is implicit negative feedback.
        """
        feedback = FeedbackData(
            user_rating=-1,
            correct_model=switched_to,
            notes=reason or f"User switched from {log.routing.model_chosen} to {switched_to}"
        )
        self.logger.update_feedback(log.timestamp, feedback)

        print(f"{Colors.YELLOW}📊 Recorded: user preferred {switched_to} "
              f"over {log.routing.model_chosen}{Colors.RESET}")

        return feedback


# ============================================================================
# Inline feedback - one-liner for minimal intrusion
# ============================================================================

def inline_feedback(log: RoutingLog, show_prompt: bool = True) -> Optional[FeedbackData]:
    """
    One-liner feedback collection.

    Usage:
        log = quick_log(...)
        inline_feedback(log)  # Shows prompt if in interactive mode
    """
    if not show_prompt:
        return None

    # Only show in interactive terminal
    if not sys.stdout.isatty():
        return None

    collector = FeedbackCollector()
    return collector.prompt(log, timeout=2.0)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from core_logger import RoutingLog, RequestData, RoutingData, quick_log

    print("=" * 50)
    print("FEEDBACK COLLECTOR TEST")
    print("=" * 50)

    # Create a test log entry
    log = quick_log(
        request="goto amazon.com and search for headphones",
        model="command-r-35b",
        task_type="browser_automation",
        confidence=0.92,
        tools=["browser_navigate", "browser_type"]
    )

    # Test feedback collection
    collector = FeedbackCollector()
    print("\nTest 1: Good routing (press 'y' or Enter)")
    collector.prompt(log, timeout=0)

    # Create another log for testing
    log2 = quick_log(
        request="explain quantum computing",
        model="gemma-3-1b",  # Intentionally wrong
        task_type="simple_chat",
        confidence=0.65
    )

    print("\nTest 2: Wrong model (press 'n' and select correct model)")
    collector.prompt(log2, timeout=0)

    print("\n" + "=" * 50)
    print("FEEDBACK TEST COMPLETE")
