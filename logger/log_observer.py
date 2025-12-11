"""
Log Observer - Lightweight model watches logs in real-time and generates insights.

Uses gemma-3-1b (720MB) to:
- Notice patterns in routing decisions and user corrections
- Identify when the router is confused or failing
- Spot edge cases that need special handling
- Suggest improvements based on observations
- Flag high-value training data

Modes:
- PASSIVE: Every 5th request (low overhead, ~50ms every 5 requests)
- ACTIVE: Every request (real-time insights, ~100ms per request)
- OFF: Disabled
"""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from collections import deque


class ObserverMode(Enum):
    OFF = "off"
    PASSIVE = "passive"  # Every 5th request
    ACTIVE = "active"    # Every request


@dataclass
class Observation:
    """A single observation from the observer model"""
    timestamp: str
    observer_model: str
    observation: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    request_context: Optional[str] = None
    alert_level: str = "info"  # info, warning, alert

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObserverContext:
    """Context window for the observer"""
    request: str
    model_chosen: str
    task_type: str
    confidence: float
    user_feedback: Optional[int] = None  # 1, -1, or None
    corrected_to: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    success: bool = True


class LogObserver:
    """
    Lightweight model (gemma-3-1b) watches logs in real-time
    and generates natural language observations.
    """

    SYSTEM_PROMPT = """You are an AI observing a model routing system. Your job is to:

1. Notice patterns in routing decisions and user corrections
2. Identify when the router is confused or failing
3. Spot edge cases that need special handling
4. Suggest improvements based on what you see
5. Flag high-value training data

Be concise (under 50 words). Focus on actionable insights. Flag anomalies immediately.

Context: This is a local LLM routing system with these models:
- command-r-35b: Tool use champion, browser automation
- qwen3-coder-30b: Code specialist
- ministral-14b: Fast reasoning, math (but bad at tools)
- qwen3-vl-8b: Vision only
- gpt-oss-20b: Complex reasoning
- gemma-3n-e4b: Fast chat
- gemma-3-1b: Ultra-fast simple queries

User specializes in video production, FCPXML editing, browser automation, and code generation."""

    def __init__(
        self,
        mode: ObserverMode = ObserverMode.OFF,
        lmstudio_url: str = "http://localhost:1234/v1",
        observer_model: str = "gemma-3-1b",
        context_window_size: int = 10,
        passive_interval: int = 5
    ):
        self.mode = mode
        self.lmstudio_url = lmstudio_url
        self.observer_model = observer_model
        self.context_window: deque = deque(maxlen=context_window_size)
        self.passive_interval = passive_interval
        self.request_count = 0
        self.observations: List[Observation] = []

        # Stats
        self.total_observations = 0
        self.alerts_raised = 0
        self.patterns_detected = 0

    def is_enabled(self) -> bool:
        return self.mode != ObserverMode.OFF

    def should_observe(self) -> bool:
        """Check if we should run observation based on mode"""
        if self.mode == ObserverMode.OFF:
            return False
        if self.mode == ObserverMode.ACTIVE:
            return True
        if self.mode == ObserverMode.PASSIVE:
            return self.request_count % self.passive_interval == 0
        return False

    def add_context(self, ctx: ObserverContext):
        """Add a request to the context window"""
        self.context_window.append(ctx)
        self.request_count += 1

    def observe(self, current: ObserverContext) -> Optional[Observation]:
        """
        Generate an observation about the current state.
        Returns Observation or None if observer is off/skipped.
        """
        if not self.should_observe():
            return None

        # Build prompt
        prompt = self._build_prompt(current)

        # Call observer model
        try:
            response = self._call_lmstudio(prompt)
            if not response:
                return None

            # Parse response and create observation
            observation = self._parse_response(response, current)
            self.observations.append(observation)
            self.total_observations += 1

            # Track alerts
            if observation.alert_level in ["warning", "alert"]:
                self.alerts_raised += 1
            if "pattern" in observation.observation.lower():
                self.patterns_detected += 1

            return observation

        except Exception as e:
            # Observer failure shouldn't break the system
            print(f"Observer error: {e}")
            return None

    def _build_prompt(self, current: ObserverContext) -> str:
        """Build the observation prompt"""
        # Format recent context
        recent_context = ""
        if self.context_window:
            recent_items = list(self.context_window)[-3:]
            for i, ctx in enumerate(recent_items, 1):
                feedback_str = "👍" if ctx.user_feedback == 1 else "👎" if ctx.user_feedback == -1 else "no feedback"
                correction_str = f" → corrected to {ctx.corrected_to}" if ctx.corrected_to else ""
                recent_context += f"{i}. \"{ctx.request[:50]}...\" → {ctx.model_chosen} ({feedback_str}{correction_str})\n"

        # Format current request
        feedback_str = "👍" if current.user_feedback == 1 else "👎" if current.user_feedback == -1 else "no feedback yet"
        correction_str = f"\nUser corrected to: {current.corrected_to}" if current.corrected_to else ""

        prompt = f"""{self.SYSTEM_PROMPT}

Recent requests:
{recent_context if recent_context else "No previous context"}

Current request:
- Request: "{current.request}"
- Routed to: {current.model_chosen}
- Task type: {current.task_type}
- Confidence: {current.confidence:.0%}
- User feedback: {feedback_str}{correction_str}
- Tools used: {current.tools_used if current.tools_used else "none"}
- Success: {current.success}

Analyze this interaction. Write a brief observation (under 50 words) about:
- Is the routing correct for this request?
- Any patterns emerging from recent context?
- Red flags or concerns?
- Is this high-value training data?

Format: [TAGS] observation
Tags: pattern_detected, edge_case, correction_needed, high_value_data, model_drift, red_flag, all_good

Example: [edge_case, high_value_data] User said "screenshot code" but wanted code formatting, not visual analysis. The word "code" should override "screenshot" trigger. Add to training data."""

        return prompt

    def _call_lmstudio(self, prompt: str, timeout: float = 5.0) -> Optional[str]:
        """Call LM Studio API"""
        try:
            url = f"{self.lmstudio_url}/chat/completions"

            data = json.dumps({
                "model": self.observer_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }).encode('utf-8')

            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode())
                return result["choices"][0]["message"]["content"]

        except urllib.error.URLError:
            return None
        except Exception as e:
            print(f"LM Studio call failed: {e}")
            return None

    def _parse_response(self, response: str, current: ObserverContext) -> Observation:
        """Parse the observer model's response into an Observation"""
        # Extract tags
        tags = []
        observation_text = response.strip()

        # Parse [tags] format
        if observation_text.startswith("["):
            bracket_end = observation_text.find("]")
            if bracket_end > 0:
                tags_str = observation_text[1:bracket_end]
                tags = [t.strip() for t in tags_str.split(",")]
                observation_text = observation_text[bracket_end + 1:].strip()

        # Determine alert level
        alert_level = "info"
        if any(t in tags for t in ["red_flag", "model_drift"]):
            alert_level = "alert"
        elif any(t in tags for t in ["correction_needed", "edge_case"]):
            alert_level = "warning"

        # Calculate confidence based on response coherence
        confidence = 0.7
        if len(observation_text) > 20 and len(tags) > 0:
            confidence = 0.85
        if "pattern" in observation_text.lower() or "suggest" in observation_text.lower():
            confidence = 0.9

        return Observation(
            timestamp=datetime.now().isoformat(),
            observer_model=self.observer_model,
            observation=observation_text,
            confidence=confidence,
            tags=tags,
            request_context=current.request[:100],
            alert_level=alert_level
        )

    def get_recent_alerts(self, n: int = 5) -> List[Observation]:
        """Get recent alerts/warnings"""
        alerts = [o for o in self.observations if o.alert_level in ["warning", "alert"]]
        return alerts[-n:]

    def get_patterns_detected(self) -> List[Observation]:
        """Get observations that detected patterns"""
        return [o for o in self.observations if "pattern_detected" in o.tags]

    def get_high_value_data(self) -> List[Observation]:
        """Get observations flagged as high-value training data"""
        return [o for o in self.observations if "high_value_data" in o.tags]

    def get_stats(self) -> Dict[str, Any]:
        """Get observer statistics"""
        return {
            "mode": self.mode.value,
            "total_observations": self.total_observations,
            "alerts_raised": self.alerts_raised,
            "patterns_detected": self.patterns_detected,
            "context_window_size": len(self.context_window),
            "high_value_count": len(self.get_high_value_data())
        }

    def save_observations(self, path: Path):
        """Save all observations to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for obs in self.observations:
                f.write(json.dumps(obs.to_dict()) + '\n')

    def print_observation(self, obs: Observation):
        """Print an observation nicely"""
        # Colors
        GRAY = "\033[90m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # Color based on alert level
        if obs.alert_level == "alert":
            prefix = f"{RED}🚨 ALERT{RESET}"
        elif obs.alert_level == "warning":
            prefix = f"{YELLOW}⚠️  WARNING{RESET}"
        else:
            prefix = f"{CYAN}👁️  Observer{RESET}"

        # Format tags
        tags_str = f"{GRAY}[{', '.join(obs.tags)}]{RESET}" if obs.tags else ""

        print(f"\n{prefix} {tags_str}")
        print(f"   {obs.observation}")


# ============================================================================
# Real-time Observer Wrapper
# ============================================================================

class RealtimeObserver:
    """
    Higher-level wrapper that integrates with the routing system.
    Provides easy hooks for the router to call.
    """

    def __init__(
        self,
        mode: str = "off",
        lmstudio_url: str = "http://localhost:1234/v1"
    ):
        mode_enum = ObserverMode(mode) if mode in ["off", "passive", "active"] else ObserverMode.OFF
        self.observer = LogObserver(
            mode=mode_enum,
            lmstudio_url=lmstudio_url
        )
        self._last_observation: Optional[Observation] = None

    @property
    def enabled(self) -> bool:
        return self.observer.is_enabled()

    @property
    def mode(self) -> str:
        return self.observer.mode.value

    def on_routing_complete(
        self,
        request: str,
        model_chosen: str,
        task_type: str,
        confidence: float,
        tools_used: List[str] = None,
        success: bool = True
    ):
        """
        Call after routing decision is made (before feedback).
        Sets up context for observation.
        """
        ctx = ObserverContext(
            request=request,
            model_chosen=model_chosen,
            task_type=task_type,
            confidence=confidence,
            tools_used=tools_used or [],
            success=success
        )
        self.observer.add_context(ctx)
        self._pending_context = ctx

    def on_feedback_received(
        self,
        user_feedback: Optional[int] = None,
        corrected_to: Optional[str] = None,
        print_observation: bool = True
    ) -> Optional[Observation]:
        """
        Call after user feedback is received.
        Triggers observation if enabled.
        """
        if not hasattr(self, '_pending_context'):
            return None

        # Update context with feedback
        self._pending_context.user_feedback = user_feedback
        self._pending_context.corrected_to = corrected_to

        # Run observation
        obs = self.observer.observe(self._pending_context)

        if obs and print_observation:
            self.observer.print_observation(obs)

        self._last_observation = obs
        return obs

    def get_insights_summary(self) -> str:
        """Get a summary of recent insights"""
        stats = self.observer.get_stats()
        alerts = self.observer.get_recent_alerts(3)
        patterns = self.observer.get_patterns_detected()[-3:]
        high_value = self.observer.get_high_value_data()[-3:]

        summary = f"""
Observer Summary ({stats['total_observations']} observations):
- Alerts raised: {stats['alerts_raised']}
- Patterns detected: {stats['patterns_detected']}
- High-value data flagged: {stats['high_value_count']}
"""

        if alerts:
            summary += "\nRecent Alerts:\n"
            for a in alerts:
                summary += f"  • {a.observation[:80]}...\n"

        if patterns:
            summary += "\nPatterns Detected:\n"
            for p in patterns:
                summary += f"  • {p.observation[:80]}...\n"

        return summary

    def save(self, base_dir: Path):
        """Save observer data"""
        obs_path = base_dir / "research" / "observations" / f"observations_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.observer.save_observations(obs_path)
        return obs_path


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LOG OBSERVER TEST")
    print("=" * 60)

    # Create observer in active mode
    observer = RealtimeObserver(mode="active")

    print(f"Observer mode: {observer.mode}")
    print(f"Enabled: {observer.enabled}")

    # Simulate some routing decisions
    test_scenarios = [
        {
            "request": "goto amazon.com and search for headphones",
            "model": "command-r-35b",
            "task_type": "browser_automation",
            "confidence": 0.92,
            "feedback": 1,  # thumbs up
            "corrected_to": None
        },
        {
            "request": "navigate to github and screenshot",
            "model": "command-r-35b",
            "task_type": "tool_use",
            "confidence": 0.85,
            "feedback": -1,  # thumbs down
            "corrected_to": "qwen3-coder-30b"
        },
        {
            "request": "screenshot this code snippet",
            "model": "qwen3-vl-8b",
            "task_type": "vision",
            "confidence": 0.78,
            "feedback": -1,
            "corrected_to": "qwen3-coder-30b"
        },
        {
            "request": "explain how transformers work",
            "model": "ministral-14b-reasoning",
            "task_type": "reasoning",
            "confidence": 0.75,
            "feedback": 1,
            "corrected_to": None
        },
        {
            "request": "browse to reddit and post something",
            "model": "command-r-35b",
            "task_type": "browser_automation",
            "confidence": 0.90,
            "feedback": -1,
            "corrected_to": "qwen3-coder-30b"
        },
    ]

    print("\nSimulating routing decisions...\n")

    for scenario in test_scenarios:
        print(f"\n{'─' * 40}")
        print(f"Request: {scenario['request']}")
        print(f"Routed to: {scenario['model']}")

        # Simulate routing complete
        observer.on_routing_complete(
            request=scenario['request'],
            model_chosen=scenario['model'],
            task_type=scenario['task_type'],
            confidence=scenario['confidence']
        )

        # Simulate feedback
        obs = observer.on_feedback_received(
            user_feedback=scenario['feedback'],
            corrected_to=scenario['corrected_to']
        )

    # Show summary
    print("\n" + "=" * 60)
    print("OBSERVER SUMMARY")
    print("=" * 60)
    print(observer.get_insights_summary())

    # Show stats
    stats = observer.observer.get_stats()
    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
