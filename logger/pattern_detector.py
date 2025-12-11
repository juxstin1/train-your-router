"""
Pattern Detector - Find misrouting patterns and suggest improvements.
Analyzes logs to discover what's working and what's not.
"""

import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core_logger import RoutingLogger, RoutingLog, FeedbackData, get_logger


@dataclass
class MisroutingPattern:
    """A detected pattern of misrouting"""
    pattern_type: str  # "keyword", "task_type", "model_confusion"
    description: str
    occurrences: int
    examples: List[str]
    suggested_fix: str
    impact_score: float  # 0-1, how much this affects accuracy
    confidence: float


@dataclass
class KeywordSuggestion:
    """Suggested keyword change"""
    action: str  # "add", "remove", "move"
    keyword: str
    from_category: Optional[str]
    to_category: str
    reason: str
    supporting_examples: List[str]


@dataclass
class PatternReport:
    """Complete pattern analysis report"""
    analyzed_logs: int
    date_range: Tuple[str, str]
    patterns: List[MisroutingPattern]
    keyword_suggestions: List[KeywordSuggestion]
    model_performance: Dict[str, Dict[str, float]]
    edge_cases: List[Dict[str, Any]]


class PatternDetector:
    """
    Analyzes routing logs to find patterns and suggest improvements.
    """

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.logger = logger or get_logger()
        self.patterns: List[MisroutingPattern] = []
        self.suggestions: List[KeywordSuggestion] = []

    def analyze(
        self,
        logs: Optional[List[RoutingLog]] = None,
        days_back: int = 7
    ) -> PatternReport:
        """
        Run full pattern analysis.
        """
        if logs is None:
            logs = self._load_logs(days_back)

        if not logs:
            return PatternReport(
                analyzed_logs=0,
                date_range=("", ""),
                patterns=[],
                keyword_suggestions=[],
                model_performance={},
                edge_cases=[]
            )

        # Run all detectors
        misrouting_patterns = self.detect_misrouting_patterns(logs)
        keyword_suggestions = self.suggest_keyword_updates(logs)
        edge_cases = self.identify_edge_cases(logs)
        model_perf = self.analyze_model_performance(logs)

        # Build report
        timestamps = [log.timestamp for log in logs]
        date_range = (min(timestamps)[:10], max(timestamps)[:10])

        return PatternReport(
            analyzed_logs=len(logs),
            date_range=date_range,
            patterns=misrouting_patterns,
            keyword_suggestions=keyword_suggestions,
            model_performance=model_perf,
            edge_cases=edge_cases
        )

    def _load_logs(self, days_back: int) -> List[RoutingLog]:
        """Load logs from the past N days"""
        all_logs = []
        today = date.today()

        for i in range(days_back):
            day = today - timedelta(days=i)
            day_logs = self.logger.read_daily_logs(day.isoformat())
            all_logs.extend(day_logs)

        return all_logs

    def detect_misrouting_patterns(self, logs: List[RoutingLog]) -> List[MisroutingPattern]:
        """
        Find patterns in misrouted requests.

        Looks for:
        - Requests with certain keywords going to wrong models
        - Task types being confused
        - Specific model pairs that get swapped
        """
        patterns = []

        # Get feedback data to identify corrections
        corrections = self._extract_corrections(logs)

        if not corrections:
            return patterns

        # Pattern 1: Keyword-based misrouting
        keyword_patterns = self._analyze_keyword_misrouting(corrections)
        patterns.extend(keyword_patterns)

        # Pattern 2: Task type confusion
        task_patterns = self._analyze_task_confusion(corrections)
        patterns.extend(task_patterns)

        # Pattern 3: Model confusion pairs
        model_patterns = self._analyze_model_confusion(corrections)
        patterns.extend(model_patterns)

        return patterns

    def _extract_corrections(self, logs: List[RoutingLog]) -> List[Dict[str, Any]]:
        """Extract logs where user corrected the routing"""
        corrections = []

        for log in logs:
            if log.feedback.user_rating == -1 or log.feedback.correct_model:
                corrections.append({
                    "request": log.request.text,
                    "routed_to": log.routing.model_chosen,
                    "correct_model": log.feedback.correct_model,
                    "task_type": log.routing.task_type,
                    "confidence": log.routing.confidence,
                    "keywords": log.features.keywords_matched,
                    "notes": log.feedback.notes
                })

        return corrections

    def _analyze_keyword_misrouting(
        self,
        corrections: List[Dict[str, Any]]
    ) -> List[MisroutingPattern]:
        """Find keywords that consistently lead to wrong routing"""
        patterns = []

        # Extract keywords from corrected requests
        keyword_to_corrections = defaultdict(list)

        for corr in corrections:
            # Extract keywords from request
            words = re.findall(r'\b\w+\b', corr["request"].lower())
            for word in words:
                keyword_to_corrections[word].append(corr)

        # Find keywords with multiple corrections
        for keyword, corrs in keyword_to_corrections.items():
            if len(corrs) >= 2:
                # Check if they're consistently going to wrong model
                wrong_models = Counter(c["routed_to"] for c in corrs)
                correct_models = Counter(c["correct_model"] for c in corrs if c["correct_model"])

                if correct_models:
                    most_common_correct = correct_models.most_common(1)[0][0]
                    most_common_wrong = wrong_models.most_common(1)[0][0]

                    if most_common_correct != most_common_wrong:
                        pattern = MisroutingPattern(
                            pattern_type="keyword",
                            description=f"'{keyword}' routes to {most_common_wrong} but should go to {most_common_correct}",
                            occurrences=len(corrs),
                            examples=[c["request"] for c in corrs[:3]],
                            suggested_fix=f"Add '{keyword}' to {most_common_correct} routing keywords",
                            impact_score=len(corrs) / len(corrections),
                            confidence=len(corrs) / max(len(corrections), 1)
                        )
                        patterns.append(pattern)

        return patterns

    def _analyze_task_confusion(
        self,
        corrections: List[Dict[str, Any]]
    ) -> List[MisroutingPattern]:
        """Find task types that get confused"""
        patterns = []

        # Group by task type
        task_corrections = defaultdict(list)
        for corr in corrections:
            task_corrections[corr["task_type"]].append(corr)

        for task_type, corrs in task_corrections.items():
            if len(corrs) >= 2:
                # What models should these have gone to?
                correct_models = Counter(c["correct_model"] for c in corrs if c["correct_model"])

                if correct_models:
                    most_common = correct_models.most_common(1)[0]
                    pattern = MisroutingPattern(
                        pattern_type="task_type",
                        description=f"Task type '{task_type}' often miscategorized, should usually go to {most_common[0]}",
                        occurrences=len(corrs),
                        examples=[c["request"] for c in corrs[:3]],
                        suggested_fix=f"Review {task_type} classification rules",
                        impact_score=len(corrs) / len(corrections),
                        confidence=most_common[1] / len(corrs)
                    )
                    patterns.append(pattern)

        return patterns

    def _analyze_model_confusion(
        self,
        corrections: List[Dict[str, Any]]
    ) -> List[MisroutingPattern]:
        """Find model pairs that get confused"""
        patterns = []

        # Count model swaps
        swaps = Counter()
        swap_examples = defaultdict(list)

        for corr in corrections:
            if corr["correct_model"]:
                swap = (corr["routed_to"], corr["correct_model"])
                swaps[swap] += 1
                swap_examples[swap].append(corr["request"])

        for (wrong, right), count in swaps.most_common(5):
            if count >= 2:
                pattern = MisroutingPattern(
                    pattern_type="model_confusion",
                    description=f"Requests meant for {right} often go to {wrong}",
                    occurrences=count,
                    examples=swap_examples[(wrong, right)][:3],
                    suggested_fix=f"Strengthen distinction between {wrong} and {right}",
                    impact_score=count / len(corrections),
                    confidence=0.8
                )
                patterns.append(pattern)

        return patterns

    def suggest_keyword_updates(self, logs: List[RoutingLog]) -> List[KeywordSuggestion]:
        """
        Suggest keyword changes based on patterns.
        """
        suggestions = []
        corrections = self._extract_corrections(logs)

        if not corrections:
            return suggestions

        # Analyze which keywords should move where
        keyword_destinations = defaultdict(lambda: defaultdict(int))

        for corr in corrections:
            if corr["correct_model"]:
                words = re.findall(r'\b\w+\b', corr["request"].lower())
                for word in words:
                    keyword_destinations[word][corr["correct_model"]] += 1

        # Suggest moves for keywords with clear patterns
        for keyword, destinations in keyword_destinations.items():
            if len(destinations) >= 1:
                best_dest = max(destinations.items(), key=lambda x: x[1])
                if best_dest[1] >= 2:  # At least 2 occurrences
                    # Find examples
                    examples = [
                        c["request"] for c in corrections
                        if keyword in c["request"].lower() and c["correct_model"] == best_dest[0]
                    ]

                    suggestion = KeywordSuggestion(
                        action="add",
                        keyword=keyword,
                        from_category=None,
                        to_category=best_dest[0],
                        reason=f"Found in {best_dest[1]} requests that should go to {best_dest[0]}",
                        supporting_examples=examples[:3]
                    )
                    suggestions.append(suggestion)

        return suggestions

    def identify_edge_cases(self, logs: List[RoutingLog]) -> List[Dict[str, Any]]:
        """
        Find edge cases that confuse the router.
        These are requests with low confidence or mixed signals.
        """
        edge_cases = []

        for log in logs:
            is_edge_case = False
            reasons = []

            # Low confidence
            if log.routing.confidence < 0.6:
                is_edge_case = True
                reasons.append(f"Low confidence: {log.routing.confidence:.2f}")

            # Mixed task indicators
            indicators = log.features.task_indicators
            if indicators:
                high_indicators = [k for k, v in indicators.items() if v > 0.5]
                if len(high_indicators) > 1:
                    is_edge_case = True
                    reasons.append(f"Multiple task indicators: {high_indicators}")

            # Conflicting keywords
            if log.request.has_image and log.routing.task_type != "vision":
                is_edge_case = True
                reasons.append("Has image but not routed to vision")

            # User corrected
            if log.feedback.correct_model:
                is_edge_case = True
                reasons.append(f"User corrected to {log.feedback.correct_model}")

            if is_edge_case:
                edge_cases.append({
                    "request": log.request.text,
                    "routed_to": log.routing.model_chosen,
                    "task_type": log.routing.task_type,
                    "confidence": log.routing.confidence,
                    "reasons": reasons,
                    "correct_model": log.feedback.correct_model
                })

        return edge_cases

    def analyze_model_performance(self, logs: List[RoutingLog]) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics per model.
        """
        model_stats = defaultdict(lambda: {"total": 0, "success": 0, "corrected": 0})

        for log in logs:
            model = log.routing.model_chosen
            model_stats[model]["total"] += 1

            if log.execution.success:
                model_stats[model]["success"] += 1

            if log.feedback.user_rating == -1 or log.feedback.correct_model:
                model_stats[model]["corrected"] += 1

        # Calculate rates
        performance = {}
        for model, stats in model_stats.items():
            total = stats["total"]
            performance[model] = {
                "usage_count": total,
                "success_rate": stats["success"] / total if total > 0 else 0,
                "correction_rate": stats["corrected"] / total if total > 0 else 0,
                "effective_accuracy": (total - stats["corrected"]) / total if total > 0 else 1
            }

        return performance


# ============================================================================
# Quick pattern analysis function
# ============================================================================

def quick_analyze(days_back: int = 7) -> PatternReport:
    """Quick one-liner to run pattern analysis"""
    detector = PatternDetector()
    return detector.analyze(days_back=days_back)


def print_pattern_report(report: PatternReport):
    """Print a human-readable pattern report"""
    print("=" * 60)
    print("PATTERN ANALYSIS REPORT")
    print("=" * 60)
    print(f"Analyzed: {report.analyzed_logs} logs")
    print(f"Date range: {report.date_range[0]} to {report.date_range[1]}")

    if report.patterns:
        print("\n" + "-" * 40)
        print("MISROUTING PATTERNS")
        print("-" * 40)
        for i, p in enumerate(report.patterns, 1):
            print(f"\n{i}. [{p.pattern_type.upper()}] {p.description}")
            print(f"   Occurrences: {p.occurrences}")
            print(f"   Impact: {p.impact_score:.0%}")
            print(f"   Fix: {p.suggested_fix}")
            print(f"   Examples:")
            for ex in p.examples[:2]:
                print(f"     - {ex[:50]}...")

    if report.keyword_suggestions:
        print("\n" + "-" * 40)
        print("KEYWORD SUGGESTIONS")
        print("-" * 40)
        for s in report.keyword_suggestions[:5]:
            print(f"\n  {s.action.upper()} '{s.keyword}' -> {s.to_category}")
            print(f"  Reason: {s.reason}")

    if report.model_performance:
        print("\n" + "-" * 40)
        print("MODEL PERFORMANCE")
        print("-" * 40)
        for model, stats in sorted(
            report.model_performance.items(),
            key=lambda x: x[1]["usage_count"],
            reverse=True
        ):
            print(f"\n  {model}:")
            print(f"    Usage: {stats['usage_count']}")
            print(f"    Success rate: {stats['success_rate']:.0%}")
            print(f"    Correction rate: {stats['correction_rate']:.0%}")
            print(f"    Effective accuracy: {stats['effective_accuracy']:.0%}")

    if report.edge_cases:
        print("\n" + "-" * 40)
        print(f"EDGE CASES ({len(report.edge_cases)} found)")
        print("-" * 40)
        for case in report.edge_cases[:5]:
            print(f"\n  Request: {case['request'][:50]}...")
            print(f"  Routed to: {case['routed_to']}")
            print(f"  Reasons: {', '.join(case['reasons'])}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PATTERN DETECTOR TEST")
    print("=" * 60)

    # Create some test logs with feedback
    from core_logger import (
        RoutingLog, RequestData, RoutingData,
        ExecutionData, FeedbackData, FeatureData
    )

    test_logs = [
        # Correct routing
        RoutingLog(
            timestamp="2025-12-11T10:00:00",
            session_id="test",
            request=RequestData(text="goto amazon.com"),
            routing=RoutingData(
                task_type="browser_automation",
                confidence=0.92,
                model_chosen="command-r-35b"
            ),
            feedback=FeedbackData(user_rating=1)
        ),
        # Misrouting - should be command-r
        RoutingLog(
            timestamp="2025-12-11T10:01:00",
            session_id="test",
            request=RequestData(text="navigate to google.com"),
            routing=RoutingData(
                task_type="general_chat",
                confidence=0.65,
                model_chosen="ministral-14b-reasoning"
            ),
            feedback=FeedbackData(user_rating=-1, correct_model="command-r-35b")
        ),
        # Another misrouting with same pattern
        RoutingLog(
            timestamp="2025-12-11T10:02:00",
            session_id="test",
            request=RequestData(text="navigate to twitter and post"),
            routing=RoutingData(
                task_type="general_chat",
                confidence=0.60,
                model_chosen="ministral-14b-reasoning"
            ),
            feedback=FeedbackData(user_rating=-1, correct_model="command-r-35b")
        ),
        # Edge case - low confidence
        RoutingLog(
            timestamp="2025-12-11T10:03:00",
            session_id="test",
            request=RequestData(text="explain how to browse the web"),
            routing=RoutingData(
                task_type="reasoning",
                confidence=0.55,
                model_chosen="ministral-14b-reasoning"
            ),
            features=FeatureData(
                task_indicators={"tool_use": 0.6, "reasoning": 0.7}
            )
        ),
    ]

    detector = PatternDetector()
    report = detector.analyze(logs=test_logs)

    print_pattern_report(report)
