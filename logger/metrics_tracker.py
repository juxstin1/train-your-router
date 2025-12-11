"""
Metrics Tracker - Calculate accuracy, timing, and performance metrics.
Provides real-time and historical analytics.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import statistics

from core_logger import RoutingLogger, RoutingLog, get_logger


@dataclass
class AccuracyMetrics:
    """Accuracy-related metrics"""
    total_requests: int = 0
    correct_routings: int = 0  # Explicit thumbs up
    incorrect_routings: int = 0  # Explicit thumbs down / corrections
    implicit_ok: int = 0  # No feedback = assumed OK
    accuracy_rate: float = 0.0  # (correct + implicit) / total
    explicit_accuracy: float = 0.0  # correct / (correct + incorrect)
    correction_rate: float = 0.0  # incorrect / total


@dataclass
class TimingMetrics:
    """Timing-related metrics"""
    avg_decision_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    p50_decision_time_ms: float = 0.0
    p95_decision_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0


@dataclass
class ModelMetrics:
    """Per-model metrics"""
    model_name: str
    usage_count: int = 0
    usage_percentage: float = 0.0
    success_rate: float = 0.0
    correction_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_response_time_ms: float = 0.0
    common_task_types: List[str] = field(default_factory=list)


@dataclass
class TaskTypeMetrics:
    """Per-task-type metrics"""
    task_type: str
    count: int = 0
    percentage: float = 0.0
    accuracy: float = 0.0
    most_used_model: str = ""


@dataclass
class ConfusionMatrix:
    """Task type confusion matrix"""
    # matrix[actual][predicted] = count
    matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    task_types: List[str] = field(default_factory=list)


@dataclass
class DailyMetrics:
    """Complete metrics for a day"""
    date: str
    accuracy: AccuracyMetrics
    timing: TimingMetrics
    models: Dict[str, ModelMetrics]
    task_types: Dict[str, TaskTypeMetrics]
    confusion_matrix: ConfusionMatrix
    total_tokens: int = 0
    total_tool_calls: int = 0


class MetricsTracker:
    """
    Calculates and tracks routing metrics over time.
    """

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.logger = logger or get_logger()
        self.analytics_dir = Path(self.logger.base_dir) / "routing" / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

    def calculate_daily_metrics(
        self,
        date_str: Optional[str] = None
    ) -> DailyMetrics:
        """
        Calculate all metrics for a specific day.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        logs = self.logger.read_daily_logs(date_str)

        return DailyMetrics(
            date=date_str,
            accuracy=self._calc_accuracy(logs),
            timing=self._calc_timing(logs),
            models=self._calc_model_metrics(logs),
            task_types=self._calc_task_metrics(logs),
            confusion_matrix=self._calc_confusion_matrix(logs),
            total_tokens=sum(log.execution.tokens_generated for log in logs),
            total_tool_calls=sum(len(log.execution.tool_calls) for log in logs)
        )

    def _calc_accuracy(self, logs: List[RoutingLog]) -> AccuracyMetrics:
        """Calculate accuracy metrics"""
        if not logs:
            return AccuracyMetrics()

        correct = 0
        incorrect = 0
        implicit = 0

        for log in logs:
            if log.feedback.user_rating == 1:
                correct += 1
            elif log.feedback.user_rating == -1 or log.feedback.correct_model:
                incorrect += 1
            else:
                implicit += 1

        total = len(logs)
        explicit_total = correct + incorrect

        return AccuracyMetrics(
            total_requests=total,
            correct_routings=correct,
            incorrect_routings=incorrect,
            implicit_ok=implicit,
            accuracy_rate=(correct + implicit) / total if total > 0 else 0,
            explicit_accuracy=correct / explicit_total if explicit_total > 0 else 1.0,
            correction_rate=incorrect / total if total > 0 else 0
        )

    def _calc_timing(self, logs: List[RoutingLog]) -> TimingMetrics:
        """Calculate timing metrics"""
        if not logs:
            return TimingMetrics()

        decision_times = [log.routing.decision_time_ms for log in logs if log.routing.decision_time_ms > 0]
        response_times = [log.execution.response_time_ms for log in logs if log.execution.response_time_ms > 0]

        def percentile(data: List[float], p: int) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return TimingMetrics(
            avg_decision_time_ms=statistics.mean(decision_times) if decision_times else 0,
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
            p50_decision_time_ms=percentile(decision_times, 50),
            p95_decision_time_ms=percentile(decision_times, 95),
            p50_response_time_ms=percentile(response_times, 50),
            p95_response_time_ms=percentile(response_times, 95)
        )

    def _calc_model_metrics(self, logs: List[RoutingLog]) -> Dict[str, ModelMetrics]:
        """Calculate per-model metrics"""
        if not logs:
            return {}

        model_logs = defaultdict(list)
        for log in logs:
            model_logs[log.routing.model_chosen].append(log)

        total = len(logs)
        metrics = {}

        for model, model_log_list in model_logs.items():
            count = len(model_log_list)
            successes = sum(1 for log in model_log_list if log.execution.success)
            corrections = sum(1 for log in model_log_list
                            if log.feedback.user_rating == -1 or log.feedback.correct_model)
            confidences = [log.routing.confidence for log in model_log_list]
            response_times = [log.execution.response_time_ms for log in model_log_list
                            if log.execution.response_time_ms > 0]
            task_types = [log.routing.task_type for log in model_log_list]

            # Get most common task types
            task_counts = defaultdict(int)
            for tt in task_types:
                task_counts[tt] += 1
            common_tasks = sorted(task_counts.keys(), key=lambda x: task_counts[x], reverse=True)[:3]

            metrics[model] = ModelMetrics(
                model_name=model,
                usage_count=count,
                usage_percentage=count / total if total > 0 else 0,
                success_rate=successes / count if count > 0 else 0,
                correction_rate=corrections / count if count > 0 else 0,
                avg_confidence=statistics.mean(confidences) if confidences else 0,
                avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
                common_task_types=common_tasks
            )

        return metrics

    def _calc_task_metrics(self, logs: List[RoutingLog]) -> Dict[str, TaskTypeMetrics]:
        """Calculate per-task-type metrics"""
        if not logs:
            return {}

        task_logs = defaultdict(list)
        for log in logs:
            task_logs[log.routing.task_type].append(log)

        total = len(logs)
        metrics = {}

        for task_type, task_log_list in task_logs.items():
            count = len(task_log_list)
            correct = sum(1 for log in task_log_list
                        if log.feedback.user_rating != -1 and not log.feedback.correct_model)

            # Most used model for this task
            model_counts = defaultdict(int)
            for log in task_log_list:
                model_counts[log.routing.model_chosen] += 1
            most_used = max(model_counts.keys(), key=lambda x: model_counts[x]) if model_counts else ""

            metrics[task_type] = TaskTypeMetrics(
                task_type=task_type,
                count=count,
                percentage=count / total if total > 0 else 0,
                accuracy=correct / count if count > 0 else 1.0,
                most_used_model=most_used
            )

        return metrics

    def _calc_confusion_matrix(self, logs: List[RoutingLog]) -> ConfusionMatrix:
        """
        Build confusion matrix for task types.
        Only includes logs where user provided correction.
        """
        matrix = defaultdict(lambda: defaultdict(int))
        task_types = set()

        for log in logs:
            if log.feedback.correct_model:
                # We have a correction - this tells us what was wrong
                predicted = log.routing.task_type
                # We don't know the "actual" task type, but we know the model was wrong
                # For now, record as "corrected" vs "original"
                matrix[predicted]["corrected"] += 1
                task_types.add(predicted)
            else:
                # No correction - assume correct
                predicted = log.routing.task_type
                matrix[predicted]["correct"] += 1
                task_types.add(predicted)

        return ConfusionMatrix(
            matrix=dict(matrix),
            task_types=list(task_types)
        )

    def get_accuracy_trend(self, days: int = 7) -> List[Tuple[str, float]]:
        """Get accuracy trend over past N days"""
        trend = []
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            day_str = day.isoformat()
            metrics = self.calculate_daily_metrics(day_str)
            trend.append((day_str, metrics.accuracy.accuracy_rate))

        return list(reversed(trend))  # Oldest first

    def get_model_usage_trend(self, days: int = 7) -> Dict[str, List[Tuple[str, int]]]:
        """Get model usage trends over past N days"""
        trends = defaultdict(list)
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            day_str = day.isoformat()
            metrics = self.calculate_daily_metrics(day_str)

            for model, model_metrics in metrics.models.items():
                trends[model].append((day_str, model_metrics.usage_count))

        # Reverse to oldest first
        return {model: list(reversed(data)) for model, data in trends.items()}

    def save_daily_analytics(self, date_str: Optional[str] = None):
        """Save daily analytics to file"""
        if date_str is None:
            date_str = date.today().isoformat()

        metrics = self.calculate_daily_metrics(date_str)

        # Convert to serializable dict
        data = {
            "date": metrics.date,
            "accuracy": asdict(metrics.accuracy),
            "timing": asdict(metrics.timing),
            "models": {k: asdict(v) for k, v in metrics.models.items()},
            "task_types": {k: asdict(v) for k, v in metrics.task_types.items()},
            "confusion_matrix": asdict(metrics.confusion_matrix),
            "total_tokens": metrics.total_tokens,
            "total_tool_calls": metrics.total_tool_calls
        }

        filepath = self.analytics_dir / f"daily_{date_str}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics over past N days"""
        all_logs = []
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            logs = self.logger.read_daily_logs(day.isoformat())
            all_logs.extend(logs)

        accuracy = self._calc_accuracy(all_logs)
        timing = self._calc_timing(all_logs)
        models = self._calc_model_metrics(all_logs)

        return {
            "period_days": days,
            "total_requests": accuracy.total_requests,
            "accuracy_rate": accuracy.accuracy_rate,
            "correction_rate": accuracy.correction_rate,
            "avg_decision_time_ms": timing.avg_decision_time_ms,
            "avg_response_time_ms": timing.avg_response_time_ms,
            "top_models": sorted(
                [(m, d.usage_count) for m, d in models.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3],
            "model_with_highest_correction_rate": max(
                models.items(),
                key=lambda x: x[1].correction_rate,
                default=(None, ModelMetrics(model_name=""))
            )[0] if models else None
        }


# ============================================================================
# Quick metrics functions
# ============================================================================

def quick_metrics(days: int = 1) -> DailyMetrics:
    """Quick one-liner to get today's metrics"""
    tracker = MetricsTracker()
    return tracker.calculate_daily_metrics()


def print_metrics(metrics: DailyMetrics):
    """Print metrics in a readable format"""
    print("=" * 60)
    print(f"METRICS REPORT - {metrics.date}")
    print("=" * 60)

    print("\nACCURACY:")
    print(f"  Total requests: {metrics.accuracy.total_requests}")
    print(f"  Accuracy rate: {metrics.accuracy.accuracy_rate:.1%}")
    print(f"  Correction rate: {metrics.accuracy.correction_rate:.1%}")
    print(f"  Explicit feedback: {metrics.accuracy.correct_routings} correct, "
          f"{metrics.accuracy.incorrect_routings} incorrect")

    print("\nTIMING:")
    print(f"  Avg decision time: {metrics.timing.avg_decision_time_ms:.1f}ms")
    print(f"  Avg response time: {metrics.timing.avg_response_time_ms:.1f}ms")
    print(f"  P95 decision time: {metrics.timing.p95_decision_time_ms:.1f}ms")

    print("\nMODEL USAGE:")
    for model, m in sorted(metrics.models.items(), key=lambda x: x[1].usage_count, reverse=True):
        print(f"  {model}:")
        print(f"    Usage: {m.usage_count} ({m.usage_percentage:.0%})")
        print(f"    Success rate: {m.success_rate:.0%}")
        print(f"    Correction rate: {m.correction_rate:.0%}")

    print("\nTASK TYPES:")
    for task, t in sorted(metrics.task_types.items(), key=lambda x: x[1].count, reverse=True):
        print(f"  {task}: {t.count} ({t.percentage:.0%}) -> {t.most_used_model}")

    print(f"\nTOTALS:")
    print(f"  Tokens generated: {metrics.total_tokens}")
    print(f"  Tool calls made: {metrics.total_tool_calls}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    from core_logger import quick_log

    print("=" * 60)
    print("METRICS TRACKER TEST")
    print("=" * 60)

    # Generate some test data
    print("\nGenerating test data...")
    test_requests = [
        ("goto amazon.com", "command-r-35b", "browser_automation", True),
        ("write a function", "qwen3-coder-30b", "code_generation", True),
        ("explain this", "ministral-14b-reasoning", "reasoning", True),
        ("hello", "gemma-3-1b", "simple_chat", True),
        ("navigate to google", "ministral-14b-reasoning", "general_chat", False),  # Will be corrected
    ]

    for request, model, task_type, success in test_requests:
        quick_log(
            request=request,
            model=model,
            task_type=task_type,
            decision_time_ms=10.5,
            response_time_ms=1500,
            tokens=50,
            success=success
        )

    # Calculate and print metrics
    tracker = MetricsTracker()
    metrics = tracker.calculate_daily_metrics()
    print_metrics(metrics)

    # Get summary
    print("\n" + "-" * 40)
    print("7-DAY SUMMARY:")
    summary = tracker.get_summary(days=7)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save analytics
    filepath = tracker.save_daily_analytics()
    print(f"\nAnalytics saved to: {filepath}")
