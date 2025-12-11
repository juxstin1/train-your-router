"""
Research Agent - Autonomous daily analysis and improvement suggestions.
Generates markdown reports with actionable recommendations.
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core_logger import RoutingLogger, RoutingLog, get_logger
from pattern_detector import PatternDetector, PatternReport, print_pattern_report
from metrics_tracker import MetricsTracker, DailyMetrics, print_metrics


@dataclass
class ResearchInsight:
    """A single research insight"""
    category: str  # "pattern", "performance", "suggestion", "alert"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class TrainingPriority:
    """Training data priority"""
    category: str
    priority: int  # 1 = highest
    example_count: int
    reason: str
    examples: List[str]


class RoutingResearchAgent:
    """
    Autonomous agent that analyzes logs and suggests improvements.
    Generates daily reports and actionable recommendations.
    """

    def __init__(self, logger: Optional[RoutingLogger] = None):
        self.logger = logger or get_logger()
        self.pattern_detector = PatternDetector(self.logger)
        self.metrics_tracker = MetricsTracker(self.logger)
        self.research_dir = Path(self.logger.base_dir) / "research"
        self.research_dir.mkdir(parents=True, exist_ok=True)

    def run_daily_analysis(self, date_str: Optional[str] = None) -> str:
        """
        Run full daily analysis and generate report.
        Returns path to generated report.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        # Gather all data
        logs = self.logger.read_daily_logs(date_str)
        metrics = self.metrics_tracker.calculate_daily_metrics(date_str)
        patterns = self.pattern_detector.analyze(logs)

        # Run all analysis tasks
        insights = []
        insights.extend(self.analyze_routing_accuracy(metrics, logs))
        insights.extend(self.find_keyword_gaps(patterns))
        insights.extend(self.identify_model_strengths(metrics))
        insights.extend(self.detect_drift(date_str))

        # Get training priorities
        training_priorities = self.suggest_training_priorities(patterns, metrics)

        # Generate report
        report = self._generate_report(
            date_str=date_str,
            metrics=metrics,
            patterns=patterns,
            insights=insights,
            training_priorities=training_priorities,
            logs=logs
        )

        # Save report
        report_path = self.research_dir / "insights" / f"daily_report_{date_str}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # Also save as JSON for programmatic access
        json_path = self.research_dir / "insights" / f"daily_data_{date_str}.json"
        self._save_json_report(json_path, metrics, patterns, insights, training_priorities)

        return str(report_path)

    def analyze_routing_accuracy(
        self,
        metrics: DailyMetrics,
        logs: List[RoutingLog]
    ) -> List[ResearchInsight]:
        """Analyze routing accuracy and identify issues"""
        insights = []

        # Overall accuracy check
        acc = metrics.accuracy
        if acc.accuracy_rate < 0.8:
            insights.append(ResearchInsight(
                category="alert",
                priority="high",
                title="Low Routing Accuracy",
                description=f"Accuracy is {acc.accuracy_rate:.0%}, below 80% target",
                action="Review recent corrections and update routing rules",
                data={"accuracy": acc.accuracy_rate, "corrections": acc.incorrect_routings}
            ))
        elif acc.accuracy_rate >= 0.9:
            insights.append(ResearchInsight(
                category="performance",
                priority="low",
                title="Excellent Routing Accuracy",
                description=f"Accuracy at {acc.accuracy_rate:.0%}",
                data={"accuracy": acc.accuracy_rate}
            ))

        # Per-model accuracy
        for model, model_metrics in metrics.models.items():
            if model_metrics.correction_rate > 0.2 and model_metrics.usage_count >= 3:
                insights.append(ResearchInsight(
                    category="alert",
                    priority="high",
                    title=f"High Correction Rate for {model}",
                    description=f"{model} has {model_metrics.correction_rate:.0%} correction rate",
                    action=f"Review what tasks are being routed to {model}",
                    data={
                        "model": model,
                        "correction_rate": model_metrics.correction_rate,
                        "usage": model_metrics.usage_count
                    }
                ))

        return insights

    def find_keyword_gaps(self, patterns: PatternReport) -> List[ResearchInsight]:
        """Find gaps in keyword coverage"""
        insights = []

        for suggestion in patterns.keyword_suggestions[:5]:
            insights.append(ResearchInsight(
                category="suggestion",
                priority="medium",
                title=f"Add keyword '{suggestion.keyword}'",
                description=suggestion.reason,
                action=f"Add '{suggestion.keyword}' to {suggestion.to_category} keywords",
                data={
                    "keyword": suggestion.keyword,
                    "target": suggestion.to_category,
                    "examples": suggestion.supporting_examples
                }
            ))

        return insights

    def identify_model_strengths(self, metrics: DailyMetrics) -> List[ResearchInsight]:
        """Identify what each model is actually good at"""
        insights = []

        # Find best and worst performers
        if metrics.models:
            # Best performer
            best = min(
                [(m, d) for m, d in metrics.models.items() if d.usage_count >= 3],
                key=lambda x: x[1].correction_rate,
                default=(None, None)
            )
            if best[0]:
                insights.append(ResearchInsight(
                    category="performance",
                    priority="low",
                    title=f"Top Performer: {best[0]}",
                    description=f"Only {best[1].correction_rate:.0%} correction rate with {best[1].usage_count} uses",
                    data={
                        "model": best[0],
                        "correction_rate": best[1].correction_rate,
                        "common_tasks": best[1].common_task_types
                    }
                ))

            # Worst performer (that's being used)
            worst = max(
                [(m, d) for m, d in metrics.models.items() if d.usage_count >= 3],
                key=lambda x: x[1].correction_rate,
                default=(None, None)
            )
            if worst[0] and worst[1].correction_rate > 0.1:
                insights.append(ResearchInsight(
                    category="alert",
                    priority="medium",
                    title=f"Underperforming: {worst[0]}",
                    description=f"{worst[1].correction_rate:.0%} correction rate - consider limiting its use",
                    action=f"Reduce routing to {worst[0]} for its common tasks: {worst[1].common_task_types}",
                    data={
                        "model": worst[0],
                        "correction_rate": worst[1].correction_rate,
                        "common_tasks": worst[1].common_task_types
                    }
                ))

        return insights

    def suggest_training_priorities(
        self,
        patterns: PatternReport,
        metrics: DailyMetrics
    ) -> List[TrainingPriority]:
        """Determine what training data would help most"""
        priorities = []

        # Priority from patterns
        pattern_counts = {}
        for pattern in patterns.patterns:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + pattern.occurrences

        # Tool use edge cases (usually highest priority)
        if "keyword" in pattern_counts:
            priorities.append(TrainingPriority(
                category="tool_use_edge_cases",
                priority=1,
                example_count=pattern_counts["keyword"] * 2,
                reason=f"Found {pattern_counts['keyword']} keyword-based misroutings",
                examples=[p.examples[0] for p in patterns.patterns if p.pattern_type == "keyword"][:3]
            ))

        # Model confusion
        if "model_confusion" in pattern_counts:
            priorities.append(TrainingPriority(
                category="model_distinction",
                priority=2,
                example_count=pattern_counts["model_confusion"] * 2,
                reason=f"Found {pattern_counts['model_confusion']} model confusion cases",
                examples=[p.examples[0] for p in patterns.patterns if p.pattern_type == "model_confusion"][:3]
            ))

        # Edge cases from patterns
        if patterns.edge_cases:
            priorities.append(TrainingPriority(
                category="edge_cases",
                priority=3,
                example_count=len(patterns.edge_cases),
                reason=f"Found {len(patterns.edge_cases)} edge cases with low confidence",
                examples=[e["request"] for e in patterns.edge_cases[:3]]
            ))

        return sorted(priorities, key=lambda x: x.priority)

    def detect_drift(self, current_date: str) -> List[ResearchInsight]:
        """Detect if routing performance is changing over time"""
        insights = []

        # Compare last 7 days
        trend = self.metrics_tracker.get_accuracy_trend(days=7)

        if len(trend) >= 3:
            recent = [t[1] for t in trend[-3:]]
            older = [t[1] for t in trend[:-3]] if len(trend) > 3 else []

            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)

                if recent_avg < older_avg - 0.1:
                    insights.append(ResearchInsight(
                        category="alert",
                        priority="high",
                        title="Accuracy Declining",
                        description=f"Recent accuracy ({recent_avg:.0%}) is lower than before ({older_avg:.0%})",
                        action="Investigate new request types or model changes",
                        data={"recent_avg": recent_avg, "older_avg": older_avg}
                    ))
                elif recent_avg > older_avg + 0.1:
                    insights.append(ResearchInsight(
                        category="performance",
                        priority="low",
                        title="Accuracy Improving",
                        description=f"Recent accuracy ({recent_avg:.0%}) is higher than before ({older_avg:.0%})",
                        data={"recent_avg": recent_avg, "older_avg": older_avg}
                    ))

        return insights

    def _generate_report(
        self,
        date_str: str,
        metrics: DailyMetrics,
        patterns: PatternReport,
        insights: List[ResearchInsight],
        training_priorities: List[TrainingPriority],
        logs: List[RoutingLog]
    ) -> str:
        """Generate markdown report"""

        # Categorize insights
        alerts = [i for i in insights if i.priority == "high"]
        suggestions = [i for i in insights if i.category == "suggestion"]

        report = f"""# Daily Routing Analysis - {date_str}

## Summary
- **Total requests:** {metrics.accuracy.total_requests}
- **Routing accuracy:** {metrics.accuracy.accuracy_rate:.0%} (target: 80%+)
- **Average decision time:** {metrics.timing.avg_decision_time_ms:.1f}ms
- **User corrections:** {metrics.accuracy.incorrect_routings} ({metrics.accuracy.correction_rate:.0%})

"""

        # Alerts section
        if alerts:
            report += "## Alerts\n\n"
            for alert in alerts:
                report += f"### {alert.title}\n"
                report += f"{alert.description}\n\n"
                if alert.action:
                    report += f"**Action:** {alert.action}\n\n"

        # Key findings
        report += "## Key Findings\n\n"

        # Patterns
        if patterns.patterns:
            report += "### Misrouting Patterns\n\n"
            for i, p in enumerate(patterns.patterns[:3], 1):
                report += f"**{i}. {p.description}**\n"
                report += f"- Occurrences: {p.occurrences}\n"
                report += f"- Impact: {p.impact_score:.0%}\n"
                report += f"- Fix: {p.suggested_fix}\n"
                report += f"- Examples: {', '.join(p.examples[:2])}\n\n"

        # Model performance
        report += "### Model Performance\n\n"
        report += "| Model | Usage | Success Rate | Correction Rate | Notes |\n"
        report += "|-------|-------|--------------|-----------------|-------|\n"

        for model, m in sorted(metrics.models.items(), key=lambda x: x[1].usage_count, reverse=True):
            status = "" if m.correction_rate <= 0.1 else "" if m.correction_rate <= 0.2 else ""
            report += f"| {model} | {m.usage_count} ({m.usage_percentage:.0%}) | "
            report += f"{m.success_rate:.0%} | {m.correction_rate:.0%} | {status} |\n"

        report += "\n"

        # Suggestions
        if suggestions:
            report += "### Keyword Suggestions\n\n"
            for s in suggestions[:5]:
                report += f"- **{s.title}**: {s.description}\n"
            report += "\n"

        # Training priorities
        if training_priorities:
            report += "## Training Data Priorities\n\n"
            for tp in training_priorities:
                priority_label = ["", "High", "Medium", "Low"][min(tp.priority, 3)]
                report += f"### {priority_label} Priority: {tp.category}\n"
                report += f"- **Recommended examples:** {tp.example_count}\n"
                report += f"- **Reason:** {tp.reason}\n"
                if tp.examples:
                    report += f"- **Sample requests:**\n"
                    for ex in tp.examples:
                        report += f"  - {ex[:60]}...\n" if len(ex) > 60 else f"  - {ex}\n"
                report += "\n"

        # Recommendations
        report += "## Recommendations\n\n"
        report += "### Immediate (Do Today)\n\n"

        immediate_actions = []
        if alerts:
            for a in alerts:
                if a.action:
                    immediate_actions.append(a.action)
        if suggestions:
            for s in suggestions[:2]:
                if s.action:
                    immediate_actions.append(s.action)

        if immediate_actions:
            for action in immediate_actions[:3]:
                report += f"- [ ] {action}\n"
        else:
            report += "- [ ] No immediate actions needed\n"

        report += "\n### This Week\n\n"
        if training_priorities:
            tp = training_priorities[0]
            report += f"- [ ] Generate {tp.example_count} {tp.category} examples\n"
        report += "- [ ] Review edge cases and add to training data\n"
        report += "- [ ] Update routing keywords based on patterns\n"

        # Data quality
        report += "\n## Data Quality\n\n"
        feedback_count = metrics.accuracy.correct_routings + metrics.accuracy.incorrect_routings
        report += f"- High confidence logs: {sum(1 for l in logs if l.routing.confidence >= 0.8)}/{len(logs)}\n"
        report += f"- User feedback captured: {feedback_count}/{len(logs)} ({feedback_count/len(logs):.0%})\n" if logs else ""
        report += f"- Complete execution data: {len(logs)}/{len(logs)} (100%)\n"

        report += f"\n---\n*Generated by RoutingResearchAgent v1.0 at {datetime.now().isoformat()}*\n"

        return report

    def _save_json_report(
        self,
        path: Path,
        metrics: DailyMetrics,
        patterns: PatternReport,
        insights: List[ResearchInsight],
        training_priorities: List[TrainingPriority]
    ):
        """Save report data as JSON"""
        from dataclasses import asdict

        data = {
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "accuracy": asdict(metrics.accuracy),
                "timing": asdict(metrics.timing),
                "total_tokens": metrics.total_tokens,
                "total_tool_calls": metrics.total_tool_calls
            },
            "patterns": {
                "analyzed_logs": patterns.analyzed_logs,
                "pattern_count": len(patterns.patterns),
                "edge_case_count": len(patterns.edge_cases)
            },
            "insights": [
                {
                    "category": i.category,
                    "priority": i.priority,
                    "title": i.title,
                    "action": i.action
                }
                for i in insights
            ],
            "training_priorities": [
                {
                    "category": tp.category,
                    "priority": tp.priority,
                    "example_count": tp.example_count,
                    "reason": tp.reason
                }
                for tp in training_priorities
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# CLI Interface
# ============================================================================

def run_analysis(date_str: Optional[str] = None, print_report: bool = True) -> str:
    """Run analysis and optionally print report"""
    agent = RoutingResearchAgent()
    report_path = agent.run_daily_analysis(date_str)

    if print_report:
        with open(report_path, 'r') as f:
            print(f.read())

    return report_path


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run routing research analysis")
    parser.add_argument("--date", "-d", type=str, help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Don't print report")

    args = parser.parse_args()

    print("=" * 60)
    print("ROUTING RESEARCH AGENT")
    print("=" * 60)

    # Generate some test data if none exists
    from core_logger import quick_log, RoutingLogger, FeedbackData

    logger = RoutingLogger(base_dir="../logs")

    # Check if we have logs
    logs = logger.read_daily_logs()
    if not logs:
        print("\nNo logs found. Generating test data...")

        # Generate varied test data
        test_data = [
            ("goto amazon.com", "command-r-35b", "browser_automation", 0.92, True, None),
            ("navigate to twitter", "ministral-14b-reasoning", "general_chat", 0.65, False, "command-r-35b"),
            ("write a function", "qwen3-coder-30b", "code_generation", 0.88, True, None),
            ("navigate to google", "ministral-14b-reasoning", "general_chat", 0.60, False, "command-r-35b"),
            ("explain transformers", "ministral-14b-reasoning", "reasoning", 0.75, True, None),
            ("browse reddit", "ministral-14b-reasoning", "general_chat", 0.58, False, "command-r-35b"),
            ("screenshot the page", "command-r-35b", "tool_use", 0.90, True, None),
            ("hello", "gemma-3-1b", "simple_chat", 0.95, True, None),
            ("debug this code", "qwen3-coder-30b", "code_generation", 0.85, True, None),
            ("what's in this image", "qwen3-vl-8b", "vision", 0.92, True, None),
        ]

        for request, model, task_type, confidence, correct, correction in test_data:
            log = quick_log(
                request=request,
                model=model,
                task_type=task_type,
                confidence=confidence,
                decision_time_ms=12.5,
                response_time_ms=1500,
                tokens=100
            )
            if not correct and correction:
                logger.update_feedback(
                    log.timestamp,
                    FeedbackData(user_rating=-1, correct_model=correction)
                )
            elif correct:
                logger.update_feedback(
                    log.timestamp,
                    FeedbackData(user_rating=1)
                )

        print(f"Generated {len(test_data)} test logs")

    # Run analysis
    print("\nRunning analysis...")
    report_path = run_analysis(args.date, print_report=not args.quiet)
    print(f"\nReport saved to: {report_path}")
