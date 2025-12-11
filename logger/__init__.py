"""
Routing Logger & Research Agent

A comprehensive logging system for model routing decisions.

Quick Start:
    from logger import quick_log, inline_feedback, run_analysis

    # Log a routing decision
    log = quick_log(
        request="goto amazon.com",
        model="command-r-35b",
        task_type="browser_automation",
        tools=["browser_navigate"]
    )

    # Collect feedback (optional)
    inline_feedback(log)

    # Run daily analysis
    run_analysis()
"""

from .core_logger import (
    RoutingLogger,
    RoutingLog,
    RequestData,
    RequestContext,
    RoutingData,
    ExecutionData,
    FeedbackData,
    FeatureData,
    LoggingContext,
    get_logger,
    quick_log,
)

from .feedback_prompt import (
    FeedbackCollector,
    inline_feedback,
)

from .session_tracker import (
    SessionTracker,
    Session,
    ConversationTurn,
    ContextAwareRouter,
)

from .pattern_detector import (
    PatternDetector,
    PatternReport,
    MisroutingPattern,
    KeywordSuggestion,
    quick_analyze,
    print_pattern_report,
)

from .metrics_tracker import (
    MetricsTracker,
    DailyMetrics,
    AccuracyMetrics,
    TimingMetrics,
    ModelMetrics,
    quick_metrics,
    print_metrics,
)

from .research_agent import (
    RoutingResearchAgent,
    ResearchInsight,
    TrainingPriority,
    run_analysis,
)

__all__ = [
    # Core logging
    "RoutingLogger",
    "RoutingLog",
    "RequestData",
    "RequestContext",
    "RoutingData",
    "ExecutionData",
    "FeedbackData",
    "FeatureData",
    "LoggingContext",
    "get_logger",
    "quick_log",
    # Feedback
    "FeedbackCollector",
    "inline_feedback",
    # Session tracking
    "SessionTracker",
    "Session",
    "ConversationTurn",
    "ContextAwareRouter",
    # Pattern detection
    "PatternDetector",
    "PatternReport",
    "MisroutingPattern",
    "KeywordSuggestion",
    "quick_analyze",
    "print_pattern_report",
    # Metrics
    "MetricsTracker",
    "DailyMetrics",
    "AccuracyMetrics",
    "TimingMetrics",
    "ModelMetrics",
    "quick_metrics",
    "print_metrics",
    # Research agent
    "RoutingResearchAgent",
    "ResearchInsight",
    "TrainingPriority",
    "run_analysis",
]
