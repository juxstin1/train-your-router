#!/usr/bin/env python3
"""
Start an interactive routing session with full logging and health checks.

Features:
- Model health check on startup
- Automatic fallback when models unavailable
- Tool availability checking for browser tasks
- Low confidence prompts for user selection
- Live stats dashboard
- Daily analysis reports

Usage:
    python start_session.py                    # Interactive mode
    python start_session.py --analyze          # Run analysis on today's data
    python start_session.py --analyze --days 7 # Analyze last 7 days
    python start_session.py --health           # Just check system health
"""

import argparse
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "router"))
sys.path.insert(0, str(Path(__file__).parent / "logger"))

from logged_router import LoggedRouter, RouterConfig, Colors
from research_agent import run_analysis
from metrics_tracker import MetricsTracker, print_metrics
from health_check import SystemHealthCheck


def show_help():
    """Show available commands"""
    print(f"""
{Colors.CYAN}Commands:{Colors.RESET}
  [request]    Route a request to a model
  /stats       Show session statistics
  /health      Show system health status
  /models      Show available models
  /analyze     Run pattern analysis
  /feedback    Give feedback on last routing
  /observer    Show observer insights (if enabled)
  /help        Show this help
  /quit        Exit session

{Colors.CYAN}Flags (prefix to request):{Colors.RESET}
  @image       Request includes an image
  @code        Request includes code

{Colors.CYAN}Examples:{Colors.RESET}
  > goto amazon.com
  > @image what's in this picture?
  > @code debug this function
  > write a python function to parse JSON
""")


def interactive_session(config: RouterConfig):
    """Run interactive routing session"""
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}MODEL ROUTER - INTERACTIVE SESSION{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")

    router = LoggedRouter(config)

    # Startup health check
    print(f"\n{Colors.GRAY}Running startup health check...{Colors.RESET}")
    router.show_health()

    status = router.check_health()
    if not status['ready_for_routing']:
        print(f"{Colors.RED}WARNING: System not ready for routing!{Colors.RESET}")
        print(f"{Colors.YELLOW}Start LM Studio and load at least one model.{Colors.RESET}")
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            return

    print(f"\nSession ID: {Colors.BOLD}{router.session_id}{Colors.RESET}")

    # Show observer status
    if router.observer.enabled:
        print(f"Observer: {Colors.GREEN}{router.observer.mode} mode{Colors.RESET} (AI watching logs)")
    else:
        print(f"Observer: {Colors.GRAY}off{Colors.RESET}")

    print(f"Type {Colors.CYAN}/help{Colors.RESET} for commands\n")
    print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}")

    while True:
        try:
            user_input = input(f"\n{Colors.GREEN}>{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/help":
                show_help()

            elif cmd == "/feedback":
                router.collect_feedback(timeout=0)

            elif cmd == "/stats":
                router.show_stats()

            elif cmd == "/health":
                router.show_health()

            elif cmd == "/models":
                router.show_models()

            elif cmd == "/analyze":
                print(f"\n{Colors.GRAY}Running analysis...{Colors.RESET}")
                report_path = run_analysis(print_report=True)
                print(f"\n{Colors.GREEN}Report saved to: {report_path}{Colors.RESET}")

            elif cmd == "/observer":
                router.show_observer_insights()

            else:
                print(f"{Colors.RED}Unknown command: {cmd}{Colors.RESET}")
                print(f"Type {Colors.CYAN}/help{Colors.RESET} for available commands")

            continue

        # Parse flags
        has_image = False
        has_code = False
        request = user_input

        if request.startswith("@image "):
            has_image = True
            request = request[7:]
        elif request.startswith("@code "):
            has_code = True
            request = request[6:]

        # Route the request
        decision, warnings = router.route(request, has_image=has_image, has_code=has_code)

        # Display decision
        print(f"\n  {Colors.BOLD}Model:{Colors.RESET}  {decision.model.value}")
        print(f"  {Colors.BOLD}Type:{Colors.RESET}   {decision.task_type}")

        # Color code confidence
        conf = decision.confidence
        if conf >= 0.85:
            conf_color = Colors.GREEN
        elif conf >= 0.70:
            conf_color = Colors.YELLOW
        else:
            conf_color = Colors.RED
        print(f"  {Colors.BOLD}Conf:{Colors.RESET}   {conf_color}{conf:.0%}{Colors.RESET}")

        if decision.tools_required:
            print(f"  {Colors.BOLD}Tools:{Colors.RESET}  {', '.join(decision.tools_required)}")

        # Complete the request (simulate success for now)
        router.complete_request(
            success=True,
            tokens=0,
            tool_calls=decision.tools_required
        )

        # Collect feedback
        router.collect_feedback()

    # Save session on exit
    print(f"\n{Colors.GRAY}Saving session...{Colors.RESET}")
    router.export_session()

    # Show final stats
    router.show_stats()

    summary = router.get_session_summary()
    print(f"{Colors.GREEN}Session complete!{Colors.RESET}")
    print(f"  Requests: {summary['turn_count']}")
    print(f"  Models used: {', '.join(summary['models_used'])}")


def run_daily_analysis(days: int = 1):
    """Run analysis on routing data"""
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}ROUTING ANALYSIS{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")

    # Show metrics
    tracker = MetricsTracker()
    metrics = tracker.calculate_daily_metrics()

    if metrics.accuracy.total_requests > 0:
        print_metrics(metrics)
    else:
        print(f"\n{Colors.YELLOW}No routing data found for today.{Colors.RESET}")
        print(f"Run an interactive session first to collect data.")
        return

    # Run research agent
    print(f"\n{Colors.GRAY}{'─' * 40}{Colors.RESET}")
    print(f"{Colors.BOLD}RESEARCH AGENT ANALYSIS{Colors.RESET}")
    print(f"{Colors.GRAY}{'─' * 40}{Colors.RESET}")

    report_path = run_analysis(print_report=True)
    print(f"\n{Colors.GREEN}Full report saved to: {report_path}{Colors.RESET}")


def check_health_only():
    """Just check system health"""
    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}SYSTEM HEALTH CHECK{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")

    health = SystemHealthCheck()
    status = health.full_check()

    # LM Studio
    print(f"\n{Colors.BOLD}LM Studio:{Colors.RESET}")
    if status['lmstudio_running']:
        print(f"  Status: {Colors.GREEN}Running{Colors.RESET}")
        print(f"  Models loaded: {status['model_count']}")
        for model in status['models_loaded']:
            print(f"    • {model}")
    else:
        print(f"  Status: {Colors.RED}NOT RUNNING{Colors.RESET}")
        print(f"  {Colors.GRAY}Start LM Studio and load a model.{Colors.RESET}")

    # Playwright
    print(f"\n{Colors.BOLD}Playwright/MCP:{Colors.RESET}")
    if status['playwright_available']:
        print(f"  Status: {Colors.GREEN}Available{Colors.RESET}")
        print(f"  Browser tools: {', '.join(status['browser_tools'])}")
    else:
        print(f"  Status: {Colors.RED}NOT AVAILABLE{Colors.RESET}")
        print(f"  {Colors.GRAY}Run: docker start playwright-mcp{Colors.RESET}")

    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    if status['ready_for_routing']:
        print(f"  {Colors.GREEN}✓ Ready for routing{Colors.RESET}")
    else:
        print(f"  {Colors.RED}✗ NOT ready for routing{Colors.RESET}")

    if status['ready_for_browser']:
        print(f"  {Colors.GREEN}✓ Ready for browser automation{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}⚠ Browser automation unavailable{Colors.RESET}")

    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations per task type:{Colors.RESET}")
    for task_type in ["tool_use", "code_generation", "vision", "simple_chat"]:
        rec = health.get_routing_recommendation(task_type)
        if rec['can_execute']:
            print(f"  {task_type}: {Colors.GREEN}{rec['recommended_model']}{Colors.RESET}")
        else:
            print(f"  {task_type}: {Colors.RED}No model available{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Model Router with Logging, Health Checks, and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_session.py                    # Interactive mode
  python start_session.py --analyze          # Analyze today's data
  python start_session.py --analyze --days 7 # Analyze last 7 days
  python start_session.py --health           # Just check health
        """
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Run analysis instead of interactive session"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=1,
        help="Number of days to analyze (default: 1)"
    )
    parser.add_argument(
        "--health", "-H",
        action="store_true",
        help="Just check system health and exit"
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable feedback collection"
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Disable health checking"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for user prompts (default: 0.70)"
    )
    parser.add_argument(
        "--observer-active",
        action="store_true",
        help="Enable observer in active mode (every request)"
    )
    parser.add_argument(
        "--observer-passive",
        action="store_true",
        help="Enable observer in passive mode (every 5th request)"
    )

    args = parser.parse_args()

    if args.health:
        check_health_only()
    elif args.analyze:
        run_daily_analysis(args.days)
    else:
        # Determine observer mode
        observer_mode = "off"
        if args.observer_active:
            observer_mode = "active"
        elif args.observer_passive:
            observer_mode = "passive"

        config = RouterConfig(
            log_dir="logs",
            collect_feedback=not args.no_feedback,
            check_model_health=not args.no_health_check,
            check_tool_health=not args.no_health_check,
            confidence_threshold=args.confidence_threshold,
            observer_mode=observer_mode
        )
        interactive_session(config)


if __name__ == "__main__":
    main()
