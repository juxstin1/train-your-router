"""
Data Collection Script for Model Router Training
Wraps LM Studio API calls to collect real routing decisions.

Usage:
    python collect_decisions.py --interactive
    python collect_decisions.py --batch requests.txt
"""

import argparse
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from model_router import ModelRouter, RequestContext, RoutingDecision, ModelID

# LM Studio default endpoint
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Map model IDs to LM Studio model names (adjust these to match your loaded models)
MODEL_TO_LMSTUDIO = {
    "command-r-35b": "command-r-35b-08-2024",
    "qwen3-coder-30b": "qwen3-coder-30b",
    "ministral-14b-reasoning": "ministral-14b",
    "qwen3-vl-8b": "qwen3-vl-8b",
    "gpt-oss-20b": "gpt-oss-20b",
    "gemma-3n-e4b": "gemma-3n-e4b",
    "gemma-3-1b": "gemma-3-1b",
}


class DecisionCollector:
    """
    Collects routing decisions for training data.
    Can run in interactive mode or batch mode.
    """

    def __init__(self, output_dir: str = "../datasets"):
        self.router = ModelRouter(log_decisions=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track decisions with human feedback
        self.validated_decisions: List[Dict[str, Any]] = []

    def route_and_display(self, request: str, has_image: bool = False, has_code: bool = False) -> RoutingDecision:
        """Route request and display decision for validation"""
        context = RequestContext(
            request=request,
            has_image=has_image,
            has_code=has_code
        )

        decision = self.router.route(context)

        print("\n" + "=" * 50)
        print(f"REQUEST: {request}")
        print("-" * 50)
        print(f"  Model:      {decision.model.value}")
        print(f"  Task Type:  {decision.task_type}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Tools:      {decision.tools_required}")
        print(f"  Reasoning:  {decision.reasoning}")
        print(f"  Fallback:   {decision.fallback_model.value if decision.fallback_model else 'None'}")
        print("=" * 50)

        return decision

    def interactive_mode(self):
        """Run interactive collection with human validation"""
        print("\n" + "=" * 60)
        print("MODEL ROUTER - INTERACTIVE DATA COLLECTION")
        print("=" * 60)
        print("\nCommands:")
        print("  [Enter]     - Accept routing decision")
        print("  [m]odel     - Override model choice")
        print("  [t]ask      - Override task type")
        print("  [s]kip      - Skip this example")
        print("  [q]uit      - Save and exit")
        print("  [i]mage     - Toggle has_image flag")
        print("  [c]ode      - Toggle has_code flag")
        print("-" * 60)

        has_image = False
        has_code = False

        while True:
            # Get request
            print(f"\nFlags: has_image={has_image}, has_code={has_code}")
            request = input("\nEnter request (or 'q' to quit): ").strip()

            if request.lower() == 'q':
                break
            if request.lower() == 'i':
                has_image = not has_image
                print(f"has_image toggled to {has_image}")
                continue
            if request.lower() == 'c':
                has_code = not has_code
                print(f"has_code toggled to {has_code}")
                continue
            if not request:
                continue

            # Get routing decision
            decision = self.route_and_display(request, has_image, has_code)

            # Get validation
            action = input("\n[Enter=accept, m=model, t=task, s=skip]: ").strip().lower()

            if action == 's':
                print("Skipped.")
                continue

            if action == 'm':
                print("\nAvailable models:")
                for i, m in enumerate(ModelID):
                    print(f"  {i}: {m.value}")
                try:
                    idx = int(input("Select model number: "))
                    decision.model = list(ModelID)[idx]
                    print(f"Model changed to: {decision.model.value}")
                except (ValueError, IndexError):
                    print("Invalid selection, keeping original")

            if action == 't':
                new_task = input("Enter task type: ").strip()
                if new_task:
                    decision.task_type = new_task
                    print(f"Task type changed to: {decision.task_type}")

            # Build validated entry
            entry = {
                "request": request,
                "context": {
                    "has_image": has_image,
                    "has_code": has_code
                },
                "best_model": decision.model.value,
                "task_type": decision.task_type,
                "tools_required": decision.tools_required,
                "complexity": self._estimate_complexity(request),
                "reasoning": decision.reasoning,
                "fallback_model": decision.fallback_model.value if decision.fallback_model else None,
                "validated": True,
                "timestamp": datetime.now().isoformat()
            }

            self.validated_decisions.append(entry)
            print(f"Saved! Total collected: {len(self.validated_decisions)}")

            # Reset flags after each entry
            has_image = False
            has_code = False

        # Save on exit
        self._save_validated()

    def batch_mode(self, input_file: str):
        """Process a batch of requests from file"""
        requests = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requests.append(line)

        print(f"\nProcessing {len(requests)} requests...")

        for request in requests:
            decision = self.route_and_display(request)

            entry = {
                "request": request,
                "context": {
                    "has_image": False,
                    "has_code": False
                },
                "best_model": decision.model.value,
                "task_type": decision.task_type,
                "tools_required": decision.tools_required,
                "complexity": self._estimate_complexity(request),
                "reasoning": decision.reasoning,
                "fallback_model": decision.fallback_model.value if decision.fallback_model else None,
                "validated": False,
                "timestamp": datetime.now().isoformat()
            }
            self.validated_decisions.append(entry)

        self._save_validated()

    def _estimate_complexity(self, request: str) -> str:
        word_count = len(request.split())
        if word_count <= 5:
            return "simple"
        elif word_count <= 20:
            return "medium"
        return "complex"

    def _save_validated(self):
        """Save validated decisions to JSONL"""
        if not self.validated_decisions:
            print("No decisions to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"routing_decisions_{timestamp}.jsonl"

        with open(output_file, 'w') as f:
            for entry in self.validated_decisions:
                f.write(json.dumps(entry) + '\n')

        print(f"\nSaved {len(self.validated_decisions)} decisions to {output_file}")

        # Also append to master file
        master_file = self.output_dir / "routing_decisions_all.jsonl"
        with open(master_file, 'a') as f:
            for entry in self.validated_decisions:
                f.write(json.dumps(entry) + '\n')

        print(f"Also appended to {master_file}")


def main():
    parser = argparse.ArgumentParser(description="Collect routing decisions for training")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with validation")
    parser.add_argument("--batch", "-b", type=str, help="Batch mode: process requests from file")
    parser.add_argument("--output", "-o", type=str, default="../datasets", help="Output directory")

    args = parser.parse_args()

    collector = DecisionCollector(output_dir=args.output)

    if args.batch:
        collector.batch_mode(args.batch)
    else:
        # Default to interactive
        collector.interactive_mode()


if __name__ == "__main__":
    main()
