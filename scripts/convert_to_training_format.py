"""
Convert routing dataset to SFT format for training gemma-3-1b as router.

Input: Routing decision JSONL
Output: Chat format suitable for QLoRA training
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a model router. Given a user request, determine which model should handle it.

Available models:
- command-r-35b: Tool use, browser automation, multi-step workflows (128K context)
- qwen3-coder-30b: Code generation, scripting, debugging
- ministral-14b-reasoning: Fast reasoning, math calculations
- qwen3-vl-8b: Vision tasks, image analysis, OCR
- gpt-oss-20b: Complex reasoning, deep analysis
- gemma-3n-e4b: Fast general chat
- gemma-3-1b: Ultra-fast simple queries

Respond with ONLY the model name, nothing else."""


DETAILED_SYSTEM_PROMPT = """You are a model router for a multi-model AI system. Analyze incoming requests and route them to the optimal model.

## Models Available

| Model | Best For | Avoid For |
|-------|----------|-----------|
| command-r-35b | Browser automation, tool execution, multi-step tasks | Simple queries |
| qwen3-coder-30b | Code writing, debugging, scripting | Non-code tasks |
| ministral-14b-reasoning | Math, quick analysis | Tool execution |
| qwen3-vl-8b | Image analysis, OCR, vision | Everything else |
| gpt-oss-20b | Deep reasoning, complex analysis | Simple tasks |
| gemma-3n-e4b | General chat, medium queries | Complex tasks |
| gemma-3-1b | Simple greetings, basic questions | Anything complex |

## Response Format
Respond with a JSON object:
{"model": "model-name", "task_type": "type", "tools": ["tool1", "tool2"]}"""


def convert_to_simple_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to simple classification format (model name only)"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("best_model", "")

        if not request or not model:
            continue

        # Format for training
        training_entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": model}
            ]
        }

        converted.append(training_entry)

    return converted


def convert_to_detailed_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to detailed JSON output format"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("best_model", "")
        task_type = entry.get("task_type", "")
        tools = entry.get("tools_required", [])

        if not request or not model:
            continue

        # Build response JSON
        response = {
            "model": model,
            "task_type": task_type,
            "tools": tools
        }

        training_entry = {
            "messages": [
                {"role": "system", "content": DETAILED_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": json.dumps(response)}
            ]
        }

        converted.append(training_entry)

    return converted


def convert_to_sharegpt_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to ShareGPT format (for axolotl, etc.)"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("best_model", "")

        if not request or not model:
            continue

        training_entry = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": request},
                {"from": "gpt", "value": model}
            ]
        }

        converted.append(training_entry)

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert routing data to training format")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input JSONL file with routing decisions")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--format", "-f", type=str, default="simple",
                       choices=["simple", "detailed", "sharegpt"],
                       help="Output format")

    args = parser.parse_args()

    # Load input
    entries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.input}")

    # Convert
    if args.format == "simple":
        converted = convert_to_simple_format(entries)
    elif args.format == "detailed":
        converted = convert_to_detailed_format(entries)
    else:
        converted = convert_to_sharegpt_format(entries)

    print(f"Converted {len(converted)} entries to {args.format} format")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Saved to {args.output}")

    # Print sample
    if converted:
        print("\nSample entry:")
        print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    main()
