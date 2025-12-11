"""
Convert routing logs to SFT format for training gemma-3-1b as router.

Input: Routing logs from logs/routing_*.jsonl
Output: Chat format suitable for QLoRA training

Usage:
    python convert_to_training_format.py --input ../logs --output ../datasets/train.jsonl
    python convert_to_training_format.py --input ../logs --output ../datasets --split 0.1
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


# System prompts for different training formats
SIMPLE_SYSTEM_PROMPT = """You are a routing classifier. Given a user request, decide which model should handle it.

Available models:
- gpt-oss-20b: Tool use, browser automation, complex reasoning (128K context)
- qwen3-coder-30b: Code generation, FCPXML, scripting, debugging
- ministral-14b-reasoning: Fast reasoning, math calculations
- qwen3-vl-8b: Vision tasks, image analysis, OCR (images only)
- gemma-3n-e4b: General chat, medium complexity
- gemma-3-1b: Simple queries, greetings

Respond with ONLY: ROUTE_TO=<model_name>"""


DETAILED_SYSTEM_PROMPT = """You are a routing classifier for a multi-model AI system.

## Models
- gpt-oss-20b: tool_use, browser_automation, complex_reasoning
- qwen3-coder-30b: code_generation, scripting
- ministral-14b-reasoning: reasoning, math
- qwen3-vl-8b: vision (ONLY for images)
- gemma-3n-e4b: general_chat
- gemma-3-1b: simple_chat

## Output Format
TASK_TYPE=<type>
TOOL_REQUIRED=<yes|no>
ROUTE_TO=<model>"""


def load_routing_logs(input_path: Path) -> List[Dict[str, Any]]:
    """Load routing logs from directory or single file"""
    entries = []

    if input_path.is_dir():
        # Load all routing_*.jsonl files
        for log_file in sorted(input_path.glob("routing_*.jsonl")):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    else:
        # Single file
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return entries


def filter_high_quality(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter for high-quality training examples:
    - Has user feedback
    - Either thumbs up OR thumbs down with correction
    """
    filtered = []

    for entry in entries:
        feedback = entry.get("feedback", {})

        if not feedback:
            continue

        rating = feedback.get("user_rating")

        if rating == 1:
            # Thumbs up - use the routed model
            filtered.append({
                "request": entry.get("request", {}).get("text", ""),
                "model": entry.get("routing", {}).get("model_chosen", ""),
                "task_type": entry.get("routing", {}).get("task_type", ""),
                "tools": entry.get("routing", {}).get("tools_required", []),
                "confidence": entry.get("routing", {}).get("confidence", 0),
                "source": "validated"
            })
        elif rating == -1 and feedback.get("correct_model"):
            # Thumbs down with correction - use the corrected model
            filtered.append({
                "request": entry.get("request", {}).get("text", ""),
                "model": feedback.get("correct_model"),
                "task_type": entry.get("routing", {}).get("task_type", ""),
                "tools": [],  # Unknown for corrections
                "confidence": 1.0,  # User explicitly chose
                "source": "corrected"
            })

    return filtered


def convert_to_simple_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to simple ROUTE_TO format"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("model", "")

        if not request or not model:
            continue

        training_entry = {
            "messages": [
                {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": f"ROUTE_TO={model}"}
            ]
        }
        converted.append(training_entry)

    return converted


def convert_to_detailed_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to detailed format with task type and tool requirement"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("model", "")
        task_type = entry.get("task_type", "unknown")
        tools = entry.get("tools", [])

        if not request or not model:
            continue

        tool_required = "yes" if tools else "no"

        response = f"TASK_TYPE={task_type}\nTOOL_REQUIRED={tool_required}\nROUTE_TO={model}"

        training_entry = {
            "messages": [
                {"role": "system", "content": DETAILED_SYSTEM_PROMPT},
                {"role": "user", "content": request},
                {"role": "assistant", "content": response}
            ]
        }
        converted.append(training_entry)

    return converted


def convert_to_sharegpt_format(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to ShareGPT format (for axolotl, etc.)"""
    converted = []

    for entry in entries:
        request = entry.get("request", "")
        model = entry.get("model", "")

        if not request or not model:
            continue

        training_entry = {
            "conversations": [
                {"from": "system", "value": SIMPLE_SYSTEM_PROMPT},
                {"from": "human", "value": request},
                {"from": "gpt", "value": f"ROUTE_TO={model}"}
            ]
        }
        converted.append(training_entry)

    return converted


def split_train_val(
    entries: List[Dict[str, Any]],
    val_ratio: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into train and validation sets"""
    random.shuffle(entries)

    split_idx = int(len(entries) * (1 - val_ratio))
    train = entries[:split_idx]
    val = entries[split_idx:]

    return train, val


def save_jsonl(entries: List[Dict[str, Any]], path: Path):
    """Save entries as JSONL"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def print_stats(entries: List[Dict[str, Any]], name: str = ""):
    """Print statistics about the dataset"""
    if not entries:
        print(f"{name}: 0 entries")
        return

    # Count by model
    model_counts = {}
    for e in entries:
        model = e.get("model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1

    # Count by source
    source_counts = {}
    for e in entries:
        source = e.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n{name}: {len(entries)} entries")
    print("  By model:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model}: {count}")
    print("  By source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert routing logs to SFT training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all logs to single training file
  python convert_to_training_format.py -i ../logs -o ../datasets/train.jsonl

  # Split into train/val
  python convert_to_training_format.py -i ../logs -o ../datasets --split 0.1

  # Use detailed format with task types
  python convert_to_training_format.py -i ../logs -o ../datasets/train.jsonl -f detailed
        """
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input directory with routing_*.jsonl or single file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output file or directory (if --split)")
    parser.add_argument("--format", "-f", type=str, default="simple",
                       choices=["simple", "detailed", "sharegpt"],
                       help="Output format (default: simple)")
    parser.add_argument("--split", "-s", type=float, default=0,
                       help="Validation split ratio (e.g., 0.1 for 10%)")
    parser.add_argument("--include-unvalidated", action="store_true",
                       help="Include entries without user feedback")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splitting")

    args = parser.parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load logs
    print(f"Loading from {input_path}...")
    raw_entries = load_routing_logs(input_path)
    print(f"Loaded {len(raw_entries)} raw log entries")

    # Filter for high-quality examples
    if args.include_unvalidated:
        # Include all entries, use routed model as label
        entries = []
        for entry in raw_entries:
            entries.append({
                "request": entry.get("request", {}).get("text", ""),
                "model": entry.get("routing", {}).get("model_chosen", ""),
                "task_type": entry.get("routing", {}).get("task_type", ""),
                "tools": entry.get("routing", {}).get("tools_required", []),
                "confidence": entry.get("routing", {}).get("confidence", 0),
                "source": "unvalidated"
            })
    else:
        entries = filter_high_quality(raw_entries)

    print_stats(entries, "Filtered dataset")

    if not entries:
        print("\nNo high-quality entries found!")
        print("Run interactive sessions and provide feedback (y/n) to generate training data.")
        return

    # Convert to training format
    print(f"\nConverting to {args.format} format...")

    if args.format == "simple":
        converted = convert_to_simple_format(entries)
    elif args.format == "detailed":
        converted = convert_to_detailed_format(entries)
    else:
        converted = convert_to_sharegpt_format(entries)

    print(f"Converted {len(converted)} entries")

    # Split or save
    if args.split > 0:
        train, val = split_train_val(converted, args.split)

        train_path = output_path / "train.jsonl"
        val_path = output_path / "val.jsonl"

        save_jsonl(train, train_path)
        save_jsonl(val, val_path)

        print(f"\nSaved:")
        print(f"  Train: {train_path} ({len(train)} examples)")
        print(f"  Val:   {val_path} ({len(val)} examples)")
    else:
        save_jsonl(converted, output_path)
        print(f"\nSaved: {output_path} ({len(converted)} examples)")

    # Print sample
    if converted:
        print("\n" + "="*50)
        print("Sample training entry:")
        print("="*50)
        print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    main()
