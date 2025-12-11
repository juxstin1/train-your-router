"""
Dataset Generation Script
Uses Claude/local LLM to generate training data from prompts.

Can use:
- Anthropic API (Claude)
- Local LLM via LM Studio
- OpenAI-compatible API
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import time


def load_prompt(prompt_file: str) -> str:
    """Load generation prompt from file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def extract_jsonl(response: str) -> List[Dict[str, Any]]:
    """Extract JSONL entries from LLM response"""
    entries = []

    # Try to find JSON objects in the response
    # Handle both ```json blocks and raw JSONL

    # First, try to extract from code blocks
    code_blocks = re.findall(r'```(?:json|jsonl)?\n(.*?)```', response, re.DOTALL)

    text_to_parse = '\n'.join(code_blocks) if code_blocks else response

    # Parse each line as JSON
    for line in text_to_parse.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue

        try:
            entry = json.loads(line)
            entries.append(entry)
        except json.JSONDecodeError:
            # Try to find JSON object in line
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    entry = json.loads(match.group())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

    return entries


def generate_with_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Generate using Anthropic API"""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def generate_with_openai_compatible(
    prompt: str,
    base_url: str = "http://localhost:1234/v1",
    model: str = "local-model",
    api_key: str = "not-needed"
) -> str:
    """Generate using OpenAI-compatible API (LM Studio, etc.)"""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=8000,
        temperature=0.7
    )

    return response.choices[0].message.content


def validate_routing_entry(entry: Dict[str, Any]) -> bool:
    """Validate a routing dataset entry"""
    required_fields = ["request", "best_model", "task_type"]
    valid_models = [
        "command-r-35b", "qwen3-coder-30b", "ministral-14b-reasoning",
        "qwen3-vl-8b", "gpt-oss-20b", "gemma-3n-e4b", "gemma-3-1b"
    ]

    # Check required fields
    for field in required_fields:
        if field not in entry:
            return False

    # Check model is valid
    if entry["best_model"] not in valid_models:
        return False

    return True


def validate_tool_use_entry(entry: Dict[str, Any]) -> bool:
    """Validate a tool use SFT entry"""
    if "conversations" not in entry:
        return False

    convos = entry["conversations"]
    if not isinstance(convos, list) or len(convos) < 2:
        return False

    # Check for at least one tool call in assistant responses
    has_tool_call = False
    for turn in convos:
        if turn.get("from") == "gpt" and "<tool_call>" in turn.get("value", ""):
            has_tool_call = True
            break

    return has_tool_call


def main():
    parser = argparse.ArgumentParser(description="Generate training datasets")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                       help="Path to prompt file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--backend", "-b", type=str, default="anthropic",
                       choices=["anthropic", "lmstudio", "openai"],
                       help="LLM backend to use")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Model to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1",
                       help="Base URL for OpenAI-compatible API")
    parser.add_argument("--iterations", "-n", type=int, default=1,
                       help="Number of generation iterations")
    parser.add_argument("--validate", "-v", type=str, default=None,
                       choices=["routing", "tool_use"],
                       help="Validation type for entries")

    args = parser.parse_args()

    # Load prompt
    prompt = load_prompt(args.prompt)
    print(f"Loaded prompt from {args.prompt}")

    all_entries = []

    for i in range(args.iterations):
        print(f"\nGeneration iteration {i+1}/{args.iterations}...")

        # Generate
        if args.backend == "anthropic":
            model = args.model or "claude-sonnet-4-20250514"
            response = generate_with_anthropic(prompt, model)
        elif args.backend == "lmstudio":
            model = args.model or "local-model"
            response = generate_with_openai_compatible(
                prompt, args.base_url, model
            )
        elif args.backend == "openai":
            import openai
            model = args.model or "gpt-4"
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000
            )
            response = resp.choices[0].message.content

        # Extract entries
        entries = extract_jsonl(response)
        print(f"Extracted {len(entries)} entries")

        # Validate if requested
        if args.validate:
            validator = validate_routing_entry if args.validate == "routing" else validate_tool_use_entry
            valid_entries = [e for e in entries if validator(e)]
            print(f"Valid entries: {len(valid_entries)}/{len(entries)}")
            entries = valid_entries

        all_entries.extend(entries)

        # Brief pause between iterations
        if i < args.iterations - 1:
            time.sleep(2)

    # Deduplicate by request (for routing data)
    if args.validate == "routing":
        seen_requests = set()
        unique_entries = []
        for entry in all_entries:
            req = entry.get("request", "")
            if req not in seen_requests:
                seen_requests.add(req)
                unique_entries.append(entry)
        print(f"After dedup: {len(unique_entries)} unique entries")
        all_entries = unique_entries

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(all_entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
