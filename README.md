# Train Your Router

Generate high-quality SFT datasets to train a lightweight LLM router. Route requests to specialized local models based on task type.

## What This Does

1. **Rules-based router** dispatches requests to the right model
2. **Logs every decision** with user feedback (thumbs up/down)
3. **Collects corrections** when routing is wrong
4. **Generates training data** for a small router model (gemma-3-1b)
5. **Train once, route forever** - replace rules with your trained model

## Model Fleet (Customizable)

| Model | Size | Best For |
|-------|------|----------|
| gpt-oss-20b | 12GB | Tool use, browser automation, 128K context |
| qwen3-coder-30b | 19GB | Code generation, FCPXML, scripting |
| ministral-14b-reasoning | 9GB | Fast reasoning, math |
| qwen3-vl-8b | 6GB | Vision only |
| gemma-3n-e4b | 4GB | Fast chat |
| gemma-3-1b | 720MB | Ultra-fast, simple queries, **ROUTER TARGET** |

## Quick Start

### 1. Check System Health

```bash
python start_session.py --health
```

Verifies LM Studio is running and which models are loaded.

### 2. Run Interactive Session

```bash
# Basic session
python start_session.py

# With AI observer watching for patterns
python start_session.py --observer-active
```

### 3. Route Requests & Give Feedback

```
> goto amazon.com              # → gpt-oss-20b (tool_use)
> write a python function      # → qwen3-coder-30b (code)
> explain transformers         # → ministral-14b (reasoning)
> hello                        # → gemma-3-1b (simple_chat)
> @image what's this?          # → qwen3-vl-8b (vision)
```

After each routing:
- `y` = correct routing (thumbs up)
- `n` = wrong routing (thumbs down) → specify correct model
- `Enter` = skip feedback

### 4. View Stats & Analyze

```
> /stats      # Session accuracy dashboard
> /health     # Model availability
> /analyze    # Pattern analysis report
> /observer   # AI observer insights (if enabled)
```

### 5. Train Your Router

Once you have 200+ validated examples:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b",
    max_seq_length=512,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
)

# Train on logs/routing_*.jsonl
```

## Project Structure

```
train-your-router/
├── start_session.py          # Main entry point
├── router/
│   ├── model_router.py       # Rules-based router
│   ├── logged_router.py      # Router with logging & health checks
│   └── health_check.py       # Model/tool availability checking
├── logger/
│   ├── core_logger.py        # JSONL logging
│   ├── feedback_prompt.py    # Thumbs up/down collection
│   ├── session_tracker.py    # Multi-turn conversation tracking
│   ├── pattern_detector.py   # Find misrouting patterns
│   ├── metrics_tracker.py    # Accuracy metrics
│   ├── research_agent.py     # Daily analysis reports
│   └── log_observer.py       # AI observer (gemma-3-1b watches logs)
├── datasets/
│   └── seed_routing_data.jsonl
├── prompts/
│   ├── generate_routing_data.md
│   └── generate_tool_use_sft.md
├── scripts/
│   ├── generate_dataset.py
│   └── convert_to_training_format.py
└── logs/                     # Generated data lives here
```

## CLI Options

```bash
python start_session.py [OPTIONS]

Options:
  --health              Check system health and exit
  --analyze             Run analysis on collected data
  --days N              Days of data to analyze (default: 1)
  --no-feedback         Disable feedback collection
  --no-health-check     Disable model health checking
  --confidence-threshold FLOAT  Prompt user below this confidence (default: 0.70)
  --observer-active     AI observer on every request
  --observer-passive    AI observer every 5th request
```

## Features

- **Health Checks** - Verify models loaded before routing
- **Automatic Fallbacks** - If primary model unavailable, use fallback chain
- **Low Confidence Prompts** - Ask user when router is uncertain
- **Live Stats** - Real-time accuracy dashboard
- **AI Observer** - Small model watches logs and flags patterns
- **Pattern Detection** - Find systematic misroutings
- **Research Reports** - Daily analysis with improvement suggestions

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai/) running on `localhost:1234`
- At least one model loaded
- (Optional) Playwright Docker for browser automation

## Customization

Edit `router/model_router.py` to:
- Add your own models to `MODEL_REGISTRY`
- Modify trigger patterns (`TOOL_USE_TRIGGERS`, `CODE_TRIGGERS`, etc.)
- Adjust routing priorities

Edit `router/health_check.py` to:
- Update `FALLBACK_CHAINS` for your model fleet

## License

MIT
