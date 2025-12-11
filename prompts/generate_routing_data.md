# Routing Dataset Generation Prompt

You are generating training data for a model router. The router dispatches user requests to specialized models based on task type.

## Model Fleet

| Model | Size | Strengths | Weaknesses | Speed |
|-------|------|-----------|------------|-------|
| command-r-35b | 20GB | Tool use, browser automation, multi-step workflows, 128K context | Slow for simple tasks | Medium |
| qwen3-coder-30b | 19GB | Code generation, FCPXML, scripting, tool use | Not for chat | Medium |
| ministral-14b-reasoning | 9GB | Fast reasoning, math | Overthinks, loops on tools | Fast |
| qwen3-vl-8b | 6GB | Vision, image analysis, OCR | Everything except vision | Fast |
| gpt-oss-20b | 12GB | Complex reasoning, analysis | Overthinks, slow | Medium |
| gemma-3n-e4b | 4GB | Fast chat, simple responses | Complex tasks, tools | Fast |
| gemma-3-1b | 720MB | Ultra-fast, simple queries | Everything complex | Ultra-fast |

## Task Types

- `tool_use` - Any task requiring tool execution
- `browser_automation` - Web browsing, clicking, typing, screenshots
- `code_generation` - Writing, refactoring, debugging code
- `vision` - Image analysis, OCR, visual understanding
- `math_reasoning` - Calculations, equations, math problems
- `complex_reasoning` - Long explanations, deep analysis
- `reasoning` - Medium reasoning tasks
- `simple_chat` - Greetings, simple questions
- `general_chat` - General conversation

## Available Tools

- browser_navigate, browser_screenshot, browser_click, browser_type
- browser_scroll, browser_wait, browser_snapshot
- file_read, file_write, code_execute, search

## Output Format (JSONL)

```json
{"request": "user request here", "context": {"has_image": false, "has_code": false}, "best_model": "model-id", "task_type": "type", "tools_required": ["tool1", "tool2"], "complexity": "simple|medium|complex", "reasoning": "why this model", "fallback_model": "backup-model-id"}
```

## Generation Rules

1. **CRITICAL**: Route to the model that will EXECUTE best, not just understand
2. Tool use tasks ALWAYS go to command-r-35b (it actually executes tools)
3. Vision tasks ALWAYS go to qwen3-vl-8b (only model that can see images)
4. Code tasks go to qwen3-coder-30b
5. Simple queries go to gemma-3-1b for speed
6. NEVER route tool tasks to ministral-14b (it overthinks and loops)
7. If the task mentions "image", "picture", "screenshot" WITH analysis, use qwen3-vl-8b
8. If the task mentions browser actions, use command-r-35b

## Generate Diverse Examples

Generate 50 examples covering:
- 15 browser/tool use scenarios (navigation, clicking, forms, screenshots)
- 10 code generation scenarios (various languages, debugging, refactoring)
- 8 vision scenarios (image analysis, OCR, visual QA)
- 7 reasoning scenarios (math, analysis, explanations)
- 10 simple chat/general scenarios

Make requests varied and realistic:
- Different phrasings ("go to X", "open X", "navigate to X", "browse X")
- Different complexity levels
- Edge cases (ambiguous requests)
- Real-world tasks users actually ask

## Example Outputs

```json
{"request": "open youtube and search for python tutorials", "best_model": "command-r-35b", "task_type": "browser_automation", "tools_required": ["browser_navigate", "browser_type", "browser_click"], "complexity": "medium", "reasoning": "Multi-step browser workflow requires reliable tool execution", "fallback_model": "qwen3-coder-30b"}
{"request": "convert this typescript to python", "context": {"has_code": true}, "best_model": "qwen3-coder-30b", "task_type": "code_generation", "tools_required": [], "complexity": "medium", "reasoning": "Code conversion is pure code task - code specialist optimal", "fallback_model": "command-r-35b"}
{"request": "what breed is this dog", "context": {"has_image": true}, "best_model": "qwen3-vl-8b", "task_type": "vision", "tools_required": [], "complexity": "simple", "reasoning": "Image classification requires vision model", "fallback_model": "gpt-oss-20b"}
{"request": "hi there", "best_model": "gemma-3-1b", "task_type": "simple_chat", "tools_required": [], "complexity": "simple", "reasoning": "Simple greeting - ultra-fast model most efficient", "fallback_model": "gemma-3n-e4b"}
```

Now generate 50 diverse, realistic examples in JSONL format:
