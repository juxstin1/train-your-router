# Tool Use SFT Dataset Generation Prompt

You are generating training data for teaching models to use tools correctly. The goal is models that:
- Execute tools IMMEDIATELY without overthinking
- Use proper tool sequences
- Don't ask permission for simple tasks
- Don't loop on themselves

## Anti-Patterns to Avoid

BAD (overthinking):
```
User: goto google.com
Assistant: I'll help you navigate to Google. First, let me use the browser_navigate tool to open the website. This will allow us to...
```

BAD (asking permission):
```
User: take a screenshot
Assistant: Would you like me to take a screenshot of the current page? I can use the browser_screenshot tool to capture...
```

BAD (excessive explanation):
```
User: click the submit button
Assistant: I understand you want to click the submit button. I'll need to use the browser_click tool with the appropriate selector...
```

## Good Patterns

GOOD (immediate execution):
```
User: goto google.com
Assistant: <tool_call>browser_navigate(url="https://google.com")</tool_call>
```

GOOD (multi-step without filler):
```
User: search amazon for headphones
Assistant: <tool_call>browser_navigate(url="https://amazon.com")</tool_call>
<tool_call>browser_wait(seconds=2)</tool_call>
<tool_call>browser_type(element="search box", text="headphones")</tool_call>
<tool_call>browser_click(element="search button")</tool_call>
```

GOOD (minimal acknowledgment when needed):
```
User: find the cheapest option on this page
Assistant: Looking for the cheapest option.
<tool_call>browser_snapshot()</tool_call>
[After seeing snapshot]
The cheapest option is $24.99 - the "Basic Wireless Earbuds" in the third row.
```

## Output Format

Generate conversations in ShareGPT format:

```json
{
  "conversations": [
    {"from": "system", "value": "You have browser automation tools. Execute tasks directly without asking permission or over-explaining."},
    {"from": "human", "value": "goto amazon.com"},
    {"from": "gpt", "value": "<tool_call>browser_navigate(url=\"https://amazon.com\")</tool_call>"}
  ]
}
```

For multi-turn with tool results:

```json
{
  "conversations": [
    {"from": "system", "value": "You have browser automation tools. Execute tasks directly."},
    {"from": "human", "value": "go to reddit and screenshot the front page"},
    {"from": "gpt", "value": "<tool_call>browser_navigate(url=\"https://reddit.com\")</tool_call>"},
    {"from": "tool", "value": "Navigation successful. Page loaded."},
    {"from": "gpt", "value": "<tool_call>browser_wait(seconds=2)</tool_call>"},
    {"from": "tool", "value": "Waited 2 seconds."},
    {"from": "gpt", "value": "<tool_call>browser_screenshot()</tool_call>"},
    {"from": "tool", "value": "Screenshot saved to screenshot_001.png"},
    {"from": "gpt", "value": "Done. Screenshot saved."}
  ]
}
```

## Available Tools

```
browser_navigate(url: str) - Navigate to URL
browser_click(element: str, ref: str) - Click element
browser_type(element: str, ref: str, text: str) - Type text
browser_screenshot() - Take screenshot
browser_snapshot() - Get page accessibility tree
browser_wait(seconds: int) - Wait
browser_scroll(direction: str) - Scroll page
browser_press_key(key: str) - Press keyboard key
```

## Task Categories to Generate

### 1. Single Tool Calls (15 examples)
Simple, immediate execution:
- "goto [website]"
- "take a screenshot"
- "click [element]"
- "scroll down"
- "type [text]"

### 2. Multi-Step Workflows (15 examples)
Sequential tool chains:
- Navigate + search + screenshot
- Navigate + fill form + submit
- Navigate + scroll + find element + click
- Navigate + wait + interact

### 3. Error Recovery (10 examples)
Handle problems without looping:
- Page not loading -> try alternate URL
- Element not found -> take snapshot, reassess
- Timeout -> brief wait and retry once

### 4. Context-Dependent Actions (10 examples)
Use information from previous steps:
- After snapshot, identify correct element
- After screenshot, describe what's visible
- Use page content to decide next action

## Generation Rules

1. Assistant responses should be 80%+ tool calls, minimal text
2. Never include phrases like "I'll help you", "Let me", "First, I need to"
3. When text is needed, keep it under 10 words
4. Tool calls should be syntactically correct
5. Multi-step tasks should chain tools logically
6. Include realistic websites and tasks
7. Vary the phrasing of user requests

## System Prompts to Use

Rotate through these:
- "You have browser automation tools. Execute tasks directly without asking permission."
- "You control a web browser. When asked to do something, do it immediately."
- "Browser assistant. Act on requests without explaining or asking for confirmation."
- "You can control the browser. Just execute the requested actions."

Now generate 50 tool use training examples in ShareGPT JSONL format:
