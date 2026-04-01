"""
Memory and Session Prompt Templates.

Prompts for conversation memory management:
- Progressive summarization of conversation history
- Session context generation for returning users
- Memory compression for long conversations

These prompts enable efficient context management for multi-turn
conversations while preserving important information.

Usage:
    from hybridrag.prompts import (
        MEMORY_SUMMARIZATION_PROMPT,
        SESSION_CONTEXT_PROMPT,
    )

    # Summarize conversation history
    prompt = MEMORY_SUMMARIZATION_PROMPT.format(
        existing_summary=previous_summary,
        new_messages=recent_messages,
        max_tokens=500
    )
"""

from __future__ import annotations

from typing import Final

MEMORY_SUMMARIZATION_PROMPT: Final[
    str
] = """You are a conversation summarizer. Create a progressive summary of the conversation.

## Task
Update the existing summary with new conversation turns while preserving important context.

## Existing Summary
{existing_summary}

## New Messages
{new_messages}

## Instructions
1. Integrate new information into the existing summary
2. Preserve:
   - Key decisions and conclusions
   - User preferences and requirements
   - Important facts and context
   - Action items and next steps
   - Entities and relationships discussed
3. Remove:
   - Redundant information already captured
   - Chitchat and pleasantries
   - Superseded information (keep latest)
4. Keep the summary under {max_tokens} tokens
5. Use bullet points for clarity
6. Maintain chronological flow of important events
7. Mark any unresolved questions or pending items

## Output Format
```
## Conversation Summary

### Key Topics Discussed
- Topic 1: Brief description
- Topic 2: Brief description

### Important Decisions/Facts
- Decision/Fact 1
- Decision/Fact 2

### User Context
- Preferences: ...
- Requirements: ...

### Pending Items
- Unresolved question 1
- Action item 1

### Last Context
[Most recent relevant context for continuation]
```

## Updated Summary
"""


MEMORY_SUMMARIZATION_PROMPT_LITE: Final[
    str
] = """Summarize this conversation in 2-3 sentences, preserving the most critical context.

## Conversation
{conversation}

## Focus On
- What was the user trying to accomplish?
- What was the outcome or current state?
- Any key decisions or preferences mentioned?

## Summary (2-3 sentences)
"""


SESSION_CONTEXT_PROMPT: Final[
    str
] = """Generate a brief context reminder for a returning user.

## Previous Session Summary
{session_summary}

## Time Since Last Session
{time_elapsed}

## Instructions
Create a natural, friendly context reminder that:
1. Acknowledges the user's return
2. Briefly reminds them of the previous context
3. Asks if they'd like to continue or start fresh
4. Keeps it concise (2-3 sentences)

## Example Output
"Welcome back! Last time we were discussing [topic] and you mentioned [key point]. Would you like to continue from there or explore something new?"

## Context Reminder
"""


CONVERSATION_COMPRESSION_PROMPT: Final[
    str
] = """Compress this conversation history while preserving essential context.

## Conversation History
{conversation_history}

## Target Token Count
{target_tokens}

## Compression Strategy
1. Keep the most recent 2-3 exchanges verbatim
2. Summarize earlier exchanges into key points
3. Preserve:
   - User's original question/goal
   - Key facts and decisions
   - Technical details if relevant
   - Any corrections or clarifications
4. Remove:
   - Repetitive acknowledgments
   - Verbose explanations (condense to key points)
   - Failed attempts (unless instructive)

## Output Format
```
[SUMMARY OF EARLIER CONTEXT]
- Key point 1
- Key point 2

[RECENT EXCHANGES - VERBATIM]
User: ...
Assistant: ...
```

## Compressed Conversation
"""


# Memory management constants
DEFAULT_SUMMARY_MAX_TOKENS: Final[int] = 500
LITE_SUMMARY_MAX_TOKENS: Final[int] = 100
COMPRESSION_TARGET_RATIO: Final[float] = 0.3  # Target 30% of original size
