"""
Topic Extraction Prompt Templates.

Prompts for semantic topic/tag extraction from documents:
- Single document topic extraction
- Batch processing for multiple documents
- Topic clustering and taxonomy building

These prompts help categorize and tag content for improved
search and organization in RAG systems.

Usage:
    from hybridrag.prompts import (
        TOPIC_EXTRACTION_PROMPT,
        BATCH_TOPIC_EXTRACTION_PROMPT,
        TOPIC_CLUSTERING_PROMPT,
    )

    # Extract topics from a document
    prompt = TOPIC_EXTRACTION_PROMPT.format(
        text=document_text,
        max_topics=5,
        existing_taxonomy=taxonomy_list
    )
"""

from __future__ import annotations

from typing import Final

TOPIC_EXTRACTION_PROMPT: Final[
    str
] = """Extract semantic topics/tags from this text for search and categorization.

## Task
Identify 3-{max_topics} topics that best describe the content of this text.

## Text
```
{text}
```

## Existing Taxonomy (use these when applicable)
{existing_taxonomy}

## Instructions
1. Extract topics that capture the main themes
2. Use existing taxonomy terms when they fit
3. Create new topics only if necessary
4. Topics should be:
   - Specific enough to be useful for search
   - General enough to group related content
   - Lowercase, hyphenated for multi-word (e.g., "machine-learning")
5. Include both:
   - Primary topics (main subject matter)
   - Secondary topics (related concepts mentioned)
6. Assign confidence scores (0.0-1.0)

## Output Format (JSON)
```json
{{
  "primary_topics": [
    {{"topic": "topic-name", "confidence": 0.95}}
  ],
  "secondary_topics": [
    {{"topic": "topic-name", "confidence": 0.80}}
  ],
  "suggested_new_topics": [
    {{"topic": "topic-name", "definition": "Brief definition"}}
  ]
}}
```

## Output (JSON only)
"""


BATCH_TOPIC_EXTRACTION_PROMPT: Final[
    str
] = """Extract topics from multiple documents efficiently.

## Task
Process the following documents and extract topics for each.

## Documents
{documents}

## Existing Taxonomy
{existing_taxonomy}

## Instructions
1. Extract 3-5 topics per document
2. Reuse taxonomy terms across documents when applicable
3. Note topic frequency across the batch
4. Identify emerging topics not in taxonomy

## Output Format (JSON)
```json
{{
  "document_topics": [
    {{
      "doc_id": "document_identifier",
      "topics": ["topic-1", "topic-2", "topic-3"],
      "confidence": 0.90
    }}
  ],
  "batch_statistics": {{
    "most_common_topics": [
      {{"topic": "topic-name", "count": 5}}
    ],
    "new_topics_suggested": [
      {{"topic": "topic-name", "frequency": 3, "definition": "Brief definition"}}
    ]
  }}
}}
```

## Output (JSON only)
"""


TOPIC_CLUSTERING_PROMPT: Final[
    str
] = """Group related topics into a hierarchical taxonomy.

## Task
Organize these topics into a logical hierarchy.

## Topics to Organize
{topics}

## Existing Hierarchy (if any)
{existing_hierarchy}

## Instructions
1. Group related topics under parent categories
2. Create max 3 levels of hierarchy
3. Each parent should have 2-7 children
4. Identify synonym groups (topics that mean the same thing)
5. Suggest merges for redundant topics

## Output Format (JSON)
```json
{{
  "taxonomy": {{
    "category-name": {{
      "description": "Category description",
      "children": {{
        "subcategory-name": {{
          "description": "Subcategory description",
          "topics": ["topic-1", "topic-2"]
        }}
      }}
    }}
  }},
  "synonyms": [
    {{
      "canonical": "preferred-term",
      "aliases": ["alias-1", "alias-2"]
    }}
  ],
  "merge_suggestions": [
    {{
      "topics": ["topic-a", "topic-b"],
      "suggested_name": "merged-topic",
      "reason": "Why these should be merged"
    }}
  ]
}}
```

## Output (JSON only)
"""


# Default topic taxonomy for general-purpose use
DEFAULT_TOPIC_TAXONOMY: Final[list[str]] = [
    # Technology
    "artificial-intelligence",
    "machine-learning",
    "deep-learning",
    "natural-language-processing",
    "computer-vision",
    "data-science",
    "software-engineering",
    "devops",
    "cloud-computing",
    "cybersecurity",
    # Business
    "strategy",
    "management",
    "finance",
    "marketing",
    "operations",
    "human-resources",
    # Science
    "research",
    "methodology",
    "analysis",
    "experiment",
    "theory",
    # General
    "tutorial",
    "guide",
    "overview",
    "comparison",
    "case-study",
    "best-practices",
]

# Topic extraction settings
DEFAULT_MAX_TOPICS: Final[int] = 5
MIN_TOPIC_CONFIDENCE: Final[float] = 0.6
