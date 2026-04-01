"""
Entity Extraction Prompt Templates.

Knowledge graph entity and relationship extraction prompts for:
- Full document entity extraction (batch processing)
- Query entity extraction (lightweight, real-time)
- Entity normalization and deduplication

These prompts are designed for LLM-based entity extraction to build
and maintain knowledge graphs for RAG systems.

Usage:
    from hybridrag.prompts import (
        ENTITY_EXTRACTION_PROMPT,
        QUERY_ENTITY_EXTRACTION_PROMPT,
        ENTITY_NORMALIZATION_PROMPT,
    )

    # Format for document extraction
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        text=document_text,
        entity_types="person, organization, location, concept",
        language="English"
    )
"""

from __future__ import annotations

from typing import Final

ENTITY_EXTRACTION_PROMPT: Final[
    str
] = """You are a Knowledge Graph Specialist. Extract entities and relationships from the text.

## Task
Extract all meaningful entities and their relationships from the provided text.

## Entity Types to Extract
{entity_types}

## Output Format
Return a JSON object with two arrays: "entities" and "relationships"

### Entity Format
```json
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "entity_type",
      "description": "Brief description based on context",
      "confidence": 0.95
    }}
  ]
}}
```

### Relationship Format
```json
{{
  "relationships": [
    {{
      "source": "Entity A",
      "target": "Entity B",
      "type": "relationship_type",
      "description": "Description of the relationship",
      "confidence": 0.90
    }}
  ]
}}
```

## Instructions
1. Extract entities that are clearly defined in the text
2. Use consistent naming (title case for names, preserve technical terms)
3. Only create relationships that are explicitly stated or strongly implied
4. Assign confidence scores based on clarity of mention:
   - 0.95-1.0: Explicitly named and described
   - 0.80-0.95: Clearly mentioned but limited context
   - 0.60-0.80: Implied or partially mentioned
5. Do not invent entities or relationships not supported by the text
6. Write descriptions in {language}
7. Preserve proper nouns in their original form

## Text to Process
```
{text}
```

## Output (JSON only, no additional text)
"""


QUERY_ENTITY_EXTRACTION_PROMPT: Final[
    str
] = """Extract key entities from this search query for knowledge graph lookup.

## Task
Identify entities in the query that could be used to search a knowledge graph.

## Entity Types
{entity_types}

## Output Format
Return a JSON array of entities:
```json
[
  {{
    "name": "Entity Name",
    "type": "entity_type",
    "normalized_name": "normalized_entity_name",
    "importance": "high|medium|low"
  }}
]
```

## Instructions
1. Extract only entities relevant for search
2. Normalize names (lowercase, remove special characters)
3. Mark importance based on query focus:
   - high: Main subject of the query
   - medium: Supporting context
   - low: Peripheral mention
4. Keep extraction lightweight - focus on searchable terms
5. Include variations (e.g., "ML" and "Machine Learning")

## Query
{query}

## Output (JSON array only)
"""


ENTITY_NORMALIZATION_PROMPT: Final[str] = """Normalize and deduplicate entity mentions.

## Task
Given a list of entity mentions, group duplicates and provide canonical names.

## Input Entities
{entities}

## Output Format
Return a JSON object mapping variations to canonical forms:
```json
{{
  "canonical_entities": [
    {{
      "canonical_name": "Canonical Entity Name",
      "type": "entity_type",
      "variations": ["variation1", "variation2", "Variation3"],
      "merged_description": "Combined description from all variations"
    }}
  ]
}}
```

## Normalization Rules
1. Group entities that refer to the same real-world entity
2. Choose the most formal/complete name as canonical
3. Consider:
   - Acronyms (e.g., "ML" = "Machine Learning")
   - Name variations (e.g., "Bob Smith" = "Robert Smith")
   - Typos and misspellings
   - Different capitalizations
4. Merge descriptions from all variations
5. Preserve the most specific entity type

## Output (JSON only)
"""


# Relationship types commonly used in knowledge graphs
COMMON_RELATIONSHIP_TYPES: Final[list[str]] = [
    "works_for",
    "works_with",
    "reports_to",
    "manages",
    "created",
    "authored",
    "founded",
    "member_of",
    "located_in",
    "part_of",
    "related_to",
    "uses",
    "depends_on",
    "implements",
    "extends",
    "similar_to",
    "opposite_of",
    "causes",
    "prevents",
    "supports",
]

# Default entity types for general-purpose extraction
DEFAULT_ENTITY_TYPES: Final[list[str]] = [
    "person",
    "organization",
    "location",
    "event",
    "concept",
    "product",
    "technology",
    "date",
    "quantity",
]
