# ADR-0003: Hybrid Search with $rankFusion

## Status
Superseded by ADR-0008

## Context
Need to combine vector and keyword search results effectively.

## Decision
Use MongoDB `$rankFusion` with explicit weights (0.6/0.4 default).

## Consequences
**Positive**: Native MongoDB support, tunable weights
**Negative**: Requires M10+ cluster

## Date
2026-01-20
