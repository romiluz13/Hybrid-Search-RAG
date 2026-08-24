# Background

This section records the architectural rationale behind HybridRAG. It complements the chronological [Project lore](../lore.md) with the reasons current implementation boundaries exist.

## Pages

- [Design decisions](design-decisions.md) summarizes the accepted and superseded ADRs under `docs/adr/`.

The most important themes are a single MongoDB data plane, an async-first library API, explicit provider boundaries, separate filter translations, and latest-first search behavior that fails closed rather than silently degrading.

For the resulting component layout, see [Architecture](../overview/architecture.md). For concrete storage shapes, see [Data models](../reference/data-models.md).
