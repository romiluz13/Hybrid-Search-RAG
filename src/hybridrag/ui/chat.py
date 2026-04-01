"""
Chainlit chat handlers for HybridRAG.

Features:
- PDF/TXT/MD file upload and ingestion
- Chat with documents using hybrid search
- Slash commands for mode switching
- Knowledge Base management panel
- Progress tracking with TaskList
- Streaming responses
"""

import asyncio
import os
import sys

import chainlit as cl

# Add project paths
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_project_root, "src"))

from hybridrag import Settings, create_hybridrag

# ============================================================================
# Knowledge Base Display Functions
# ============================================================================


def _spark_bar(value: float, max_value: float, width: int = 16) -> str:
    """Create a compact spark bar for quick visual comparison."""
    if max_value <= 0:
        return "░" * width
    filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division with zero guard."""
    return numerator / denominator if denominator else 0.0


async def _get_entity_samples(rag, limit: int = 8) -> list[str]:
    """Fetch a small sample of entity names for a graph preview."""
    candidates = []
    try:
        cursor = rag.entities_vdb._data.find(
            {},
            {"_id": 0, "entity": 1, "name": 1, "label": 1, "title": 1},
        ).limit(limit * 3)
        async for doc in cursor:
            for key in ("entity", "name", "label", "title"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
                    break
            if len(candidates) >= limit:
                break
    except Exception:
        return []
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for value in candidates:
        if value not in seen:
            unique.append(value)
            seen.add(value)
        if len(unique) >= limit:
            break
    return unique


async def format_dashboard_panel(rag, session_id: str, mode: str) -> str:
    """Build the main HybridRAG Studio dashboard view."""
    stats = await rag.get_knowledge_base_stats()
    status = await rag.get_status()

    docs_total = stats["documents"]["total"]
    chunks = stats["chunks"]
    entities = stats["entities"]
    relationships = stats["relationships"]
    by_status = stats["documents"]["by_status"]

    processed = by_status.get("processed", 0)
    pending = by_status.get("pending", 0)
    failed = by_status.get("failed", 0)
    processing = by_status.get("processing", 0)

    workspace = os.getenv("MONGODB_WORKSPACE") or os.getenv("WORKSPACE") or "default"
    session_short = f"{session_id[:8]}..." if session_id else "none"

    chunks_per_doc = _safe_div(chunks, docs_total)
    entities_per_doc = _safe_div(entities, docs_total)
    rels_per_entity = _safe_div(relationships, entities)

    entity_samples = await _get_entity_samples(rag)

    max_metric = max(docs_total, chunks, entities, relationships, 1)

    lines = []
    lines.append("# ✨ HybridRAG Studio")
    lines.append(
        "**The developer‑first command center for demos, debug, and discovery.**"
    )
    lines.append("")
    lines.append("### 🧭 Workspace Snapshot")
    lines.append("| Field | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Workspace | `{workspace}` |")
    lines.append(f"| Session | `{session_short}` |")
    lines.append(f"| Mode | **{mode}** |")
    lines.append(
        f"| LLM | `{status.get('llm_provider', 'unknown')}` · `{status.get('llm_model', 'unknown')}` |"
    )
    lines.append(f"| Embeddings | `{status.get('embedding_model', 'unknown')}` |")
    lines.append(
        f"| Reranker | `{status.get('rerank_model', 'disabled') or 'disabled'}` |"
    )
    lines.append("")

    lines.append("### 📊 Knowledge Base Health")
    lines.append("| Metric | Count | Signal |")
    lines.append("|--------|-------|--------|")
    lines.append(
        f"| Documents | **{docs_total:,}** | `{_spark_bar(docs_total, max_metric)}` |"
    )
    lines.append(f"| Chunks | **{chunks:,}** | `{_spark_bar(chunks, max_metric)}` |")
    lines.append(
        f"| Entities | **{entities:,}** | `{_spark_bar(entities, max_metric)}` |"
    )
    lines.append(
        f"| Relationships | **{relationships:,}** | `{_spark_bar(relationships, max_metric)}` |"
    )
    lines.append("")

    lines.append("### 🧪 Ingestion Health")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| ✅ Processed | **{processed}** |")
    lines.append(f"| 🔄 Processing | **{processing}** |")
    lines.append(f"| ⏳ Pending | **{pending}** |")
    lines.append(f"| ❌ Failed | **{failed}** |")
    lines.append("")

    lines.append("### 🧠 Graph Snapshot")
    lines.append("| Signal | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Entities / Doc | **{entities_per_doc:.2f}** |")
    lines.append(f"| Chunks / Doc | **{chunks_per_doc:.2f}** |")
    lines.append(f"| Relationships / Entity | **{rels_per_entity:.2f}** |")
    lines.append("")
    if entity_samples:
        lines.append("**Entity Constellation (sample):**")
        lines.append(" · ".join(f"`{e[:24]}`" for e in entity_samples))
        lines.append("")
    else:
        lines.append("*No entities yet. Upload docs to light up the graph.*")
        lines.append("")

    # Recent documents
    recent_docs = stats["recent_documents"]
    lines.append("### 📁 Recent Documents")
    if recent_docs:
        lines.append("| File | Status | Chunks |")
        lines.append("|------|--------|--------|")
        for doc in recent_docs[:8]:
            file_name = doc["file"]
            if len(file_name) > 32:
                file_name = file_name[:29] + "..."
            status_icon = {
                "processed": "✅",
                "pending": "⏳",
                "processing": "🔄",
                "failed": "❌",
            }.get(doc["status"], "📄")
            lines.append(
                f"| `{file_name}` | {status_icon} {doc['status']} | {doc['chunks']} |"
            )
    else:
        lines.append("*No documents indexed yet.*")
    lines.append("")

    lines.append("### 💡 Demo Prompts")
    lines.append("- *“Show me the highest SLA credit tier and how to claim it.”*")
    lines.append("- *“Summarize the release freeze rules and exceptions.”*")
    lines.append(
        "- *“What’s our current ingestion health and how many entities exist?”*"
    )
    lines.append("")

    return "\n".join(lines)


async def format_knowledge_base_stats(rag) -> str:
    """Format knowledge base statistics as a beautiful markdown display."""
    try:
        stats = await rag.get_knowledge_base_stats()

        docs_total = stats["documents"]["total"]
        entities = stats["entities"]
        relationships = stats["relationships"]
        chunks = stats["chunks"]
        by_status = stats["documents"]["by_status"]
        recent_docs = stats["recent_documents"]

        max_metric = max(docs_total, chunks, entities, relationships, 1)

        # Build the display
        lines = []
        lines.append("## 📚 Knowledge Base Snapshot")
        lines.append("")

        # Stats grid
        lines.append("| Metric | Count | Signal |")
        lines.append("|--------|-------|--------|")
        lines.append(
            f"| Documents | **{docs_total:,}** | `{_spark_bar(docs_total, max_metric)}` |"
        )
        lines.append(
            f"| Chunks | **{chunks:,}** | `{_spark_bar(chunks, max_metric)}` |"
        )
        lines.append(
            f"| Entities | **{entities:,}** | `{_spark_bar(entities, max_metric)}` |"
        )
        lines.append(
            f"| Relationships | **{relationships:,}** | `{_spark_bar(relationships, max_metric)}` |"
        )
        lines.append("")

        # Status breakdown if there are documents
        if by_status:
            lines.append("### Document Status")
            for status, count in sorted(by_status.items()):
                icon = {"processed": "✅", "pending": "⏳", "failed": "❌"}.get(
                    status, "📄"
                )
                lines.append(f"- {icon} **{status}**: {count}")
            lines.append("")

        # Recent documents
        if recent_docs:
            lines.append("### Recent Documents")
            lines.append("| File | Status | Chunks |")
            lines.append("|------|--------|--------|")
            for doc in recent_docs[:5]:
                file_name = doc["file"]
                if len(file_name) > 30:
                    file_name = file_name[:27] + "..."
                status_icon = {"processed": "✅", "pending": "⏳", "failed": "❌"}.get(
                    doc["status"], "📄"
                )
                lines.append(
                    f"| `{file_name}` | {status_icon} {doc['status']} | {doc['chunks']} |"
                )
            lines.append("")

        if docs_total == 0:
            lines.append(
                "> **No documents yet.** Upload PDF, TXT, or MD files to get started!"
            )
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"*Error loading knowledge base stats: {e}*"


async def format_kb_management_panel(rag) -> tuple[str, list]:
    """
    Format KB management panel with delete actions.
    Returns (markdown_content, list_of_actions).
    """
    try:
        stats = await rag.get_knowledge_base_stats()
        recent_docs = stats["recent_documents"]
        docs_total = stats["documents"]["total"]
        entities = stats["entities"]
        relationships = stats["relationships"]
        chunks = stats["chunks"]

        lines = []
        lines.append("## 🧰 Knowledge Base Manager")
        lines.append("")

        # Visual stats cards (markdown style)
        lines.append("### 📊 Statistics")
        lines.append("```")
        lines.append("┌─────────────────────────────────────────────┐")
        lines.append(f"│  📄 Documents: {docs_total:<7} │  📦 Chunks: {chunks:<9} │")
        lines.append(
            f"│  🔷 Entities:  {entities:<7} │  🔗 Relations: {relationships:<7} │"
        )
        lines.append("└─────────────────────────────────────────────┘")
        lines.append("```")
        lines.append("")

        # Documents list with delete options
        actions = []

        if recent_docs:
            lines.append("### 📁 Documents")
            lines.append(
                "*Delete is only enabled for processed documents to prevent partial removal.*"
            )
            lines.append("")

            for i, doc in enumerate(recent_docs[:10]):  # Show up to 10 docs
                doc_id = doc.get("id", "unknown")
                file_name = doc["file"]
                status = doc["status"]
                chunk_count = doc.get("chunks", 0)

                # Status icon
                status_icon = {
                    "processed": "✅",
                    "pending": "⏳",
                    "processing": "🔄",
                    "failed": "❌",
                }.get(status, "📄")

                # Display name (truncate if needed)
                display_name = file_name
                if len(display_name) > 35:
                    display_name = display_name[:32] + "..."

                lines.append(f"**{i + 1}.** {status_icon} `{display_name}`")
                lines.append(f"   └── Chunks: **{chunk_count}** | Status: **{status}**")
                lines.append("")

                # Create delete action for each document
                # Only allow deletion of processed documents
                if status == "processed":
                    action = cl.Action(
                        name="delete_doc",
                        payload={"doc_id": doc_id, "file_name": file_name},
                        label=f"🗑️ Delete {file_name[:20]}...",
                        tooltip=f"Delete document: {file_name}",
                    )
                    actions.append(action)
        else:
            lines.append("### 📁 No Documents")
            lines.append("")
            lines.append("> 📤 **Upload files** to start building your knowledge base!")
            lines.append(">")
            lines.append("> Supported formats: PDF, TXT, MD")
            lines.append("")

        lines.append("---")
        lines.append(
            "**Commands:** `/dashboard` | `/kb` | `/manage` | `/upload` | `/status` | `/help`"
        )
        lines.append("")

        return "\n".join(lines), actions

    except Exception as e:
        return f"*Error loading KB management panel: {e}*", []


# ============================================================================
# Action Builders
# ============================================================================


def build_dashboard_actions(current_mode: str) -> list:
    """Create the primary action dock for the dashboard."""
    mode_labels = {
        "mix": "mix",
        "hybrid": "hybrid",
        "local": "local",
        "global": "global",
        "naive": "naive",
        "bypass": "bypass",
    }
    mode_actions = [
        cl.Action(
            name="set_mode",
            payload={"mode": mode},
            label=f"⚙️ Mode: {label}",
            tooltip=f"Switch query mode to {mode}",
        )
        for mode, label in mode_labels.items()
    ]

    actions = [
        cl.Action(
            name="show_dashboard",
            payload={},
            label="🏠 Dashboard",
            tooltip="Show main dashboard",
        ),
        cl.Action(
            name="show_kb_manager",
            payload={},
            label="📚 KB Manager",
            tooltip="Manage documents",
        ),
        cl.Action(
            name="upload_files",
            payload={},
            label="📤 Upload",
            tooltip="Upload documents",
        ),
        cl.Action(
            name="ingest_url",
            payload={},
            label="🌐 Ingest URL",
            tooltip="Ingest a web page",
        ),
        cl.Action(
            name="crawl_website", payload={}, label="🕷️ Crawl", tooltip="Crawl a website"
        ),
        cl.Action(
            name="show_system_status",
            payload={},
            label="🧪 System",
            tooltip="Show system status",
        ),
        cl.Action(
            name="show_memory",
            payload={},
            label="🧠 Memory",
            tooltip="View conversation memory",
        ),
        cl.Action(
            name="show_help",
            payload={},
            label="❓ Help",
            tooltip="Show help & commands",
        ),
    ]

    # Keep the primary dock compact, and expose modes after it
    return actions + mode_actions


# ============================================================================
# File Processing Functions
# ============================================================================


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text_parts = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")

        doc.close()
        return "\n\n".join(text_parts)
    except ImportError as e:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf") from e


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract text from various file types."""
    if mime_type == "application/pdf" or file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif mime_type in ["text/plain", "text/markdown"] or file_path.endswith(
        (".txt", ".md")
    ):
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


# ============================================================================
# Chainlit Event Handlers
# ============================================================================


@cl.on_chat_start
async def on_chat_start():
    """Initialize HybridRAG when chat session starts."""

    await cl.Message(content="⚡ Initializing HybridRAG... Please wait.").send()

    try:
        # Get settings from environment
        settings = Settings()

        # Create and initialize HybridRAG
        rag = await create_hybridrag(settings=settings, auto_initialize=True)

        # Create a conversation session for this chat (persisted in MongoDB)
        import uuid

        session_id = str(uuid.uuid4())
        await rag.create_conversation_session(
            session_id, metadata={"source": "chainlit"}
        )

        # Store in session
        cl.user_session.set("rag", rag)
        cl.user_session.set("session_id", session_id)  # For conversation memory
        cl.user_session.set("mode", "mix")  # Default mode

        # Main dashboard
        dashboard = await format_dashboard_panel(
            rag, session_id, cl.user_session.get("mode", "mix")
        )
        actions = build_dashboard_actions(cl.user_session.get("mode", "mix"))
        await cl.Message(content=dashboard, actions=actions).send()

    except Exception as e:
        await cl.Message(
            content=f"**❌ Error initializing HybridRAG:** {str(e)}\n\nPlease check your environment variables (MONGODB_URI, VOYAGE_API_KEY, etc.)"
        ).send()


# ============================================================================
# Action Callbacks
# ============================================================================


@cl.action_callback("delete_doc")
async def on_delete_doc(action: cl.Action):
    """Handle document deletion from KB management panel."""
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    doc_id = action.payload.get("doc_id")
    file_name = action.payload.get("file_name", "unknown")

    # Confirm deletion
    confirm_msg = await cl.Message(
        content=f"🗑️ Deleting document: `{file_name}`..."
    ).send()

    try:
        # Delete the document
        await rag.delete_document(doc_id)

        confirm_msg.content = f"✅ Successfully deleted: `{file_name}`"
        await confirm_msg.update()

        # Show updated KB stats
        await asyncio.sleep(0.5)  # Brief delay for UI
        kb_stats = await format_knowledge_base_stats(rag)
        await cl.Message(content=kb_stats).send()

    except Exception as e:
        confirm_msg.content = f"❌ Error deleting `{file_name}`: {str(e)}"
        await confirm_msg.update()

    # Remove the action button
    await action.remove()


@cl.action_callback("show_dashboard")
async def on_show_dashboard(action: cl.Action):
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    session_id = cl.user_session.get("session_id", "")
    mode = cl.user_session.get("mode", "mix")
    dashboard = await format_dashboard_panel(rag, session_id, mode)
    actions = build_dashboard_actions(mode)
    await cl.Message(content=dashboard, actions=actions).send()


@cl.action_callback("show_kb_manager")
async def on_show_kb_manager(action: cl.Action):
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    content, actions = await format_kb_management_panel(rag)
    upload_action = cl.Action(
        name="upload_files",
        payload={},
        label="📤 Upload Files",
        tooltip="Upload new documents to the knowledge base",
    )
    ingest_url_action = cl.Action(
        name="ingest_url",
        payload={},
        label="🌐 Ingest URL",
        tooltip="Extract content from a single web URL",
    )
    crawl_website_action = cl.Action(
        name="crawl_website",
        payload={},
        label="🕷️ Crawl Website",
        tooltip="Crawl and ingest multiple pages from a website",
    )
    actions.insert(0, crawl_website_action)
    actions.insert(0, ingest_url_action)
    actions.insert(0, upload_action)
    await cl.Message(content=content, actions=actions).send()


@cl.action_callback("show_system_status")
async def on_show_system_status(action: cl.Action):
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    try:
        status = await rag.get_status()
        workspace = (
            os.getenv("MONGODB_WORKSPACE") or os.getenv("WORKSPACE") or "default"
        )
        status_text = f"""## 🧪 System Status

| Component | Value |
|-----------|-------|
| Workspace | `{workspace}` |
| LLM Provider | **{status.get("llm_provider", "unknown")}** |
| LLM Model | `{status.get("llm_model", "unknown")}` |
| Embedding | `{status.get("embedding_model", "unknown")}` |
| Reranker | `{status.get("rerank_model", "disabled") or "disabled"}` |
| Query Mode | **{cl.user_session.get("mode", "mix")}** |
| Database | `{status.get("mongodb_database", "unknown")}` |

### 🔧 Enhancements
- Implicit Expansion: {"✅ Enabled" if status.get("enhancements", {}).get("implicit_expansion") else "❌ Disabled"}
- Entity Boosting: {"✅ Enabled" if status.get("enhancements", {}).get("entity_boosting") else "❌ Disabled"}
"""
        await cl.Message(content=status_text).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error getting status: {e}").send()


@cl.action_callback("show_help")
async def on_show_help(action: cl.Action):
    await handle_command("/help", cl.user_session.get("rag"))


@cl.action_callback("show_memory")
async def on_show_memory(action: cl.Action):
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return
    await handle_command("/memory", rag)


@cl.action_callback("set_mode")
async def on_set_mode(action: cl.Action):
    mode = action.payload.get("mode", "").lower()
    valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
    if mode not in valid_modes:
        await cl.Message(content=f"❌ Invalid mode: `{mode}`").send()
        return

    cl.user_session.set("mode", mode)
    await cl.Message(content=f"✅ Query mode changed to **{mode}**").send()


@cl.action_callback("upload_files")
async def on_upload_files(action: cl.Action):
    """Handle upload files action."""
    files = await cl.AskFileMessage(
        content="📤 **Upload documents to add to Knowledge Base**\n\nSupported formats: PDF, TXT, MD",
        accept=[
            "application/pdf",
            "text/plain",
            "text/markdown",
            ".pdf",
            ".txt",
            ".md",
        ],
        max_size_mb=50,
        max_files=10,
    ).send()

    if files:
        rag = cl.user_session.get("rag")
        if rag:
            await handle_file_upload_with_progress(files, rag)


@cl.action_callback("ingest_url")
async def on_ingest_url(action: cl.Action):
    """Handle URL ingestion action."""
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    # Ask for URL
    url_msg = await cl.AskUserMessage(
        content="🌐 **Enter URL to extract content from**\n\nExample: https://docs.mongodb.com/atlas/",
    ).send()

    if url_msg and url_msg.get("content"):
        url = url_msg["content"].strip()

        # Validate URL
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if not all([parsed.scheme in ["http", "https"], parsed.netloc]):
                await cl.Message(content=f"❌ Invalid URL format: {url}").send()
                return
        except Exception:
            await cl.Message(content=f"❌ Invalid URL format: {url}").send()
            return

        # Show progress
        progress_msg = cl.Message(content="⏳ **Extracting content from URL...**")
        await progress_msg.send()

        try:
            result = await rag.ingest_url(url)

            if result.success:
                progress_msg.content = f"""✅ **URL Ingestion Complete**

**Title:** {result.title}
**Chunks Created:** {result.chunks_created}
**Source:** {result.source}
**Processing Time:** {result.processing_time_ms / 1000:.2f}s

You can now query this content!"""
            else:
                progress_msg.content = f"""❌ **URL Ingestion Failed**

**URL:** {url}
**Errors:** {", ".join(result.errors)}"""

            await progress_msg.update()
        except Exception as e:
            progress_msg.content = f"❌ **Error ingesting URL:** {str(e)}"
            await progress_msg.update()


@cl.action_callback("crawl_website")
async def on_crawl_website(action: cl.Action):
    """Handle website crawl action."""
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ RAG system not initialized.").send()
        return

    # Ask for URL and max pages
    url_msg = await cl.AskUserMessage(
        content="🕷️ **Enter website URL to crawl**\n\nExample: https://docs.mongodb.com/atlas/\n\nHow many pages? (default: 10)",
    ).send()

    if url_msg and url_msg.get("content"):
        parts = url_msg["content"].strip().split()
        if not parts:
            await cl.Message(content="❌ Please provide a URL.").send()
            return

        url = parts[0]
        max_pages = 10
        if len(parts) > 1:
            try:
                max_pages = int(parts[1])
            except ValueError:
                await cl.Message(content=f"❌ Invalid max_pages: {parts[1]}").send()
                return

        # Validate URL
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if not all([parsed.scheme in ["http", "https"], parsed.netloc]):
                await cl.Message(content=f"❌ Invalid URL format: {url}").send()
                return
        except Exception:
            await cl.Message(content=f"❌ Invalid URL format: {url}").send()
            return

        # Show progress
        progress_msg = cl.Message(
            content=f"⏳ **Crawling website...**\n\nURL: {url}\nMax pages: {max_pages}"
        )
        await progress_msg.send()

        try:

            async def progress_callback(current: int, total: int) -> None:
                progress_msg.content = f"""⏳ **Crawling website...**

**URL:** {url}
**Max pages:** {max_pages}
**Progress:** {current}/{total} pages processed"""
                await progress_msg.update()

            results = await rag.ingest_website(
                url, max_pages=max_pages, progress_callback=progress_callback
            )

            successful = sum(1 for r in results if r.success)
            total_chunks = sum(r.chunks_created for r in results)

            if successful > 0:
                progress_msg.content = f"""✅ **Website Crawl Complete**

**URL:** {url}
**Pages Processed:** {successful}/{len(results)}
**Total Chunks Created:** {total_chunks}

You can now query this content!"""
            else:
                progress_msg.content = f"""❌ **Website Crawl Failed**

**URL:** {url}
**Pages:** 0/{len(results)} successful
**Errors:** {", ".join(results[0].errors) if results else "No pages extracted"}"""

            await progress_msg.update()
        except Exception as e:
            progress_msg.content = f"❌ **Error crawling website:** {str(e)}"
            await progress_msg.update()


# ============================================================================
# Greeting and Meta-Question Detection
# ============================================================================

# Common greetings that shouldn't trigger RAG retrieval
GREETINGS = {
    "hello",
    "hi",
    "hey",
    "hola",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "yo",
    "sup",
    "what's up",
    "whats up",
    "hi there",
    "hello there",
    "hey there",
    "hiya",
    "heya",
}

# Meta-questions about the system/knowledge base
META_PATTERNS = [
    "what do you know",
    "what's in your",
    "whats in your",
    "what is in your",
    "do you have anything",
    "have anything in",
    "knowledge base",
    "what can you",
    "what documents",
    "what files",
    "what data",
    "tell me about yourself",
    "who are you",
    "what are you",
]


def is_greeting(text: str) -> bool:
    """Check if the message is a simple greeting."""
    normalized = text.lower().strip().rstrip("!?.,:;")
    return normalized in GREETINGS or len(normalized) < 3


def is_meta_question(text: str) -> bool:
    """Check if the message is a meta-question about the system."""
    normalized = text.lower()
    return any(pattern in normalized for pattern in META_PATTERNS)


# ============================================================================
# Message Handler
# ============================================================================


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""

    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(
            content="❌ RAG system not initialized. Please refresh the page."
        ).send()
        return

    # Check for file uploads
    if message.elements:
        await handle_file_upload_with_progress(message.elements, rag)
        return

    # Check for slash commands
    text = message.content.strip()
    if text.startswith("/"):
        await handle_command(text, rag)
        return

    # Handle greetings without RAG
    if is_greeting(text):
        await cl.Message(
            content="👋 Hello! I'm your **HybridRAG** assistant.\n\n"
            "I can help you with questions about your uploaded documents.\n\n"
            "**Try asking me:**\n"
            "- Questions about content in your documents\n"
            "- Summaries or explanations of topics\n"
            "- Specific details or facts\n\n"
            "📤 Upload a PDF, TXT, or MD file to get started!\n\n"
            "Tip: Open the **Dashboard** with `/dashboard` to manage everything."
        ).send()
        return

    # Handle meta-questions about the system
    if is_meta_question(text):
        await cl.Message(
            content="🤖 I'm **HybridRAG** - a hybrid Retrieval-Augmented Generation system:\n\n"
            "**Powered by:**\n"
            "- 🍃 **MongoDB Atlas** - vector + graph storage\n"
            "- 🚀 **Voyage AI** - embeddings & reranking\n"
            "- 🧠 **Multi-provider LLM** - response generation\n"
            "- 🔗 **Knowledge Graph** - entity extraction & querying\n\n"
            "Type `/dashboard` to open the developer cockpit.\n\n"
            "**Try asking:**\n"
            "- Questions about your uploaded documents\n"
            "- Summaries of specific topics\n"
            "- Relationship queries between concepts"
        ).send()
        return

    # Handle regular query
    await handle_query(text, rag)


# ============================================================================
# File Upload with Progress Tracking
# ============================================================================


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def create_progress_bar(percent: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {percent:.1f}%"


async def handle_file_upload_with_progress(elements: list, rag):
    """Handle file uploads with detailed progress tracking."""
    import time

    # Filter valid files
    valid_elements = [e for e in elements if hasattr(e, "path") and e.path]
    if not valid_elements:
        await cl.Message(content="❌ No valid files to process").send()
        return

    total_files = len(valid_elements)
    files_processed = []
    errors = []

    # Create progress message
    progress_msg = cl.Message(content="⏳ **Preparing ingestion...**")
    await progress_msg.send()

    # Overall timing
    overall_start = time.time()

    for file_idx, element in enumerate(valid_elements):
        file_name = getattr(element, "name", "unknown")
        mime_type = getattr(element, "mime", "")
        file_start = time.time()

        try:
            # === STEP 1: Extract text (10%) ===
            step_percent = (file_idx / total_files) * 100
            progress_msg.content = f"""## 📥 Ingesting Files

{create_progress_bar(step_percent + 2)}

**File {file_idx + 1}/{total_files}:** `{file_name}`
**Step:** 📄 Extracting text...

| Metric | Value |
|--------|-------|
| Files Done | {file_idx}/{total_files} |
| Elapsed | {format_time(time.time() - overall_start)} |
"""
            await progress_msg.update()

            text = extract_text_from_file(element.path, mime_type)

            if not text.strip():
                errors.append(f"`{file_name}`: No text content found")
                continue

            char_count = len(text)
            # Estimate chunks (rough: 1 chunk per 1000 chars)
            est_chunks = max(1, char_count // 1000)

            # === STEP 2: Chunking (30%) ===
            step_percent = ((file_idx + 0.3) / total_files) * 100
            elapsed = time.time() - overall_start
            # Estimate: ~500 chars/sec for full pipeline
            est_total = (char_count / 500) + elapsed
            eta = max(0, est_total - elapsed)

            progress_msg.content = f"""## 📥 Ingesting Files

{create_progress_bar(step_percent)}

**File {file_idx + 1}/{total_files}:** `{file_name}`
**Step:** ✂️ Chunking document...

| Metric | Value |
|--------|-------|
| Characters | {char_count:,} |
| Est. Chunks | ~{est_chunks} |
| Elapsed | {format_time(elapsed)} |
| ETA | ~{format_time(eta)} |
"""
            await progress_msg.update()

            # === STEP 3: Embedding + KG Extraction (60%) ===
            step_percent = ((file_idx + 0.5) / total_files) * 100
            elapsed = time.time() - overall_start

            progress_msg.content = f"""## 📥 Ingesting Files

{create_progress_bar(step_percent)}

**File {file_idx + 1}/{total_files}:** `{file_name}`
**Step:** 🧠 Generating embeddings & extracting entities...

| Metric | Value |
|--------|-------|
| Characters | {char_count:,} |
| Est. Chunks | ~{est_chunks} |
| Elapsed | {format_time(elapsed)} |
| ETA | ~{format_time(eta)} |

> *This step may take a while for large documents...*
"""
            await progress_msg.update()

            # Actually ingest (this is the slow part)
            await rag.insert(text, file_paths=[file_name])

            # === STEP 4: Complete (100%) ===
            file_duration = time.time() - file_start
            step_percent = ((file_idx + 1) / total_files) * 100
            elapsed = time.time() - overall_start

            # Calculate better ETA based on actual processing rate
            if file_idx < total_files - 1:
                avg_time_per_file = elapsed / (file_idx + 1)
                eta = avg_time_per_file * (total_files - file_idx - 1)
            else:
                eta = 0

            progress_msg.content = f"""## 📥 Ingesting Files

{create_progress_bar(step_percent)}

**File {file_idx + 1}/{total_files}:** `{file_name}` ✅
**Step:** ✅ Complete!

| Metric | Value |
|--------|-------|
| Characters | {char_count:,} |
| File Time | {format_time(file_duration)} |
| Total Elapsed | {format_time(elapsed)} |
| ETA | ~{format_time(eta)} |
"""
            await progress_msg.update()

            files_processed.append(
                {"name": file_name, "chars": char_count, "duration": file_duration}
            )

        except Exception as e:
            errors.append(f"`{file_name}`: {str(e)}")
            progress_msg.content = f"""## 📥 Ingesting Files

{create_progress_bar(((file_idx + 1) / total_files) * 100)}

**File {file_idx + 1}/{total_files}:** `{file_name}` ❌
**Error:** {str(e)[:100]}
"""
            await progress_msg.update()

    # === FINAL SUMMARY ===
    total_duration = time.time() - overall_start
    total_chars = sum(f["chars"] for f in files_processed)

    if files_processed:
        summary_lines = [
            "## ✅ Ingestion Complete!",
            "",
            f"{create_progress_bar(100)}",
            "",
            "### 📊 Summary",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Files Processed | **{len(files_processed)}/{total_files}** |",
            f"| Total Characters | **{total_chars:,}** |",
            f"| Total Time | **{format_time(total_duration)}** |",
            f"| Avg Speed | **{total_chars / total_duration:,.0f} chars/sec** |",
            "",
            "### 📁 Files",
        ]

        for f in files_processed:
            summary_lines.append(
                f"- ✅ `{f['name']}` ({f['chars']:,} chars, {format_time(f['duration'])})"
            )

        if errors:
            summary_lines.append("")
            summary_lines.append("### ⚠️ Errors")
            for e in errors:
                summary_lines.append(f"- ❌ {e}")

        summary_lines.append("")
        summary_lines.append("> 💡 You can now ask questions about these documents!")

        progress_msg.content = "\n".join(summary_lines)
        await progress_msg.update()
    else:
        progress_msg.content = (
            "## ❌ Ingestion Failed\n\nNo files were processed successfully.\n\n### Errors\n"
            + "\n".join(f"- {e}" for e in errors)
        )
        await progress_msg.update()


# Legacy handler (kept for compatibility)
async def handle_file_upload(elements: list, rag):
    """Handle file uploads - redirects to progress-tracked version."""
    await handle_file_upload_with_progress(elements, rag)


# ============================================================================
# Command Handler
# ============================================================================


async def handle_command(text: str, rag):
    """Handle slash commands."""

    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/mode":
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if args.lower() in valid_modes:
            cl.user_session.set("mode", args.lower())
            await cl.Message(
                content=f"✅ Query mode changed to **{args.lower()}**"
            ).send()
        else:
            await cl.Message(
                content=f"❌ Invalid mode. Valid modes: {', '.join(valid_modes)}"
            ).send()

    elif command == "/kb":
        # Quick knowledge base statistics
        kb_stats = await format_knowledge_base_stats(rag)
        await cl.Message(content=kb_stats).send()

    elif command == "/dashboard":
        session_id = cl.user_session.get("session_id", "")
        mode = cl.user_session.get("mode", "mix")
        dashboard = await format_dashboard_panel(rag, session_id, mode)
        actions = build_dashboard_actions(mode)
        await cl.Message(content=dashboard, actions=actions).send()

    elif command == "/manage":
        # Full KB Management Panel with actions
        content, actions = await format_kb_management_panel(rag)

        # Add upload and web ingestion actions
        upload_action = cl.Action(
            name="upload_files",
            payload={},
            label="📤 Upload Files",
            tooltip="Upload new documents to the knowledge base",
        )
        ingest_url_action = cl.Action(
            name="ingest_url",
            payload={},
            label="🌐 Ingest URL",
            tooltip="Extract content from a single web URL",
        )
        crawl_website_action = cl.Action(
            name="crawl_website",
            payload={},
            label="🕷️ Crawl Website",
            tooltip="Crawl and ingest multiple pages from a website",
        )
        actions.insert(0, crawl_website_action)
        actions.insert(0, ingest_url_action)
        actions.insert(0, upload_action)

        await cl.Message(content=content, actions=actions).send()

    elif command == "/upload":
        # Trigger file upload dialog
        files = await cl.AskFileMessage(
            content="📤 **Upload documents to Knowledge Base**\n\nSupported: PDF, TXT, MD (max 50MB each)",
            accept=[
                "application/pdf",
                "text/plain",
                "text/markdown",
                ".pdf",
                ".txt",
                ".md",
            ],
            max_size_mb=50,
            max_files=10,
        ).send()

        if files:
            await handle_file_upload_with_progress(files, rag)

    elif command == "/status":
        try:
            status = await rag.get_status()
            workspace = (
                os.getenv("MONGODB_WORKSPACE") or os.getenv("WORKSPACE") or "default"
            )
            status_text = f"""## ⚙️ System Status

| Component | Value |
|-----------|-------|
| Workspace | `{workspace}` |
| LLM Provider | **{status.get("llm_provider", "unknown")}** |
| LLM Model | `{status.get("llm_model", "unknown")}` |
| Embedding | `{status.get("embedding_model", "unknown")}` |
| Reranker | `{status.get("rerank_model", "disabled") or "disabled"}` |
| Query Mode | **{cl.user_session.get("mode", "mix")}** |
| Database | `{status.get("mongodb_database", "unknown")}` |

### 🔧 Enhancements
- Implicit Expansion: {"✅ Enabled" if status.get("enhancements", {}).get("implicit_expansion") else "❌ Disabled"}
- Entity Boosting: {"✅ Enabled" if status.get("enhancements", {}).get("entity_boosting") else "❌ Disabled"}
"""
            await cl.Message(content=status_text).send()
        except Exception as e:
            await cl.Message(content=f"❌ Error getting status: {e}").send()

    elif command == "/memory":
        # Show conversation memory for this session (including summary)
        session_id = cl.user_session.get("session_id")
        if not session_id:
            await cl.Message(content="❌ No session ID found.").send()
            return

        try:
            # Get full session to access summary
            session = await rag._memory.get_session(session_id)
            if not session:
                await cl.Message(
                    content="📝 **Conversation Memory**\n\nNo session found."
                ).send()
                return

            lines = ["## 📝 Conversation Memory", ""]
            lines.append(f"**Session ID:** `{session_id[:12]}...`")
            lines.append(f"**Messages:** {len(session.messages)}")

            # Show token estimate
            total_tokens = len(session.summary) // 4
            for msg in session.messages:
                total_tokens += len(msg.get("content", "")) // 4
            lines.append(f"**Est. Tokens:** ~{total_tokens:,} / 32,000")
            lines.append("")

            # Show summary if exists (from compaction)
            if session.summary:
                lines.append("### 📋 Summary (compacted history)")
                lines.append(f"> {session.summary}")
                lines.append("")

            # Show recent messages
            if session.messages:
                lines.append("### 💬 Recent Messages")
                for i, msg in enumerate(session.messages[-10:], 1):  # Show last 10
                    role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                    content = (
                        msg["content"][:150] + "..."
                        if len(msg["content"]) > 150
                        else msg["content"]
                    )
                    lines.append(f"**{i}. {role}:** {content}")
                    lines.append("")
            else:
                lines.append("*No recent messages*")

            await cl.Message(content="\n".join(lines)).send()
        except Exception as e:
            await cl.Message(content=f"❌ Error getting memory: {e}").send()

    elif command == "/clear":
        # Clear conversation memory
        session_id = cl.user_session.get("session_id")
        if session_id:
            await rag.clear_conversation(session_id)
            await cl.Message(
                content="🗑️ Conversation memory cleared for this session."
            ).send()
        else:
            await cl.Message(content="❌ No session to clear.").send()

    elif command == "/help":
        help_text = """## 📖 HybridRAG Help

### Commands
| Command | Description |
|---------|-------------|
| `/dashboard` | **Studio dashboard (recommended)** |
| `/kb` | Quick knowledge base stats |
| `/manage` | **📚 Full KB Management Panel** |
| `/upload` | Upload files dialog |
| `/mode <mode>` | Change query mode |
| `/memory` | **View conversation history** |
| `/clear` | Clear conversation memory |
| `/status` | System configuration |
| `/help` | This help message |

### Query Modes
| Mode | Description |
|------|-------------|
| `mix` | ⭐ All modes combined (recommended) |
| `local` | Entity-focused (graph neighbors) |
| `global` | Community summaries |
| `hybrid` | Local + Global combined |
| `naive` | Direct vector search |
| `bypass` | Direct LLM (no retrieval) |

### 🧠 Session Memory
- **Multi-turn conversations** are now supported!
- Each chat session maintains conversation history in MongoDB
- Follow-up questions like "explain that" or "tell me more" work naturally
- Use `/memory` to see your conversation history
- Use `/clear` to reset the conversation

### File Upload
- **Drag & drop** PDF, TXT, or MD files into the chat
- Or use `/upload` command
- Files are processed and added to the knowledge graph

### Tips
- 💡 Use `/dashboard` as your demo cockpit and developer overview
- 💡 Use `/manage` to see all documents and delete unwanted ones
- 💡 Ask follow-up questions for deeper exploration
- 💡 Try different query modes for different types of questions
"""
        await cl.Message(content=help_text).send()

    else:
        await cl.Message(
            content=f"❓ Unknown command: `{command}`\n\nType `/help` for available commands."
        ).send()


# ============================================================================
# Query Handler
# ============================================================================


async def handle_query(query: str, rag):
    """Handle a user query with conversation memory."""

    mode = cl.user_session.get("mode", "mix")
    session_id = cl.user_session.get("session_id")

    # Create a message that we'll update
    msg = cl.Message(content="🔍 Searching knowledge base...")
    await msg.send()

    try:
        # Query with memory for multi-turn conversation support
        history_used = 0
        if session_id:
            result = await rag.query_with_memory(
                query=query,
                session_id=session_id,
                mode=mode,
            )
            history_used = result.get("history_used", 0)
            if history_used > 0:
                # Log that memory was used (for debugging)
                import logging

                logging.info(f"[CHAINLIT] Query used {history_used} history messages")
        else:
            # Fallback to non-memory query if session_id missing
            result = await rag.query_with_sources(
                query=query,
                mode=mode,
            )

        answer = result.get("answer", "No answer generated.")
        context = result.get("context", "")

        # Build response text - include telemetry without exposing raw context
        telemetry_lines = [
            "---",
            "### 🧾 Retrieval Telemetry",
            f"- Mode: `{mode}`",
            f"- Context size: `{len(context):,}` chars",
            f"- History used: `{history_used}`",
        ]
        response_text = f"{answer}\n\n" + "\n".join(telemetry_lines)

        # Update the message content directly
        msg.content = response_text
        await msg.update()

    except Exception as e:
        msg.content = f"**❌ Error:** {str(e)}"
        await msg.update()
