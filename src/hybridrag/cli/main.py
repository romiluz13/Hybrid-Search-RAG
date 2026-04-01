#!/usr/bin/env python3
"""
HybridRAG CLI - Interactive conversational interface.

Provides a Rich-based terminal UI for:
- Natural language queries against the knowledge base
- Multi-turn conversations with memory
- Real-time streaming responses
- System status and configuration info
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..config.settings import Settings, get_settings
from ..core.rag import HybridRAG, create_hybridrag

if TYPE_CHECKING:
    pass


# Initialize console
console = Console() if RICH_AVAILABLE else None


def display_welcome(settings: Settings) -> None:
    """Display welcome message with configuration info."""
    if not RICH_AVAILABLE:
        print("=" * 60)
        print("HybridRAG CLI - Intelligent Knowledge Base Search")
        print("=" * 60)
        print(f"LLM Provider: {settings.llm_provider}")
        print(f"Database: {settings.mongodb_database}")
        print("Commands: 'exit', 'info', 'clear', 'new' (new session)")
        print("=" * 60)
        return

    welcome = Panel(
        "[bold blue]HybridRAG CLI[/bold blue]\n\n"
        "[green]Intelligent knowledge base search with MongoDB Atlas + Voyage AI[/green]\n\n"
        f"[dim]LLM Provider: {settings.llm_provider}[/dim]\n"
        f"[dim]Database: {settings.mongodb_database}[/dim]\n\n"
        "[dim]Commands: 'exit' to quit, 'info' for status, 'clear' to clear, 'new' for new session[/dim]",
        style="blue",
        padding=(1, 2),
    )
    console.print(welcome)
    console.print()


async def display_status(rag: HybridRAG) -> None:
    """Display system status information."""
    status = await rag.get_status()

    if not RICH_AVAILABLE:
        print("\n=== System Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        print("=====================\n")
        return

    status_text = "\n".join(
        f"[cyan]{key}:[/cyan] {value}" for key, value in status.items()
    )
    panel = Panel(
        status_text,
        title="[bold]System Status[/bold]",
        style="green",
    )
    console.print(panel)
    console.print()


async def process_query(
    rag: HybridRAG,
    query: str,
    session_id: str,
) -> str:
    """
    Process a user query and return the response.

    Args:
        rag: HybridRAG instance
        query: User query text
        session_id: Conversation session ID

    Returns:
        Response text from RAG system
    """
    try:
        result = await rag.query_with_memory(
            query=query,
            session_id=session_id,
            mode="mix",
            top_k=10,
        )
        return result.get("answer", "No response generated.")
    except Exception as e:
        return f"Error processing query: {e}"


def display_response(response: str) -> None:
    """Display the assistant response."""
    if not RICH_AVAILABLE:
        print(f"\nAssistant: {response}\n")
        return

    # Render as markdown for better formatting
    console.print("[bold blue]Assistant:[/bold blue]")
    try:
        md = Markdown(response)
        console.print(md)
    except Exception as e:
        # Log markdown rendering failure but show raw response
        import logging

        logging.debug(f"Markdown rendering failed: {e}")
        console.print(response)
    console.print()


def get_user_input() -> str:
    """Get input from the user."""
    if not RICH_AVAILABLE:
        return input("You: ").strip()

    return Prompt.ask("[bold green]You[/bold green]")


async def conversation_loop(rag: HybridRAG) -> None:
    """
    Main conversation loop.

    Args:
        rag: Initialized HybridRAG instance
    """
    import uuid

    # Create a new session for this conversation.
    # L30: The session UUID is generated here and persisted to MongoDB lazily
    # on the first call to rag.query_with_memory(), which calls
    # ConversationMemory.create_session() if the session doesn't exist yet.
    session_id = str(uuid.uuid4())

    if RICH_AVAILABLE:
        console.print(f"[dim]Session: {session_id[:8]}...[/dim]\n")
    else:
        print(f"Session: {session_id[:8]}...\n")

    while True:
        try:
            # Get user input
            user_input = get_user_input()

            if not user_input:
                continue

            # Handle commands
            command = user_input.lower().strip()

            if command in ("exit", "quit", "q"):
                if RICH_AVAILABLE:
                    console.print("[yellow]Goodbye![/yellow]")
                else:
                    print("Goodbye!")
                break

            if command == "info":
                await display_status(rag)
                continue

            if command == "clear":
                if RICH_AVAILABLE:
                    console.clear()
                else:
                    print("\033c", end="")
                display_welcome(rag.settings)
                continue

            if command == "new":
                session_id = str(uuid.uuid4())
                if RICH_AVAILABLE:
                    console.print(
                        f"[green]New session created: {session_id[:8]}...[/green]\n"
                    )
                else:
                    print(f"New session created: {session_id[:8]}...\n")
                continue

            if command == "history":
                history = await rag.get_conversation_history(session_id, limit=10)
                if not history:
                    if RICH_AVAILABLE:
                        console.print("[dim]No conversation history yet.[/dim]")
                    else:
                        print("No conversation history yet.")
                else:
                    for msg in history:
                        role = msg.get("role", "unknown").capitalize()
                        content = msg.get("content", "")[:100]
                        if RICH_AVAILABLE:
                            console.print(f"[cyan]{role}:[/cyan] {content}...")
                        else:
                            print(f"{role}: {content}...")
                console.print() if RICH_AVAILABLE else print()
                continue

            # Ingest command: ingest <file_or_folder_path>
            if command.startswith("ingest "):
                path = user_input[7:].strip()
                if not path:
                    if RICH_AVAILABLE:
                        console.print("[red]Usage: ingest <file_or_folder_path>[/red]")
                    else:
                        print("Usage: ingest <file_or_folder_path>")
                    continue

                import os

                if not os.path.exists(path):
                    if RICH_AVAILABLE:
                        console.print(f"[red]Path not found: {path}[/red]")
                    else:
                        print(f"Path not found: {path}")
                    continue

                try:
                    if RICH_AVAILABLE:
                        with console.status(
                            f"[bold green]Ingesting {path}...[/bold green]"
                        ):
                            if os.path.isfile(path):
                                await rag.ingest_file(path)
                                console.print(
                                    f"[green]Successfully ingested: {path}[/green]"
                                )
                            else:
                                # It's a folder - ingest all supported files
                                count = 0
                                for root, _, files in os.walk(path):
                                    for f in files:
                                        if f.endswith(
                                            (".txt", ".md", ".pdf", ".docx", ".html")
                                        ):
                                            filepath = os.path.join(root, f)
                                            await rag.ingest_file(filepath)
                                            count += 1
                                console.print(
                                    f"[green]Successfully ingested {count} files from: {path}[/green]"
                                )
                    else:
                        print(f"Ingesting {path}...")
                        if os.path.isfile(path):
                            await rag.ingest_file(path)
                            print(f"Successfully ingested: {path}")
                        else:
                            count = 0
                            for root, _, files in os.walk(path):
                                for f in files:
                                    if f.endswith(
                                        (".txt", ".md", ".pdf", ".docx", ".html")
                                    ):
                                        filepath = os.path.join(root, f)
                                        await rag.ingest_file(filepath)
                                        count += 1
                            print(f"Successfully ingested {count} files from: {path}")
                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Error ingesting: {e}[/red]")
                    else:
                        print(f"Error ingesting: {e}")
                continue

            # Ingest URL command: ingest-url <url>
            if command.startswith("ingest-url "):
                url = user_input[11:].strip()
                if not url:
                    if RICH_AVAILABLE:
                        console.print("[red]Usage: ingest-url <url>[/red]")
                    else:
                        print("Usage: ingest-url <url>")
                    continue

                # Validate URL format
                from urllib.parse import urlparse

                try:
                    parsed = urlparse(url)
                    if not all([parsed.scheme in ["http", "https"], parsed.netloc]):
                        if RICH_AVAILABLE:
                            console.print(f"[red]Invalid URL format: {url}[/red]")
                        else:
                            print(f"Invalid URL format: {url}")
                        continue
                except Exception:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Invalid URL format: {url}[/red]")
                    else:
                        print(f"Invalid URL format: {url}")
                    continue

                try:
                    if RICH_AVAILABLE:
                        with console.status(
                            f"[bold green]Ingesting URL {url}...[/bold green]"
                        ):
                            result = await rag.ingest_url(url)
                            if result.success:
                                console.print(
                                    f"[green]Successfully ingested: {result.title}[/green]"
                                )
                                console.print(
                                    f"[green]Chunks created: {result.chunks_created}[/green]"
                                )
                            else:
                                console.print(
                                    f"[red]Failed to ingest: {result.errors}[/red]"
                                )
                    else:
                        print(f"Ingesting URL {url}...")
                        result = await rag.ingest_url(url)
                        if result.success:
                            print(f"Successfully ingested: {result.title}")
                            print(f"Chunks created: {result.chunks_created}")
                        else:
                            print(f"Failed to ingest: {result.errors}")
                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Error ingesting URL: {e}[/red]")
                    else:
                        print(f"Error ingesting URL: {e}")
                continue

            # Ingest website command: ingest-website <url> [max_pages]
            if command.startswith("ingest-website "):
                parts = user_input[15:].strip().split()
                if not parts:
                    if RICH_AVAILABLE:
                        console.print(
                            "[red]Usage: ingest-website <url> [max_pages][/red]"
                        )
                    else:
                        print("Usage: ingest-website <url> [max_pages]")
                    continue

                url = parts[0]
                max_pages = 10
                if len(parts) > 1:
                    try:
                        max_pages = int(parts[1])
                    except ValueError:
                        if RICH_AVAILABLE:
                            console.print(f"[red]Invalid max_pages: {parts[1]}[/red]")
                        else:
                            print(f"Invalid max_pages: {parts[1]}")
                        continue

                # Validate URL format
                from urllib.parse import urlparse

                try:
                    parsed = urlparse(url)
                    if not all([parsed.scheme in ["http", "https"], parsed.netloc]):
                        if RICH_AVAILABLE:
                            console.print(f"[red]Invalid URL format: {url}[/red]")
                        else:
                            print(f"Invalid URL format: {url}")
                        continue
                except Exception:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Invalid URL format: {url}[/red]")
                    else:
                        print(f"Invalid URL format: {url}")
                    continue

                try:

                    def progress_callback(current: int, total: int) -> None:
                        if RICH_AVAILABLE:
                            console.print(
                                f"[yellow]Processing page {current}/{total}...[/yellow]"
                            )
                        else:
                            print(f"Processing page {current}/{total}...")

                    if RICH_AVAILABLE:
                        with console.status(
                            f"[bold green]Crawling website {url}...[/bold green]"
                        ):
                            results = await rag.ingest_website(
                                url,
                                max_pages=max_pages,
                                progress_callback=progress_callback,
                            )
                            successful = sum(1 for r in results if r.success)
                            total_chunks = sum(r.chunks_created for r in results)
                            console.print(
                                f"[green]Successfully ingested {successful}/{len(results)} pages[/green]"
                            )
                            console.print(
                                f"[green]Total chunks created: {total_chunks}[/green]"
                            )
                            if successful < len(results):
                                failed = [r.title for r in results if not r.success]
                                console.print(
                                    f"[yellow]Failed pages: {failed}[/yellow]"
                                )
                    else:
                        print(f"Crawling website {url}...")
                        results = await rag.ingest_website(
                            url,
                            max_pages=max_pages,
                            progress_callback=progress_callback,
                        )
                        successful = sum(1 for r in results if r.success)
                        total_chunks = sum(r.chunks_created for r in results)
                        print(
                            f"Successfully ingested {successful}/{len(results)} pages"
                        )
                        print(f"Total chunks created: {total_chunks}")
                        if successful < len(results):
                            failed = [r.title for r in results if not r.success]
                            print(f"Failed pages: {failed}")
                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Error ingesting website: {e}[/red]")
                    else:
                        print(f"Error ingesting website: {e}")
                continue

            # Process the query
            if RICH_AVAILABLE:
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    response = await process_query(rag, user_input, session_id)
            else:
                print("Thinking...")
                response = await process_query(rag, user_input, session_id)

            # Display the response
            display_response(response)

        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            else:
                print("\nInterrupted. Type 'exit' to quit.")
            continue
        except EOFError:
            break


async def main() -> None:
    """Main entry point for the CLI."""
    settings = get_settings()

    # Display welcome
    display_welcome(settings)

    # Initialize HybridRAG
    if RICH_AVAILABLE:
        with console.status("[bold green]Initializing HybridRAG...[/bold green]"):
            rag = await create_hybridrag(settings=settings)
    else:
        print("Initializing HybridRAG...")
        rag = await create_hybridrag(settings=settings)

    if RICH_AVAILABLE:
        console.print("[green]Ready![/green]\n")
    else:
        print("Ready!\n")

    # Start conversation loop
    await conversation_loop(rag)


def run_cli() -> None:
    """Run the CLI (synchronous entry point)."""
    if not RICH_AVAILABLE:
        print("Warning: 'rich' package not installed. Using basic output.")
        print("Install with: pip install rich\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    run_cli()
