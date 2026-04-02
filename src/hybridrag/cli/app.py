#!/usr/bin/env python3
"""
HybridRAG CLI - Typer-based command-line interface.

Provides commands for:
- Project initialization
- Document ingestion
- Queries and chat
- System status
- Index management
- Benchmarking
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..config.settings import get_settings
from ..core.rag import create_hybridrag

# Import existing conversational interface
from .main import conversation_loop

# Initialize Typer app
app = typer.Typer(
    name="hybridrag",
    help="HybridRAG - MongoDB Atlas + Voyage AI RAG System",
    add_completion=False,
)

# Rich console
console = Console()

SUPPORTED_QUERY_MODES = ["naive", "local", "global", "hybrid", "mix", "bypass"]


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        from .. import __version__

        console.print(f"HybridRAG version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    HybridRAG CLI - Intelligent knowledge base with MongoDB Atlas.

    Use 'hybridrag COMMAND --help' for command-specific help.
    """
    pass


@app.command()
def init(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory to initialize (default: current directory)",
    ),
    mongodb_uri: str | None = typer.Option(
        None,
        "--mongodb-uri",
        "-m",
        help="MongoDB Atlas connection URI",
    ),
    voyage_key: str | None = typer.Option(
        None,
        "--voyage-key",
        "-v",
        help="Voyage AI API key",
    ),
    anthropic_key: str | None = typer.Option(
        None,
        "--anthropic-key",
        "-a",
        help="Anthropic API key",
    ),
):
    """
    Initialize a new HybridRAG project.

    Creates .env file with configuration.
    """
    console.print("[bold blue]Initializing HybridRAG project...[/bold blue]")

    env_path = directory / ".env"

    if env_path.exists():
        overwrite = typer.confirm(f"{env_path} already exists. Overwrite?")
        if not overwrite:
            console.print("[yellow]Initialization cancelled.[/yellow]")
            raise typer.Exit()

    # Create .env file
    env_content = """# HybridRAG Configuration

# MongoDB Atlas
MONGODB_URI={mongodb_uri}
MONGODB_DATABASE=hybridrag_db

# Voyage AI
VOYAGE_API_KEY={voyage_key}

# Anthropic (Claude)
ANTHROPIC_API_KEY={anthropic_key}
LLM_PROVIDER=anthropic
""".format(
        mongodb_uri=mongodb_uri or "mongodb+srv://user:pass@cluster.mongodb.net/",
        voyage_key=voyage_key or "pa-xxxxx",
        anthropic_key=anthropic_key or "sk-ant-xxxxx",
    )

    env_path.write_text(env_content)

    console.print(f"[green]✓ Created {env_path}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Edit .env with your API keys")
    console.print("2. Run: [cyan]hybridrag status[/cyan] to verify connection")
    console.print("3. Run: [cyan]hybridrag ingest <path>[/cyan] to add documents")
    console.print("4. Run: [cyan]hybridrag chat[/cyan] to start querying")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Recursively ingest directories",
    ),
):
    """
    Ingest documents from file or directory.

    Supports: .txt, .md, .pdf, .docx, .html
    """

    async def _ingest():
        from ..core.mongodb_client import close_shared_client

        rag = await create_hybridrag()
        try:
            if not path.exists():
                console.print(f"[red]Error: Path not found: {path}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold blue]Ingesting from {path}...[/bold blue]\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing...", total=None)

                # L28: Files are ingested sequentially. For large directories,
                # consider using `hybridrag ingest` with batch-mode support
                # or the Python API directly with asyncio.gather for parallelism.
                count = 0
                if path.is_file():
                    await rag.ingest_file(str(path))
                    count = 1
                elif path.is_dir():
                    for file in path.rglob("*") if recursive else path.glob("*"):
                        if file.suffix in [".txt", ".md", ".pdf", ".docx", ".html"]:
                            progress.update(
                                task, description=f"Processing {file.name}..."
                            )
                            await rag.ingest_file(str(file))
                            count += 1

            console.print(f"\n[green]✓ Successfully ingested {count} file(s)[/green]")
        finally:
            close_shared_client()

    asyncio.run(_ingest())


@app.command(name="ingest-url")
def ingest_url(
    url: str = typer.Argument(..., help="URL to ingest"),
):
    """
    Ingest content from a single URL.
    """

    async def _ingest():
        from ..core.mongodb_client import close_shared_client

        rag = await create_hybridrag()
        try:
            console.print(f"[bold blue]Ingesting URL: {url}...[/bold blue]\n")

            with console.status("[bold green]Fetching and processing...[/bold green]"):
                result = await rag.ingest_url(url)

            if result.success:
                console.print(f"[green]✓ Successfully ingested: {result.title}[/green]")
                console.print(
                    f"[green]  Chunks created: {result.chunks_created}[/green]"
                )
            else:
                console.print(f"[red]✗ Failed: {result.errors}[/red]")
                raise typer.Exit(1)
        finally:
            close_shared_client()

    asyncio.run(_ingest())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    mode: str = typer.Option(
        "mix",
        "--mode",
        "-m",
        help=f"Search mode: {', '.join(SUPPORTED_QUERY_MODES)}",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
):
    """
    Run a single query against the knowledge base.
    """

    async def _query():
        from ..core.mongodb_client import close_shared_client

        rag = await create_hybridrag()
        try:
            console.print(f"\n[bold blue]Query:[/bold blue] {question}\n")

            if mode not in SUPPORTED_QUERY_MODES:
                console.print(
                    f"[red]Unsupported mode '{mode}'. Choose from: {', '.join(SUPPORTED_QUERY_MODES)}[/red]"
                )
                raise typer.Exit(1)

            with console.status("[bold green]Searching...[/bold green]"):
                answer = await rag.query(
                    query=question,
                    mode=mode,
                    top_k=top_k,
                )

            console.print("[bold blue]Answer:[/bold blue]")
            console.print(Panel(answer, style="green", padding=(1, 2)))
        finally:
            close_shared_client()

    asyncio.run(_query())


@app.command()
def chat():
    """
    Start an interactive chat session.

    Conversational interface with memory and streaming.
    """

    async def _chat():
        from ..core.mongodb_client import close_shared_client

        settings = get_settings()

        console.print(
            Panel(
                "[bold blue]HybridRAG Interactive Chat[/bold blue]\n\n"
                f"[dim]Database: {settings.mongodb_database}[/dim]\n"
                f"[dim]Provider: {settings.llm_provider}[/dim]\n\n"
                "[dim]Commands: 'exit', 'info', 'clear', 'new', 'history'[/dim]",
                style="blue",
                padding=(1, 2),
            )
        )

        try:
            with console.status("[bold green]Initializing...[/bold green]"):
                rag = await create_hybridrag(settings=settings)

            console.print("[green]Ready![/green]\n")
            await conversation_loop(rag)
        finally:
            close_shared_client()

    asyncio.run(_chat())


@app.command()
def status():
    """
    Show system status and configuration.
    """

    async def _status():
        from ..core.mongodb_client import close_shared_client

        settings = get_settings()
        rag = await create_hybridrag()
        try:
            status_data = await rag.get_status()

            # Create status table
            table = Table(title="HybridRAG System Status", show_header=True)
            table.add_column("Setting", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            # Add configuration
            table.add_row("MongoDB Database", settings.mongodb_database)
            table.add_row("MongoDB Workspace", settings.mongodb_workspace)
            table.add_row("LLM Provider", settings.llm_provider)
            table.add_row("Embedding Provider", settings.embedding_provider)
            table.add_row("Voyage Embedding Model", settings.voyage_embedding_model)

            # Add status data
            for key, value in status_data.items():
                table.add_row(key, str(value))

            console.print(table)
        finally:
            close_shared_client()

    asyncio.run(_status())


# Index management subcommand group
index_app = typer.Typer(help="Manage MongoDB Atlas search indexes")
app.add_typer(index_app, name="index")


@index_app.command("create")
def index_create():
    """
    Create required Atlas Search indexes.

    Creates vector and text search indexes.
    """

    settings = get_settings()
    console.print("[bold blue]Atlas Search Index Guidance[/bold blue]\n")
    console.print(
        "[yellow]Search index creation is handled by the MongoDB-backed storage layer during initialization, "
        "and the exact collection names depend on the configured workspace.[/yellow]"
    )
    console.print(f"\nDatabase: [cyan]{settings.mongodb_database}[/cyan]")
    console.print(f"Workspace: [cyan]{settings.mongodb_workspace}[/cyan]")
    console.print(
        f"Embedding model: [cyan]{settings.voyage_embedding_model}[/cyan] "
        f"(dimension {settings.embedding_dim})"
    )
    console.print(
        "\n[dim]Use 'hybridrag status' after initialization and inspect the generated MongoDB collections "
        "before creating Atlas Search indexes manually.[/dim]"
    )


@index_app.command("list")
def index_list():
    """
    List all indexes in the collection.
    """

    async def _list():
        from pymongo import MongoClient

        settings = get_settings()
        client = MongoClient(settings.mongodb_uri.get_secret_value())
        try:
            db = client[settings.mongodb_database]

            table = Table(title="MongoDB Collection Indexes", show_header=True)
            table.add_column("Collection", style="magenta")
            table.add_column("Name", style="cyan")
            table.add_column("Keys", style="green")
            table.add_column("Type", style="yellow")

            for collection_name in sorted(db.list_collection_names()):
                collection = db[collection_name]
                for idx in collection.list_indexes():
                    name = idx.get("name", "N/A")
                    keys = str(idx.get("key", {}))
                    idx_type = idx.get("type", "standard")
                    table.add_row(collection_name, name, keys, idx_type)

            console.print(table)
        finally:
            client.close()

    asyncio.run(_list())


@app.command()
def benchmark():
    """
    Run performance benchmarks.

    Tests query latency and throughput.
    """
    console.print(
        "[yellow]Benchmark command - Run with: pytest tests/benchmarks/ -m benchmark[/yellow]"
    )
    console.print("\nOr use:")
    console.print("  make benchmark          # Run benchmarks")
    console.print("  make benchmark-save     # Save baseline")


def run():
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    run()
