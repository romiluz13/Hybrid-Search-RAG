"""
Multi-format document processor using Docling.

This module handles conversion of various document formats to markdown:
- PDF documents
- Word documents (.docx, .doc)
- PowerPoint presentations (.pptx, .ppt)
- Excel spreadsheets (.xlsx, .xls)
- HTML files
- Markdown files
- Audio files (transcription via Whisper ASR)

All formats are converted to markdown for consistent processing.
The original DoclingDocument is preserved for HybridChunker usage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import ProcessedDocument

if TYPE_CHECKING:
    pass

logger = logging.getLogger("hybridrag.ingestion.document_processor")

# Supported file extensions by category
DOCLING_FORMATS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".md",
    ".markdown",
}

AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

TEXT_FORMATS = {".txt", ".text", ".rst", ".json", ".yaml", ".yml", ".csv"}


class DocumentProcessor:
    """
    Multi-format document processor using Docling.

    Converts various document formats to markdown while preserving
    the DoclingDocument for structure-aware chunking.

    Supported formats:
    - PDF, Word, PowerPoint, Excel, HTML, Markdown (via Docling)
    - Audio files (via Whisper ASR transcription)
    - Plain text files (direct read)
    """

    def __init__(self, enable_audio: bool = True) -> None:
        """
        Initialize document processor.

        Args:
            enable_audio: Whether to enable audio transcription.
        """
        self.enable_audio = enable_audio
        self._converter = None
        self._audio_converter = None

    def _get_converter(self) -> Any:
        """Lazy initialization of Docling DocumentConverter."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
                logger.info("Docling DocumentConverter initialized")
            except ImportError as e:
                logger.error("Docling not installed. Run: pip install docling")
                raise ImportError(
                    "Docling is required for document processing. "
                    "Install with: pip install docling"
                ) from e
        return self._converter

    def _get_audio_converter(self) -> Any:
        """Lazy initialization of audio converter with Whisper ASR."""
        if self._audio_converter is None:
            try:
                from docling.datamodel import asr_model_specs
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import AsrPipelineOptions
                from docling.document_converter import (
                    AudioFormatOption,
                    DocumentConverter,
                )
                from docling.pipeline.asr_pipeline import AsrPipeline

                # Configure ASR pipeline with Whisper Turbo model
                pipeline_options = AsrPipelineOptions()
                pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

                self._audio_converter = DocumentConverter(
                    format_options={
                        InputFormat.AUDIO: AudioFormatOption(
                            pipeline_cls=AsrPipeline,
                            pipeline_options=pipeline_options,
                        )
                    }
                )
                logger.info("Audio converter with Whisper ASR initialized")
            except ImportError as e:
                logger.warning(f"Audio transcription not available: {e}")
                raise ImportError(
                    "Audio transcription requires additional dependencies. "
                    "Install with: pip install docling[audio]"
                ) from e
        return self._audio_converter

    def process_file(self, file_path: str | Path) -> ProcessedDocument:
        """
        Process a single file and convert to ProcessedDocument.

        Args:
            file_path: Path to the file to process.

        Returns:
            ProcessedDocument with markdown content and optional DoclingDocument.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not supported.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        logger.info(f"Processing file: {file_path.name} (format: {file_ext})")

        # Route to appropriate processor
        if file_ext in AUDIO_FORMATS:
            if not self.enable_audio:
                raise ValueError(
                    f"Audio processing disabled. Cannot process: {file_path.name}"
                )
            return self._process_audio(file_path)

        elif file_ext in DOCLING_FORMATS:
            return self._process_docling(file_path)

        elif file_ext in TEXT_FORMATS:
            return self._process_text(file_path)

        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported: {DOCLING_FORMATS | AUDIO_FORMATS | TEXT_FORMATS}"
            )

    def _process_docling(self, file_path: Path) -> ProcessedDocument:
        """
        Process file using Docling DocumentConverter.

        Args:
            file_path: Path to the file.

        Returns:
            ProcessedDocument with DoclingDocument preserved.
        """
        try:
            converter = self._get_converter()
            result = converter.convert(file_path)

            # Export to markdown for consistent processing
            markdown_content = result.document.export_to_markdown()

            # Extract title from content or filename
            title = self._extract_title(markdown_content, file_path)

            # Build metadata
            metadata = self._build_metadata(file_path)

            logger.info(f"Successfully converted {file_path.name} to markdown")

            return ProcessedDocument(
                content=markdown_content,
                title=title,
                source=str(file_path),
                metadata=metadata,
                docling_document=result.document,  # Preserve for HybridChunker
                format_type=file_path.suffix.lower().lstrip("."),
            )

        except Exception as e:
            logger.error(f"Docling conversion failed for {file_path}: {e}")

            # Try PyMuPDF (fitz) fallback for PDFs
            if file_path.suffix.lower() == ".pdf":
                try:
                    logger.info(f"Attempting PyMuPDF (fitz) fallback for {file_path}")
                    return self._process_pdf_fitz(file_path)
                except Exception as fitz_e:
                    logger.error(f"PyMuPDF fallback failed: {fitz_e}")

            # Try fallback to text read
            logger.warning(f"Attempting fallback text read for {file_path}")
            return self._process_text(file_path)

    def _process_pdf_fitz(self, file_path: Path) -> ProcessedDocument:
        """
        Process PDF using PyMuPDF (fitz).

        Args:
            file_path: Path to PDF file.

        Returns:
            ProcessedDocument.
        """
        try:
            import fitz

            doc = fitz.open(file_path)
            content = ""

            # Extract text from each page
            for page in doc:
                text = page.get_text()
                content += text + "\n\n"

            doc.close()

            # Use filename as title if no metadata found
            title = doc.metadata.get("title") if doc.metadata else None
            if not title or title.strip() == "":
                title = file_path.stem

            metadata = self._build_metadata(file_path)
            metadata["pdf_producer"] = (
                doc.metadata.get("producer", "unknown") if doc.metadata else "unknown"
            )

            logger.info(f"Successfully processed PDF with PyMuPDF: {file_path.name}")

            return ProcessedDocument(
                content=content,
                title=title,
                source=str(file_path),
                metadata=metadata,
                docling_document=None,
                format_type="pdf",
            )

        except ImportError as e:
            raise ImportError("PyMuPDF (fitz) not installed") from e
        except Exception as e:
            raise ValueError(f"Failed to process PDF with PyMuPDF: {e}") from e

    def _process_audio(self, file_path: Path) -> ProcessedDocument:
        """
        Process audio file using Whisper ASR transcription.

        Args:
            file_path: Path to the audio file.

        Returns:
            ProcessedDocument with transcribed content.
        """
        try:
            converter = self._get_audio_converter()
            result = converter.convert(file_path)

            # Export transcription to markdown
            markdown_content = result.document.export_to_markdown()

            # Use filename as title
            title = f"Transcription: {file_path.stem}"

            # Build metadata with audio-specific info
            metadata = self._build_metadata(file_path)
            metadata["transcription_model"] = "whisper-turbo"
            metadata["original_format"] = file_path.suffix.lower()

            logger.info(f"Successfully transcribed {file_path.name}")

            return ProcessedDocument(
                content=markdown_content,
                title=title,
                source=str(file_path),
                metadata=metadata,
                docling_document=result.document,
                format_type="audio",
            )

        except Exception as e:
            logger.error(f"Audio transcription failed for {file_path}: {e}")
            raise ValueError(f"Failed to transcribe audio file: {e}") from e

    def _process_text(self, file_path: Path) -> ProcessedDocument:
        """
        Process plain text file by direct read.

        Args:
            file_path: Path to the text file.

        Returns:
            ProcessedDocument without DoclingDocument.
        """
        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")

            title = self._extract_title(content, file_path)
            metadata = self._build_metadata(file_path)

            logger.info(f"Read text file: {file_path.name}")

            return ProcessedDocument(
                content=content,
                title=title,
                source=str(file_path),
                metadata=metadata,
                docling_document=None,  # No DoclingDocument for plain text
                format_type="text",
            )

        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise ValueError(f"Failed to read file: {e}") from e

    def _extract_title(self, content: str, file_path: Path) -> str:
        """
        Extract title from document content or use filename.

        Args:
            content: Document content.
            file_path: File path.

        Returns:
            Document title.
        """
        # Try to find markdown title (# Title)
        lines = content.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        # Fallback to filename
        return file_path.stem

    def _build_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Build metadata dictionary for a file.

        Args:
            file_path: File path.

        Returns:
            Metadata dictionary.
        """
        stat = file_path.stat()
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_extension": file_path.suffix.lower(),
            "modified_time": stat.st_mtime,
        }

    @staticmethod
    def get_supported_extensions() -> set[str]:
        """Get all supported file extensions."""
        return DOCLING_FORMATS | AUDIO_FORMATS | TEXT_FORMATS

    @staticmethod
    def is_supported(file_path: str | Path) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in DocumentProcessor.get_supported_extensions()


def create_document_processor(enable_audio: bool = True) -> DocumentProcessor:
    """
    Create a DocumentProcessor instance.

    Args:
        enable_audio: Whether to enable audio transcription.

    Returns:
        DocumentProcessor instance.
    """
    return DocumentProcessor(enable_audio=enable_audio)
