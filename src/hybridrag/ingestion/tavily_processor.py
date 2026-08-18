"""
Tavily Web Content Processor for HybridRAG.

This module provides Tavily API integration for web content extraction and crawling.
Follows official Tavily SDK patterns and integrates with HybridRAG ingestion pipeline.
"""

from __future__ import annotations

# Optional Tavily imports and local fallback exception classes intentionally share
# one public compatibility surface.
# pyright: reportRedeclaration=false, reportAssignmentType=false
import importlib.util
import logging
from typing import Any, Literal, cast
from urllib.parse import urlparse

from .types import ProcessedDocument

logger = logging.getLogger("hybridrag.ingestion.tavily")

# Check if Tavily SDK is available using importlib.util.find_spec()
# This checks for module existence WITHOUT importing (no side effects or warnings)
# Per Python docs: https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
TAVILY_AVAILABLE = importlib.util.find_spec("tavily") is not None


class _FallbackBadRequestError(Exception):
    pass


class _FallbackForbiddenError(Exception):
    pass


class _FallbackInvalidAPIKeyError(Exception):
    pass


class _FallbackMissingAPIKeyError(Exception):
    pass


class _FallbackTavilyTimeoutError(Exception):
    pass


class _FallbackUsageLimitExceededError(Exception):
    pass


if TAVILY_AVAILABLE:
    from tavily import AsyncTavilyClient as _AsyncTavilyClient
    from tavily.errors import (
        BadRequestError as _BadRequestError,
    )
    from tavily.errors import (
        ForbiddenError as _ForbiddenError,
    )
    from tavily.errors import (
        InvalidAPIKeyError as _InvalidAPIKeyError,
    )
    from tavily.errors import (
        MissingAPIKeyError as _MissingAPIKeyError,
    )
    from tavily.errors import (
        TimeoutError as _TavilyTimeoutError,
    )
    from tavily.errors import (
        UsageLimitExceededError as _UsageLimitExceededError,
    )

    AsyncTavilyClient: Any = _AsyncTavilyClient
    BadRequestError = cast(type[Exception], _BadRequestError)
    ForbiddenError = cast(type[Exception], _ForbiddenError)
    InvalidAPIKeyError = cast(type[Exception], _InvalidAPIKeyError)
    MissingAPIKeyError = cast(type[Exception], _MissingAPIKeyError)
    TavilyTimeoutError = cast(type[Exception], _TavilyTimeoutError)
    UsageLimitExceededError = cast(type[Exception], _UsageLimitExceededError)
else:
    AsyncTavilyClient = None
    BadRequestError = _FallbackBadRequestError
    ForbiddenError = _FallbackForbiddenError
    InvalidAPIKeyError = _FallbackInvalidAPIKeyError
    MissingAPIKeyError = _FallbackMissingAPIKeyError
    TavilyTimeoutError = _FallbackTavilyTimeoutError
    UsageLimitExceededError = _FallbackUsageLimitExceededError

TimeoutError = TavilyTimeoutError


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL string to validate.

    Returns:
        True if URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except Exception:
        return False


class TavilyProcessor:
    """
    Tavily API processor for web content extraction and crawling.

    This class wraps Tavily's AsyncTavilyClient and provides methods
    that return ProcessedDocument objects compatible with HybridRAG's
    ingestion pipeline.

    Usage:
        ```python
        processor = TavilyProcessor(api_key="your-key")
        doc = await processor.extract_url("https://example.com")
        docs = await processor.crawl_website("https://example.com", max_pages=10)
        ```
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize Tavily processor.

        Args:
            api_key: Tavily API key.

        Raises:
            ImportError: If tavily-python package is not installed.
            MissingAPIKeyError: If API key is not provided.
        """
        if not TAVILY_AVAILABLE:
            raise ImportError(
                "Tavily SDK not available. Install with: pip install tavily-python"
            )

        if not api_key:
            raise MissingAPIKeyError("Tavily API key is required")

        self.client = AsyncTavilyClient(api_key=api_key)
        logger.info("[TAVILY] Processor initialized")

    async def extract_url(
        self,
        url: str,
        extract_depth: Literal["basic", "advanced"] = "advanced",
        format: Literal["markdown", "text"] = "markdown",
        timeout: float = 30.0,
        include_images: bool = False,
    ) -> ProcessedDocument:
        """
        Extract content from a single URL using Tavily Extract API.

        Args:
            url: URL to extract content from.
            extract_depth: Extraction depth ("basic" or "advanced").
                          "advanced" provides better quality for RAG.
            format: Output format ("markdown" or "text").
                   "markdown" is RAG-optimized.
            timeout: Request timeout in seconds (default: 30).
            include_images: Whether to include images in extraction.

        Returns:
            ProcessedDocument with extracted content.

        Raises:
            ValueError: If URL is invalid.
            MissingAPIKeyError: If API key is not configured.
            UsageLimitExceededError: If rate limit is exceeded.
            BadRequestError: If request is invalid.
            TavilyTimeoutError: If request times out.
            ForbiddenError: If access is forbidden.
        """
        if not validate_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        logger.info(f"[TAVILY] Extracting content from: {url}")

        try:
            result = await self.client.extract(
                urls=url,
                extract_depth=extract_depth,
                format=format,
                timeout=timeout,
                include_images=include_images,
            )

            # Parse response structure
            logger.debug(f"[TAVILY] Raw response keys: {list(result.keys())}")
            results = result.get("results", [])
            failed_results = result.get("failed_results", [])

            logger.debug(
                f"[TAVILY] Results count: {len(results)}, Failed count: {len(failed_results)}"
            )

            if failed_results:
                failed_urls = [r.get("url", "unknown") for r in failed_results]
                logger.warning(f"[TAVILY] Failed to extract from URLs: {failed_urls}")

            if not results:
                logger.error(
                    f"[TAVILY] No results in response. Full response: {result}"
                )
                raise ValueError(
                    f"No content extracted from URL: {url}. "
                    f"Failed results: {failed_results}"
                )

            # Get first result (single URL extraction)
            content_data = results[0]
            logger.debug(f"[TAVILY] Content data keys: {list(content_data.keys())}")
            # Tavily returns 'raw_content' field, not 'content'
            content = content_data.get("raw_content", content_data.get("content", ""))
            title = content_data.get("title", url)
            metadata = content_data.get("metadata", {})
            extracted_url = content_data.get("url", url)

            logger.debug(f"[TAVILY] Content length: {len(content)}, Title: {title}")

            if not content or not content.strip():
                logger.error(
                    f"[TAVILY] Empty content. Content data keys: {list(content_data.keys())}"
                )
                raise ValueError(f"Empty content extracted from URL: {url}")

            logger.info(
                f"[TAVILY] Extracted {len(content)} chars from: {extracted_url}"
            )

            # Create ProcessedDocument compatible with ingestion pipeline
            return ProcessedDocument(
                content=content,
                title=title,
                source=extracted_url,
                metadata={
                    **metadata,
                    "source_type": "web",
                    "extraction_method": "tavily_extract",
                    "extract_depth": extract_depth,
                    "format": format,
                },
                format_type="web",
            )

        except (
            MissingAPIKeyError,
            InvalidAPIKeyError,
            UsageLimitExceededError,
            BadRequestError,
            TavilyTimeoutError,
            ForbiddenError,
        ) as e:
            logger.error(f"[TAVILY] API error extracting {url}: {e}")
            raise
        except Exception as e:
            logger.exception(f"[TAVILY] Unexpected error extracting {url}: {e}")
            raise ValueError(f"Failed to extract content from {url}: {e}") from e

    async def crawl_website(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 10,
        extract_depth: Literal["basic", "advanced"] = "advanced",
        format: Literal["markdown", "text"] = "markdown",
        timeout: float = 150.0,
        include_images: bool = False,
    ) -> list[ProcessedDocument]:
        """
        Crawl a website using Tavily Crawl API.

        Args:
            url: Base URL to start crawling from.
            max_depth: Maximum crawl depth (default: 2).
            max_pages: Maximum number of pages to crawl (default: 10).
            extract_depth: Extraction depth ("basic" or "advanced").
                          "advanced" provides better quality for RAG.
            format: Output format ("markdown" or "text").
                   "markdown" is RAG-optimized.
            timeout: Request timeout in seconds (default: 150 for crawls).
            include_images: Whether to include images in extraction.

        Returns:
            List of ProcessedDocument objects, one per crawled page.

        Raises:
            ValueError: If URL is invalid.
            MissingAPIKeyError: If API key is not configured.
            UsageLimitExceededError: If rate limit is exceeded.
            BadRequestError: If request is invalid.
            TavilyTimeoutError: If request times out.
            ForbiddenError: If access is forbidden.
        """
        if not validate_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        logger.info(
            f"[TAVILY] Crawling website: {url} (max_depth={max_depth}, max_pages={max_pages})"
        )

        try:
            result = await self.client.crawl(
                url=url,
                max_depth=max_depth,
                limit=max_pages,
                extract_depth=extract_depth,
                format=format,
                timeout=timeout,
                include_images=include_images,
            )

            # Parse response structure (crawl returns "results" not "pages")
            logger.debug(f"[TAVILY] Crawl response keys: {list(result.keys())}")
            pages = result.get("results", result.get("pages", []))

            logger.debug(f"[TAVILY] Pages count: {len(pages)}")

            if not pages:
                logger.error(
                    f"[TAVILY] No pages in crawl response. Full response: {result}"
                )
                raise ValueError(f"No pages crawled from URL: {url}")

            logger.info(f"[TAVILY] Crawled {len(pages)} pages from: {url}")

            # Convert each page to ProcessedDocument
            processed_docs = []
            for page_data in pages:
                # Tavily returns 'raw_content' field, not 'content'
                content = page_data.get("raw_content", page_data.get("content", ""))
                title = page_data.get("title", url)
                metadata = page_data.get("metadata", {})
                page_url = page_data.get("url", url)

                if not content or not content.strip():
                    logger.warning(f"[TAVILY] Skipping empty page: {page_url}")
                    continue

                processed_docs.append(
                    ProcessedDocument(
                        content=content,
                        title=title,
                        source=page_url,
                        metadata={
                            **metadata,
                            "source_type": "web",
                            "extraction_method": "tavily_crawl",
                            "extract_depth": extract_depth,
                            "format": format,
                            "crawl_depth": metadata.get("depth", 0),
                        },
                        format_type="web",
                    )
                )

            if not processed_docs:
                raise ValueError(
                    f"No valid content extracted from crawled pages: {url}"
                )

            logger.info(f"[TAVILY] Successfully processed {len(processed_docs)} pages")
            return processed_docs

        except (
            MissingAPIKeyError,
            InvalidAPIKeyError,
            UsageLimitExceededError,
            BadRequestError,
            TavilyTimeoutError,
            ForbiddenError,
        ) as e:
            logger.error(f"[TAVILY] API error crawling {url}: {e}")
            raise
        except Exception as e:
            logger.exception(f"[TAVILY] Unexpected error crawling {url}: {e}")
            raise ValueError(f"Failed to crawl website {url}: {e}") from e


def create_tavily_processor(api_key: str) -> TavilyProcessor:
    """
    Create a TavilyProcessor instance.

    Args:
        api_key: Tavily API key.

    Returns:
        TavilyProcessor instance.

    Raises:
        ImportError: If tavily-python package is not installed.
        MissingAPIKeyError: If API key is not provided.
    """
    return TavilyProcessor(api_key=api_key)
