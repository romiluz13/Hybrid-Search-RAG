#!/usr/bin/env python3
"""
RAGAS Evaluation Framework for HybridRAG.

Evaluates RAG response quality using RAGAS metrics:
- Faithfulness: Is the answer factually accurate based on context?
- Answer Relevance: Is the answer relevant to the question?
- Context Recall: Is all relevant information retrieved?
- Context Precision: Is retrieved context clean without noise?

Usage:
    # Run with default sample dataset
    python -m hybridrag.evaluation.ragas_eval

    # Run with custom dataset
    python -m hybridrag.evaluation.ragas_eval --dataset my_test.json

    # Programmatic usage
    from hybridrag.evaluation import RAGEvaluator
    evaluator = RAGEvaluator(rag_instance=my_rag)
    results = await evaluator.run()

Environment Variables:
    EVAL_LLM_MODEL: LLM model for RAGAS evaluation (default: gpt-4o-mini)
    EVAL_EMBEDDING_MODEL: Embedding model for RAGAS (default: text-embedding-3-small)
    EVAL_LLM_BINDING_API_KEY: API key for evaluation LLM (fallback: OPENAI_API_KEY)
    EVAL_MAX_CONCURRENT: Max concurrent evaluations (default: 2)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

# Add project paths
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

load_dotenv(override=False)

logger = logging.getLogger("hybridrag.evaluation")
logger.setLevel(logging.INFO)

# Add console handler if not present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

# Suppress RAGAS deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Unexpected type for token usage.*", category=UserWarning
)

# Check RAGAS availability
RAGAS_AVAILABLE = False
try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"RAGAS not available: {e}. Install with: pip install ragas datasets langchain-openai"
    )
    Dataset = None
    evaluate = None

if TYPE_CHECKING:
    from hybridrag import HybridRAG


def _is_nan(value: Any) -> bool:
    """Check if value is NaN."""
    return isinstance(value, float) and math.isnan(value)


class RAGEvaluator:
    """
    RAGAS-based evaluator for HybridRAG.

    Measures quality using four metrics:
    - Faithfulness: Factual accuracy based on context
    - Answer Relevance: Relevance to user's question
    - Context Recall: Coverage of relevant information
    - Context Precision: Cleanliness of retrieved context
    """

    def __init__(
        self,
        rag_instance: HybridRAG | None = None,
        test_dataset_path: str | Path | None = None,
        query_mode: str = "mix",
    ):
        """
        Initialize the evaluator.

        Args:
            rag_instance: HybridRAG instance to evaluate (created if not provided)
            test_dataset_path: Path to test dataset JSON file
            query_mode: Query mode to use (naive, local, global, hybrid, mix)

        Raises:
            ImportError: If RAGAS dependencies not installed
            EnvironmentError: If required API keys not set
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS dependencies not installed. "
                "Install with: pip install ragas datasets langchain-openai"
            )

        # Configure evaluation LLM (for RAGAS scoring - needs OpenAI-compatible)
        eval_api_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        if not eval_api_key:
            raise OSError(
                "EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY required for RAGAS evaluation. "
                "RAGAS uses OpenAI-compatible models for scoring."
            )

        eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
        eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")
        eval_embedding_model = os.getenv(
            "EVAL_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # Create LLM for RAGAS
        llm_kwargs = {
            "model": eval_model,
            "api_key": eval_api_key,
            "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
            "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "180")),
        }
        if eval_llm_base_url:
            llm_kwargs["base_url"] = eval_llm_base_url

        base_llm = ChatOpenAI(**llm_kwargs)
        self.eval_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
        self.eval_embeddings = OpenAIEmbeddings(
            model=eval_embedding_model, api_key=eval_api_key
        )

        # Store RAG instance
        self.rag = rag_instance
        self.query_mode = query_mode

        # Set up paths
        if test_dataset_path is None:
            test_dataset_path = Path(__file__).parent / "sample_dataset.json"
        self.test_dataset_path = Path(test_dataset_path)
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Load test cases
        self.test_cases = self._load_test_dataset()

        # Store config for display
        self.eval_model = eval_model
        self.eval_embedding_model = eval_embedding_model

        logger.info("=" * 70)
        logger.info("HybridRAG RAGAS Evaluator Initialized")
        logger.info("=" * 70)
        logger.info(f"  Eval LLM Model:      {eval_model}")
        logger.info(f"  Eval Embedding:      {eval_embedding_model}")
        logger.info(f"  Query Mode:          {query_mode}")
        logger.info(f"  Test Cases:          {len(self.test_cases)}")
        logger.info(f"  Dataset:             {self.test_dataset_path.name}")
        logger.info("=" * 70)

    def _load_test_dataset(self) -> list[dict[str, str]]:
        """Load test cases from JSON file."""
        if not self.test_dataset_path.exists():
            logger.warning(f"Test dataset not found: {self.test_dataset_path}")
            logger.info("Creating sample dataset...")
            self._create_sample_dataset()

        with open(self.test_dataset_path) as f:
            data = json.load(f)

        return data.get("test_cases", [])

    def _create_sample_dataset(self) -> None:
        """Create a sample test dataset."""
        sample = {
            "test_cases": [
                {
                    "question": "What is the main purpose of the document?",
                    "ground_truth": "The document describes the main concepts and purpose.",
                    "project": "sample",
                },
                {
                    "question": "What are the key features mentioned?",
                    "ground_truth": "The key features include various capabilities and functions.",
                    "project": "sample",
                },
                {
                    "question": "How does the system work?",
                    "ground_truth": "The system works by processing and analyzing information.",
                    "project": "sample",
                },
            ]
        }

        with open(self.test_dataset_path, "w") as f:
            json.dump(sample, f, indent=2)

        logger.info(f"Created sample dataset: {self.test_dataset_path}")

    async def _ensure_rag(self) -> HybridRAG:
        """Ensure RAG instance is available."""
        if self.rag is None:
            from hybridrag import Settings, create_hybridrag

            logger.info("Creating HybridRAG instance...")
            settings = Settings()
            self.rag = await create_hybridrag(settings=settings, auto_initialize=True)

        return self.rag

    async def generate_rag_response(self, question: str) -> dict[str, Any]:
        """
        Generate RAG response for a question.

        Args:
            question: The user's question

        Returns:
            Dictionary with 'answer' and 'contexts' keys
        """
        rag = await self._ensure_rag()

        # Use query_with_sources to get both answer and context
        result = await rag.query_with_sources(query=question, mode=self.query_mode)

        answer = result.get("answer", "")
        context = result.get("context", "")

        # Split context into chunks (simple split by double newline)
        contexts = [c.strip() for c in context.split("\n\n") if c.strip()]
        if not contexts and context:
            contexts = [context]

        return {"answer": answer, "contexts": contexts}

    async def evaluate_single_case(
        self,
        idx: int,
        test_case: dict[str, str],
        semaphore: asyncio.Semaphore,
        pbar: Any = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            idx: Test case index
            test_case: Test case with question and ground_truth
            semaphore: Concurrency control
            pbar: Progress bar (optional)

        Returns:
            Evaluation result dictionary
        """
        async with semaphore:
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            logger.info(f"[{idx}] Evaluating: {question[:50]}...")

            try:
                # Generate RAG response
                rag_response = await self.generate_rag_response(question)
                retrieved_contexts = rag_response["contexts"]

                if not retrieved_contexts:
                    retrieved_contexts = ["No context retrieved"]

                # Prepare RAGAS dataset
                eval_dataset = Dataset.from_dict(
                    {
                        "question": [question],
                        "answer": [rag_response["answer"]],
                        "contexts": [retrieved_contexts],
                        "ground_truth": [ground_truth],
                    }
                )

                # Run RAGAS evaluation
                eval_results = evaluate(
                    dataset=eval_dataset,
                    metrics=[
                        Faithfulness(),
                        AnswerRelevancy(),
                        ContextRecall(),
                        ContextPrecision(),
                    ],
                    llm=self.eval_llm,
                    embeddings=self.eval_embeddings,
                    _pbar=pbar,
                )

                # Extract scores
                df = eval_results.to_pandas()
                scores = df.iloc[0]

                result = {
                    "test_number": idx,
                    "question": question,
                    "answer": rag_response["answer"][:200] + "..."
                    if len(rag_response["answer"]) > 200
                    else rag_response["answer"],
                    "ground_truth": ground_truth[:200] + "..."
                    if len(ground_truth) > 200
                    else ground_truth,
                    "project": test_case.get("project", "unknown"),
                    "metrics": {
                        "faithfulness": float(scores.get("faithfulness", 0)),
                        "answer_relevance": float(scores.get("answer_relevancy", 0)),
                        "context_recall": float(scores.get("context_recall", 0)),
                        "context_precision": float(scores.get("context_precision", 0)),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                # Calculate RAGAS score (average of valid metrics)
                metrics = result["metrics"]
                valid = [v for v in metrics.values() if not _is_nan(v)]
                result["ragas_score"] = (
                    round(sum(valid) / len(valid), 4) if valid else 0
                )

                logger.info(f"[{idx}] RAGAS Score: {result['ragas_score']:.4f}")

                return result

            except Exception as e:
                logger.error(f"[{idx}] Error: {e}")
                return {
                    "test_number": idx,
                    "question": question,
                    "error": str(e),
                    "metrics": {},
                    "ragas_score": 0,
                    "timestamp": datetime.now().isoformat(),
                }

    async def evaluate_all(self) -> list[dict[str, Any]]:
        """Evaluate all test cases."""
        max_concurrent = int(os.getenv("EVAL_MAX_CONCURRENT", "2"))
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"Starting evaluation of {len(self.test_cases)} test cases...")
        logger.info(f"Max concurrent: {max_concurrent}")

        tasks = [
            self.evaluate_single_case(idx, tc, semaphore)
            for idx, tc in enumerate(self.test_cases, 1)
        ]

        results = await asyncio.gather(*tasks)
        return list(results)

    def _export_csv(self, results: list[dict[str, Any]]) -> Path:
        """Export results to CSV."""
        csv_path = (
            self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "test_number",
                "question",
                "project",
                "faithfulness",
                "answer_relevance",
                "context_recall",
                "context_precision",
                "ragas_score",
                "status",
                "timestamp",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                metrics = result.get("metrics", {})
                writer.writerow(
                    {
                        "test_number": result.get("test_number", 0),
                        "question": result.get("question", ""),
                        "project": result.get("project", "unknown"),
                        "faithfulness": f"{metrics.get('faithfulness', 0):.4f}",
                        "answer_relevance": f"{metrics.get('answer_relevance', 0):.4f}",
                        "context_recall": f"{metrics.get('context_recall', 0):.4f}",
                        "context_precision": f"{metrics.get('context_precision', 0):.4f}",
                        "ragas_score": f"{result.get('ragas_score', 0):.4f}",
                        "status": "success" if metrics else "error",
                        "timestamp": result.get("timestamp", ""),
                    }
                )

        return csv_path

    def _display_results(self, results: list[dict[str, Any]]) -> None:
        """Display results table."""
        logger.info("")
        logger.info("=" * 110)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 110)
        logger.info(
            f"{'#':<4} | {'Question':<45} | {'Faith':>6} | {'AnsRel':>6} | "
            f"{'CtxRec':>6} | {'CtxPre':>6} | {'RAGAS':>6} | Status"
        )
        logger.info("-" * 110)

        for result in results:
            idx = result.get("test_number", 0)
            question = result.get("question", "")[:42]
            if len(result.get("question", "")) > 42:
                question += "..."

            metrics = result.get("metrics", {})
            if metrics:
                logger.info(
                    f"{idx:<4} | {question:<45} | "
                    f"{metrics.get('faithfulness', 0):>6.4f} | "
                    f"{metrics.get('answer_relevance', 0):>6.4f} | "
                    f"{metrics.get('context_recall', 0):>6.4f} | "
                    f"{metrics.get('context_precision', 0):>6.4f} | "
                    f"{result.get('ragas_score', 0):>6.4f} | OK"
                )
            else:
                error = result.get("error", "Unknown")[:20]
                logger.info(
                    f"{idx:<4} | {question:<45} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {error}"
                )

        logger.info("=" * 110)

    def _calculate_stats(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate benchmark statistics."""
        valid = [r for r in results if r.get("metrics")]
        total = len(results)
        successful = len(valid)

        if not valid:
            return {
                "total_tests": total,
                "successful_tests": 0,
                "failed_tests": total,
                "success_rate": 0.0,
            }

        # Calculate averages
        metrics_sums = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "ragas_score": 0.0,
        }
        metrics_counts = dict.fromkeys(metrics_sums, 0)

        for result in valid:
            m = result.get("metrics", {})
            for key in [
                "faithfulness",
                "answer_relevance",
                "context_recall",
                "context_precision",
            ]:
                val = m.get(key, 0)
                if not _is_nan(val):
                    metrics_sums[key] += val
                    metrics_counts[key] += 1

            ragas = result.get("ragas_score", 0)
            if not _is_nan(ragas):
                metrics_sums["ragas_score"] += ragas
                metrics_counts["ragas_score"] += 1

        averages = {}
        for key, total_sum in metrics_sums.items():
            count = metrics_counts[key]
            averages[key] = round(total_sum / count, 4) if count > 0 else 0.0

        ragas_scores = [
            r.get("ragas_score", 0)
            for r in valid
            if not _is_nan(r.get("ragas_score", 0))
        ]

        return {
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": total - successful,
            "success_rate": round(successful / total * 100, 2),
            "average_metrics": averages,
            "min_ragas_score": round(min(ragas_scores), 4) if ragas_scores else 0,
            "max_ragas_score": round(max(ragas_scores), 4) if ragas_scores else 0,
        }

    async def run(self) -> dict[str, Any]:
        """Run complete evaluation pipeline."""
        start_time = time.time()

        results = await self.evaluate_all()
        elapsed = time.time() - start_time

        stats = self._calculate_stats(results)

        # Display results
        self._display_results(results)

        # Save JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "query_mode": self.query_mode,
            "benchmark_stats": stats,
            "results": results,
        }

        json_path = (
            self.results_dir
            / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save CSV
        csv_path = self._export_csv(results)

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total Tests:     {len(results)}")
        logger.info(f"Successful:      {stats['successful_tests']}")
        logger.info(f"Failed:          {stats['failed_tests']}")
        logger.info(f"Success Rate:    {stats['success_rate']:.2f}%")
        logger.info(f"Elapsed Time:    {elapsed:.2f}s")

        if stats.get("average_metrics"):
            avg = stats["average_metrics"]
            logger.info("")
            logger.info("=" * 70)
            logger.info("BENCHMARK RESULTS (Averages)")
            logger.info("=" * 70)
            logger.info(f"Faithfulness:        {avg.get('faithfulness', 0):.4f}")
            logger.info(f"Answer Relevance:    {avg.get('answer_relevance', 0):.4f}")
            logger.info(f"Context Recall:      {avg.get('context_recall', 0):.4f}")
            logger.info(f"Context Precision:   {avg.get('context_precision', 0):.4f}")
            logger.info(f"Average RAGAS:       {avg.get('ragas_score', 0):.4f}")
            logger.info("-" * 70)
            logger.info(f"Min RAGAS Score:     {stats.get('min_ragas_score', 0):.4f}")
            logger.info(f"Max RAGAS Score:     {stats.get('max_ragas_score', 0):.4f}")

        logger.info("")
        logger.info("=" * 70)
        logger.info("OUTPUT FILES")
        logger.info("=" * 70)
        logger.info(f"Results Dir:  {self.results_dir}")
        logger.info(f"  CSV:        {csv_path.name}")
        logger.info(f"  JSON:       {json_path.name}")
        logger.info("=" * 70)

        return summary


async def run_evaluation(
    dataset_path: str | None = None,
    query_mode: str = "mix",
) -> dict[str, Any]:
    """
    Convenience function to run evaluation.

    Args:
        dataset_path: Path to test dataset (optional)
        query_mode: Query mode to use

    Returns:
        Evaluation summary
    """
    evaluator = RAGEvaluator(test_dataset_path=dataset_path, query_mode=query_mode)
    return await evaluator.run()


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for HybridRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to test dataset JSON file",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="mix",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Query mode to use (default: mix)",
    )

    args = parser.parse_args()

    try:
        await run_evaluation(dataset_path=args.dataset, query_mode=args.mode)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
