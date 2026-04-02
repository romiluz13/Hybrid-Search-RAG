"""
Example 08: Evaluation with RAGAS

Demonstrates:
- RAG system evaluation using RAGAS framework
- Answer relevancy, faithfulness, and context precision
- Comparing different search modes
- Generating evaluation reports

Prerequisites:
- MONGODB_URI in .env
- VOYAGE_API_KEY in .env
- OPENAI_API_KEY in .env (RAGAS uses OpenAI for evaluation)
- ragas installed: pip install ragas datasets langchain-openai

References:
- RAGAS: https://docs.ragas.io/
"""

import asyncio
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ["MONGODB_URI", "VOYAGE_API_KEY", "OPENAI_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(
            f"{var} not set in environment. RAGAS requires OpenAI API key."
        )


async def setup_test_data():
    """Insert sample documents for evaluation."""
    from hybridrag import create_hybridrag

    rag = await create_hybridrag()

    documents = [
        "MongoDB Atlas is a fully managed cloud database service that handles deployment, scaling, and maintenance.",
        "Vector search in MongoDB Atlas enables semantic similarity searches using machine learning embeddings.",
        "Atlas Search provides full-text search capabilities with features like fuzzy matching and autocomplete.",
        "MongoDB supports ACID transactions for multi-document operations.",
        "Atlas clusters can be deployed across multiple cloud providers including AWS, Azure, and GCP.",
    ]

    print("Setting up test data...")
    for doc in documents:
        await rag.insert(doc)

    print(f"✓ Inserted {len(documents)} documents\n")
    return rag


async def example_basic_evaluation():
    """Basic RAGAS evaluation of a single query."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        print("Error: RAGAS not installed.")
        print("Install with: pip install ragas datasets langchain-openai")
        return

    print("=" * 60)
    print("Example 1: Basic RAGAS Evaluation")
    print("=" * 60)

    rag = await setup_test_data()

    # Test query
    query = "What is MongoDB Atlas?"

    # Get RAG response
    print(f"\nQuery: {query}\n")
    print("Getting RAG response...")

    result = await rag.query_with_sources(query=query, mode="mix", top_k=3)
    answer = result["answer"]
    context = result["context"]

    # Prepare evaluation dataset
    eval_data = {
        "question": [query],
        "answer": [answer],
        # RAGAS expects contexts as list[list[str]] — one list of chunks per question.
        # query_with_sources() returns a single concatenated string; split by paragraph
        # as a best-effort approximation of the original chunks.
        "contexts": [context.split("\n\n") if context else []],
        "ground_truth": ["MongoDB Atlas is a fully managed cloud database service."],
    }

    dataset = Dataset.from_dict(eval_data)

    # Evaluate
    print("Evaluating with RAGAS...")
    print("Metrics: answer_relevancy, faithfulness, context_precision\n")

    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision],
    )

    # Display results
    print("Evaluation Results:")
    print("-" * 40)
    for metric, score in result.items():
        print(f"  {metric}: {score:.4f}")

    print("\nInterpretation:")
    print("  - answer_relevancy: How relevant is the answer to the question?")
    print("  - faithfulness: Is the answer faithful to the retrieved context?")
    print("  - context_precision: Are the retrieved contexts relevant?")


async def example_compare_modes():
    """Compare different search modes using RAGAS."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision
    except ImportError:
        print("Error: RAGAS not installed.")
        return

    print("\n" + "=" * 60)
    print("Example 2: Compare Search Modes")
    print("=" * 60)

    rag = await setup_test_data()

    query = "How does vector search work in MongoDB?"
    ground_truth = (
        "Vector search enables semantic similarity searches using embeddings."
    )

    print(f"\nQuery: {query}\n")

    # naive  = vector-only retrieval (no knowledge graph)
    # hybrid = LightRAG's local+global combined mode
    # mix    = hybrid_rrf + graph traversal + entity boosting (fullest pipeline)
    modes = ["naive", "hybrid", "mix"]
    results_by_mode = {}

    # Evaluate each mode
    for mode in modes:
        print(f"Evaluating {mode} mode...")

        result = await rag.query_with_sources(query=query, mode=mode, top_k=3)
        answer = result["answer"]
        context = result["context"]

        eval_data = {
            "question": [query],
            "answer": [answer],
            "contexts": [context.split("\n\n") if context else []],
            "ground_truth": [ground_truth],
        }

        dataset = Dataset.from_dict(eval_data)

        result = evaluate(
            dataset,
            metrics=[answer_relevancy, context_precision],
        )

        results_by_mode[mode] = result

    # Compare results
    print("\n" + "=" * 60)
    print("Mode Comparison:")
    print("=" * 60 + "\n")

    for mode in modes:
        print(f"{mode.upper()} Mode:")
        for metric, score in results_by_mode[mode].items():
            print(f"  {metric}: {score:.4f}")
        print()

    # Winner
    best_mode = max(
        modes,
        key=lambda m: results_by_mode[m].get("answer_relevancy", 0),
    )
    print(f"Best mode by answer_relevancy: {best_mode.upper()}")


async def example_batch_evaluation():
    """Evaluate multiple queries at once."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("Error: RAGAS not installed.")
        return

    print("\n" + "=" * 60)
    print("Example 3: Batch Evaluation")
    print("=" * 60)

    rag = await setup_test_data()

    # Test set
    test_cases = [
        {
            "question": "What is MongoDB Atlas?",
            "ground_truth": "MongoDB Atlas is a fully managed cloud database service.",
        },
        {
            "question": "What is vector search?",
            "ground_truth": "Vector search enables semantic similarity searches using embeddings.",
        },
        {
            "question": "Does MongoDB support transactions?",
            "ground_truth": "MongoDB supports ACID transactions for multi-document operations.",
        },
    ]

    print(f"\nEvaluating {len(test_cases)} queries...\n")

    # Collect results
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for test in test_cases:
        query = test["question"]
        result = await rag.query_with_sources(query=query, mode="mix", top_k=3)
        answer = result["answer"]
        context = result["context"]

        questions.append(query)
        answers.append(answer)
        # Split concatenated context into paragraph-level chunks for RAGAS
        contexts.append(context.split("\n\n") if context else [])
        ground_truths.append(test["ground_truth"])

    # Create dataset
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(eval_data)

    # Evaluate
    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness],
    )

    # Display results
    print("\nBatch Evaluation Results:")
    print("-" * 40)
    for metric, score in result.items():
        print(f"  {metric}: {score:.4f}")


async def example_custom_metrics():
    """Demonstrate available RAGAS metrics."""
    print("\n" + "=" * 60)
    print("Example 4: Available RAGAS Metrics")
    print("=" * 60)

    try:
        print("\nRAGAS Metrics Overview:\n")

        metrics = [
            ("answer_relevancy", "How relevant is the answer to the question?"),
            ("faithfulness", "Is the answer faithful to the retrieved contexts?"),
            ("context_precision", "Are the retrieved contexts relevant?"),
            ("context_recall", "Did we retrieve all relevant contexts?"),
            ("answer_similarity", "Semantic similarity to ground truth"),
            ("answer_correctness", "Correctness compared to ground truth"),
        ]

        for name, description in metrics:
            print(f"✓ {name}")
            print(f"  {description}\n")

        print("Required fields by metric:")
        print("  - answer_relevancy: question, answer")
        print("  - faithfulness: answer, contexts")
        print("  - context_precision: question, contexts, ground_truth")
        print("  - context_recall: question, contexts, ground_truth")
        print("  - answer_similarity: answer, ground_truth")
        print("  - answer_correctness: answer, ground_truth")

    except ImportError:
        print("RAGAS not installed. Install with: pip install ragas")


async def example_optimization_workflow():
    """Example workflow for iterative optimization."""
    print("\n" + "=" * 60)
    print("Example 5: Optimization Workflow")
    print("=" * 60)

    print("""
Iterative RAG Optimization Workflow:

1. **Baseline Evaluation**
   - Evaluate current system with RAGAS
   - Identify weakest metrics
   - Set target scores

2. **Hypothesis & Change**
   - Low faithfulness? → Improve context retrieval
   - Low relevancy? → Tune search weights
   - Low precision? → Add filters or reranking

3. **Re-evaluate**
   - Run RAGAS again
   - Compare to baseline
   - Verify improvement

4. **Iterate**
   - Repeat until target metrics achieved
   - Track changes in a table

Example results tracking:
""")

    print("""
Configuration            | Relevancy | Faithfulness | Precision
-------------------------|-----------|--------------|----------
Baseline (hybrid 0.5/0.5)|   0.72    |     0.85     |   0.68
Tuned (hybrid 0.6/0.4)   |   0.78    |     0.87     |   0.71
+ Reranking              |   0.82    |     0.89     |   0.76
+ Filters                |   0.85    |     0.91     |   0.81 ✓
""")

    print("\nBest Practices:")
    print("  ✓ Use a fixed test set for consistency")
    print("  ✓ Evaluate multiple metrics together")
    print("  ✓ Compare against ground truth when available")
    print("  ✓ Track configuration changes systematically")


async def main():
    """Run all examples."""
    from hybridrag.core.mongodb_client import close_shared_client

    print("\n" + "=" * 60)
    print("HybridRAG Example 08: Evaluation with RAGAS")
    print("=" * 60)

    try:
        # Check if RAGAS is installed
        try:
            import ragas

            print(f"\nRAGAS version: {ragas.__version__}")
        except ImportError:
            print("\nError: RAGAS not installed.")
            print("Install with: pip install ragas datasets langchain-openai")
            print("\nShowing documentation examples only...\n")

        # Run examples
        await example_basic_evaluation()
        await example_compare_modes()
        await example_batch_evaluation()
        await example_custom_metrics()
        await example_optimization_workflow()

        print("\n" + "=" * 60)
        print("All examples complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  - RAGAS provides objective metrics for RAG quality")
        print("  - Evaluate multiple search modes to find the best")
        print("  - Use batch evaluation for comprehensive testing")
        print("  - Iterate based on metric feedback")
    finally:
        close_shared_client()


if __name__ == "__main__":
    asyncio.run(main())
