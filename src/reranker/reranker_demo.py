import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str):
    """Utility to print and run a shell command."""
    print(f"\n[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True)


def ensure_repo():
    """Clone the repo if it doesn't exist. Then cd into it."""
    repo_name = "RAG-diagnostic"

    if not Path(repo_name).exists():
        run_cmd(f"git clone https://github.com/onichmath/{repo_name}.git")

    os.chdir(repo_name)
    print(f"[INFO] Working directory: {os.getcwd()}")


def install_dependencies():
    """Install runtime dependencies for the local-model reranker + RAG pipeline."""
    run_cmd("pip install transformers accelerate sentencepiece")
    run_cmd(
        "pip install langchain langchain-community langchain-huggingface "
        "datasets pypdf tqdm pandas seaborn matplotlib faiss-cpu"
    )


def run_pipeline_with_local_reranker():
    """
    Run the full pipeline with:
    - small dataset sizes
    - reranker enabled
    - local model (no API keys)
    """
    cmd = (
        "python scripts/build_pipeline.py "
        "--pubmed-docs 200 "
        "--textbook-docs 100 "
        "--max-test-queries 2 "
        "--use-llm-reranker "
        "--llm-model local"
    )
    run_cmd(cmd)


def run_toy_reranker_test():
    """Simple direct test of the reranker in isolation."""
    print("\n[INFO] Running toy reranker test...")

    # ensure root on path
    repo_root = Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.reranker.llm_title_rerank import rerank_by_title_llm
    from langchain_core.documents import Document

    query = "What is the recommended treatment for acute ischemic stroke?"

    docs = [
        Document(page_content="...", metadata={"title": "Heart Failure Management 2021"}),
        Document(page_content="...", metadata={"title": "Stroke Thrombolysis Within 4.5 Hours"}),
        Document(page_content="...", metadata={"title": "Management of AFib in Adults"}),
    ]

    reranked = rerank_by_title_llm(query, docs)

    print("\nOriginal Order:")
    for d in docs:
        print("-", d.metadata["title"])

    print("\nReranked Order:")
    for d in reranked:
        print("-", d.metadata["title"])


def main():
    print("[STEP 1] Clone repo if needed")
    ensure_repo()

    print("\n[STEP 2] Install Python dependencies")
    install_dependencies()

    print("\n[STEP 3] Run pipeline with reranker enabled")
    run_pipeline_with_local_reranker()

    print("\n[STEP 4] Run a small toy test of the reranker")
    run_toy_reranker_test()

    print("\n[DONE] All steps completed successfully.")


if __name__ == "__main__":
    main()
