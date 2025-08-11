import argparse
import os
import sys
from typing import List, Tuple
import importlib

# Ensure local imports work when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from retrievers.base import Retriever


def load_keywords(database_path: str) -> List[str]:
    path_lower = database_path.lower()
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file not found: {database_path}")

    keywords: List[str] = []
    if path_lower.endswith('.txt'):
        with open(database_path, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term:
                    keywords.append(term)
    elif path_lower.endswith('.csv'):
        import csv
        with open(database_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'keyword' in reader.fieldnames:
                for row in reader:
                    term = (row.get('keyword') or '').strip()
                    if term:
                        keywords.append(term)
            else:
                f.seek(0)
                reader2 = csv.reader(f)
                for row in reader2:
                    if not row:
                        continue
                    term = (row[0] or '').strip()
                    if term:
                        keywords.append(term)
    elif path_lower.endswith('.jsonl') or path_lower.endswith('.ndjson'):
        import json
        with open(database_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                term = (obj.get('keyword') or '').strip()
                if term:
                    keywords.append(term)
    else:
        raise ValueError("Unsupported database file format. Use .txt, .csv, or .jsonl")

    if not keywords:
        raise ValueError("Loaded 0 keywords from database. Please check the file.")
    return keywords


def get_retriever(name: str, keywords: List[str], **kwargs) -> Retriever:
    name = (name or '').lower()
    if name in ('sw', 'smith_waterman'):
        module = importlib.import_module('retrievers.smith_waterman_retriever')
        RetrieverClass = getattr(module, 'SmithWatermanRetriever')
        return RetrieverClass(keywords)
    if name in ('bm25',):
        try:
            module = importlib.import_module('retrievers.bm25_retriever')
        except Exception as exc:
            raise RuntimeError("BM25Retriever not available. Please install rank-bm25.") from exc
        RetrieverClass = getattr(module, 'BM25Retriever')
        return RetrieverClass(keywords)
    if name in ('embed', 'embedding', 'embeddings'):
        try:
            module = importlib.import_module('retrievers.embedding_retriever')
        except Exception as exc:
            raise RuntimeError("EmbeddingRetriever not available. Please install sentence-transformers, numpy, torch.") from exc
        RetrieverClass = getattr(module, 'EmbeddingRetriever')
        model_name = kwargs.get('embedding_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
        return RetrieverClass(keywords, model_name=model_name)
    raise ValueError(f"Unknown retriever: {name}")


def get_llm_client(provider: str, model: str, **kwargs):
    provider = (provider or '').lower()
    if provider in ('openai', 'openai_compat'):
        api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key and provider == 'openai':
            raise RuntimeError("OPENAI_API_KEY not provided. Set env or --api-key.")
        base_url = kwargs.get('base_url')
        module = importlib.import_module('llm.openai_client')
        ClientClass = getattr(module, 'OpenAIClient')
        return ClientClass(model=model, api_key=api_key, base_url=base_url)
    if provider in ('ollama',):
        base_url = kwargs.get('base_url') or 'http://localhost:11434'
        module = importlib.import_module('llm.ollama_client')
        ClientClass = getattr(module, 'OllamaClient')
        return ClientClass(model=model, base_url=base_url)
    raise ValueError(f"Unknown LLM provider: {provider}")


def parse_args():
    parser = argparse.ArgumentParser(description='RAG-ASR: refine ASR output using keyword retrieval and LLM.')
    parser.add_argument('--query', type=str, required=True, help='ASR candidate text')
    parser.add_argument('--database', type=str, required=False, default=os.path.join(SCRIPT_DIR, 'data', 'keywords.txt'), help='Path to keywords database (.txt/.csv/.jsonl)')
    parser.add_argument('--retriever', type=str, default='smith_waterman', help='Retrieval method: smith_waterman|bm25|embedding')
    parser.add_argument('--top-k', type=int, default=5, help='Top K keywords to include in prompt')

    parser.add_argument('--llm-provider', type=str, default='openai', help='LLM provider: openai|openai_compat|ollama')
    parser.add_argument('--llm-model', type=str, default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--api-key', type=str, default=None, help='API key for OpenAI or compatible providers')
    parser.add_argument('--base-url', type=str, default=None, help='Base URL for OpenAI-compatible or Ollama')
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/paraphrase-MiniLM-L6-v2', help='Embedding model for embedding retriever')

    parser.add_argument('--max-keywords-in-prompt', type=int, default=None, help='Optional cap for number of keywords inserted into prompt')
    parser.add_argument('--show-scores', action='store_true', help='Print retrieved keywords with scores')
    parser.add_argument('--dry-run', action='store_true', help='Only show retrieved keywords and constructed prompt, do not call LLM')
    return parser.parse_args()


def main():
    from prompting.prompt_builder import build_prompt

    args = parse_args()

    keywords = load_keywords(args.database)

    retriever_kwargs = {
        'embedding_model': args.embedding_model,
    }
    retriever = get_retriever(args.retriever, keywords, **retriever_kwargs)

    results: List[Tuple[str, float]] = retriever.search(args.query, top_k=args.top_k)

    if args.max_keywords_in_prompt is not None:
        results = results[: args.max_keywords_in_prompt]

    if args.show_scores:
        print('Top retrieved keywords:')
        for kw, score in results:
            print(f"  {kw}\t{score:.4f}")
        print('-' * 40)

    keyword_list = [kw for kw, _ in results]
    prompt = build_prompt(query=args.query, keywords=keyword_list)

    if args.dry_run:
        print('----- Prompt Preview -----')
        print(prompt)
        return

    client = get_llm_client(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    output_text = client.generate(prompt)

    print(output_text)


if __name__ == '__main__':
    main()
