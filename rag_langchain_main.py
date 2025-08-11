import argparse
import os
import sys
from typing import List, Tuple
import importlib

# Ensure local imports work when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from retrievers.base import Retriever as CustomRetriever
from rag_asr_main import load_keywords

# LangChain components
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# LLM implementations
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Embedding and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


class SmithWatermanLangChainRetriever(BaseRetriever):
    """
    A custom LangChain retriever that wraps our existing Smith-Waterman implementation.
    """
    custom_retriever: CustomRetriever
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        results = self.custom_retriever.search(query, top_k=self.k)
        return [Document(page_content=res[0], metadata={"score": res[1]}) for res in results]


def get_retriever(name: str, keywords: List[str], top_k: int, **kwargs) -> BaseRetriever:
    """
    Factory function to get the specified LangChain retriever.
    """
    name = (name or '').lower()
    
    if name in ('sw', 'smith_waterman'):
        # This retriever doesn't need documents, just the raw keyword list
        module = importlib.import_module('retrievers.smith_waterman_retriever')
        RetrieverClass = getattr(module, 'SmithWatermanRetriever')
        custom_retriever = RetrieverClass(keywords)
        return SmithWatermanLangChainRetriever(custom_retriever=custom_retriever, k=top_k)

    if name == 'bm25':
        docs = [Document(page_content=kw) for kw in keywords]
        try:
            return BM25Retriever.from_documents(docs, k=top_k)
        except Exception as exc:
            raise RuntimeError("BM25Retriever not available. Install with: pip install rank-bm25") from exc

    if name in ('embed', 'embedding', 'embeddings'):
        try:
            persist_directory = kwargs.get('persist_directory')
            if not persist_directory:
                raise ValueError("For embedding retriever, --persist-directory must be provided.")

            embedding_model_name = kwargs.get('embedding_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                encode_kwargs={'normalize_embeddings': True}
            )

            # Chroma will create the directory if it doesn't exist on first write
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

            # Check if the DB is empty. If so, load from source.
            if vectorstore._collection.count() == 0:
                print(f"Chroma DB at '{persist_directory}' is empty. Building from source...")
                docs = [Document(page_content=kw) for kw in keywords]
                if not docs:
                    raise ValueError("Keyword list is empty. Cannot build Chroma DB.")
                print(f"Adding {len(docs)} documents to Chroma DB. This may take a while...")
                vectorstore.add_documents(docs)
                vectorstore.persist()
                print("Chroma DB built and persisted.")
            else:
                print(f"Loaded existing Chroma DB from '{persist_directory}' with {vectorstore._collection.count()} documents.")

            return vectorstore.as_retriever(search_kwargs={"k": top_k})
        except Exception as exc:
            raise RuntimeError(f"Chroma retriever failed. Ensure dependencies are installed. Error: {exc}")

    raise ValueError(f"Unknown retriever: {name}")


def get_llm(provider: str, model: str, **kwargs):
    """
    Factory function for LangChain LLM clients.
    """
    provider = (provider or '').lower()
    api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
    base_url = kwargs.get('base_url')

    if provider in ('openai', 'vllm'):
        # vLLM is compatible with OpenAI's client
        client_kwargs = {"model": model, "api_key": api_key, "base_url": base_url, "temperature": 0.2}
        return ChatOpenAI(**{k: v for k, v in client_kwargs.items() if v is not None})
    
    if provider == 'ollama':
        client_kwargs = {"model": model, "base_url": base_url, "temperature": 0.2}
        return ChatOllama(**{k: v for k, v in client_kwargs.items() if v is not None})
        
    raise ValueError(f"Unknown or unsupported LLM provider for LangChain: {provider}")


def format_docs(docs: List[Document]) -> str:
    """Format documents for the prompt."""
    return "\n".join([d.page_content for d in docs])

def main():
    args = parse_args()

    # 1. Load keywords from the source file.
    # This is always needed now, either for direct use or for Chroma's first build.
    keywords = load_keywords(args.database)

    # 2. Setup retriever
    retriever_kwargs = {
        'embedding_model': args.embedding_model,
        'persist_directory': args.persist_directory
    }
    retriever = get_retriever(args.retriever, keywords, args.top_k, **retriever_kwargs)

    # 3. Setup LLM
    llm_kwargs = {'api_key': args.api_key, 'base_url': args.base_url}
    llm = get_llm(args.llm_provider, args.llm_model, **llm_kwargs)
    
    # 4. Define Prompt
    template = """You are an assistant for correcting and polishing text, especially for Automatic Speech Recognition (ASR) outputs.
Based on the user's raw ASR query and a list of potentially relevant keywords, please generate a corrected and natural-sounding version of the text.

Here are the relevant keywords:
{context}

User's ASR query:
{query}

Your corrected and polished version:"""
    prompt = PromptTemplate.from_template(template)

    # 5. Build RAG Chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Run the chain
    query = args.query
    result = rag_chain.invoke(query)

    # Optional: show retrieved documents
    if args.show_retrieved:
        retrieved_docs = retriever.invoke(query)
        print("\n--- Retrieved Keywords ---")
        for doc in retrieved_docs:
            score_info = f" (Score: {doc.metadata['score']:.4f})" if 'score' in doc.metadata else ""
            print(f"- {doc.page_content}{score_info}")
        print("--------------------------\n")

    print("\n--- LLM Output ---")
    print(result)
    print("------------------")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='(LangChain) RAG-ASR: refine ASR output using keyword retrieval and LLM.')
    # Common arguments from original script
    parser.add_argument('--query', type=str, required=True, help='ASR candidate text')
    parser.add_argument('--database', type=str, required=True, default=os.path.join(SCRIPT_DIR, 'data', 'keywords.txt'), help='Path to source keywords database (.txt/.csv/.jsonl)')
    parser.add_argument('--retriever', type=str, default='smith_waterman', help='Retrieval method: smith_waterman|bm25|embedding')
    parser.add_argument('--top-k', type=int, default=5, help='Top K keywords to retrieve')

    # LLM arguments
    parser.add_argument('--llm-provider', type=str, default='openai', help='LLM provider: openai|ollama|vllm')
    parser.add_argument('--llm-model', type=str, default='gpt-4o-mini', help='LLM model name')
    parser.add_argument('--api-key', type=str, default=None, help='API key for OpenAI or compatible providers')
    parser.add_argument('--base-url', type=str, default=None, help='Base URL for OpenAI-compatible, vLLM or Ollama')

    # Embedding model & Chroma DB path
    parser.add_argument('--embedding-model', type=str, default='sentence-transformers/paraphrase-MiniLM-L6-v2', help='Embedding model for embedding retriever')
    parser.add_argument('--persist-directory', type=str, default=os.path.join(SCRIPT_DIR, 'vector_stores', 'chroma_db'), help='Directory to save/load the Chroma vector store')
    
    # Control flags
    parser.add_argument('--show-retrieved', action='store_true', help='Print retrieved keywords')
    
    return parser.parse_args()

if __name__ == '__main__':
    main() 