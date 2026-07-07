"""Parent-document retriever: index children, return parent context."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore

from src.chunking.structure import recursive_parent_child, structure_aware_parent_child
from src.config import SETTINGS


class HierarchicalRetriever:
    def __init__(self, embeddings, use_structure_chunking: bool | None = None):
        self.embeddings = embeddings
        self.use_structure = (
            use_structure_chunking
            if use_structure_chunking is not None
            else SETTINGS["features"]["structure_chunking"]
        )
        cfg = SETTINGS["chunking"]
        self.child_chunk_size = cfg["child_chunk_size"]
        self.child_overlap = cfg["child_overlap"]
        self.parent_min = cfg["parent_min_chars"]
        self.parent_max = cfg["parent_max_chars"]

        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.ns = ("parents",)
        self.all_children: list[Document] = []
        self.chunking_mode = "structure" if self.use_structure else "recursive"

    def _build_parent_child(self, documents: list[Document]) -> tuple[list[Document], list[Document]]:
        if self.use_structure:
            return structure_aware_parent_child(
                documents,
                child_chunk_size=self.child_chunk_size,
                child_overlap=self.child_overlap,
                parent_min_chars=self.parent_min,
                parent_max_chars=self.parent_max,
            )
        return recursive_parent_child(
            documents,
            child_chunk_size=self.child_chunk_size,
            child_overlap=self.child_overlap,
        )

    def add_documents(self, documents: list[Document], persist_dir: Path, docstore_path: Path, chunks_path: Path):
        parents, self.all_children = self._build_parent_child(documents)

        for i, parent in enumerate(parents):
            parent_id = f"parent_{i}"
            self.docstore.put(
                self.ns,
                parent_id,
                {"page_content": parent.page_content, "metadata": parent.metadata},
            )

        self.vectorstore = Chroma.from_documents(
            self.all_children,
            embedding=self.embeddings,
            persist_directory=str(persist_dir),
            collection_name="hierarchical_children",
        )
        self._persist_docstore(docstore_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.all_children, f)
        print(
            f"Indexed [{self.chunking_mode}] {len(parents)} parents, "
            f"{len(self.all_children)} children."
        )

    def _persist_docstore(self, docstore_path: Path):
        items = self.docstore.search(self.ns)
        data = {
            item.key: {
                "page_content": item.value["page_content"],
                "metadata": item.value["metadata"],
            }
            for item in items
        }
        with open(docstore_path, "wb") as f:
            pickle.dump({"data": data, "chunking_mode": self.chunking_mode}, f)

    def _load_docstore(self, docstore_path: Path, chunks_path: Path):
        if docstore_path.exists():
            with open(docstore_path, "rb") as f:
                stored = pickle.load(f)
            data = stored["data"] if isinstance(stored, dict) and "data" in stored else stored
            for k, v in data.items():
                self.docstore.put(self.ns, k, v)
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self.all_children = pickle.load(f)

    def get_child_matches(self, query: str, k: int | None = None) -> list[Document]:
        """Dense retrieval at child granularity (for eval and reranker input)."""
        if self.vectorstore is None:
            return []
        k = k or SETTINGS["retrieval"]["hybrid_k"]
        mult = SETTINGS["retrieval"]["vector_k_multiplier"]
        return self.vectorstore.similarity_search(query, k=k * mult)

    def get_relevant_documents(self, query: str, k: int | None = None) -> list[Document]:
        """Child search mapped to parent sections (small-to-big for generation)."""
        if self.vectorstore is None:
            return []
        k = k or SETTINGS["retrieval"]["parent_k"]
        results = self.get_child_matches(query, k=k)

        parents: list[Document] = []
        seen: set[str] = set()
        for child in results:
            pid = child.metadata.get("parent_id")
            if not pid or pid in seen:
                continue
            item = self.docstore.get(self.ns, pid)
            if item and item.value:
                parents.append(
                    Document(
                        page_content=item.value["page_content"],
                        metadata=item.value["metadata"],
                    )
                )
            seen.add(pid)
        return parents[:k]


def get_retriever(
    embeddings,
    documents: list[Document],
    persist_dir: Path,
    docstore_path: Path,
    chunks_path: Path,
    force_rebuild: bool = False,
    use_structure_chunking: bool | None = None,
) -> HierarchicalRetriever:
    retriever = HierarchicalRetriever(embeddings, use_structure_chunking=use_structure_chunking)
    index_ready = (
        persist_dir.exists()
        and docstore_path.exists()
        and chunks_path.exists()
        and not force_rebuild
    )
    if index_ready:
        retriever.vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="hierarchical_children",
        )
        retriever._load_docstore(docstore_path, chunks_path)
        print(f"Loaded index ({retriever.chunking_mode} mode).")
    else:
        retriever.add_documents(documents, persist_dir, docstore_path, chunks_path)
    return retriever
