from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import faiss
from sentence_transformers import SentenceTransformer
import torch

from .base import Agent
from .model import HFModel, GenerationConfig
from .types import AgentResult, TaskInstance


@dataclass
class RAGConfig:
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_gpu: bool = True
    cache_dir: Optional[str] = None


class RAGAgent(Agent):
    def __init__(
        self,
        model: HFModel,
        corpus: List[str],
        rag_config: Optional[RAGConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(name="rag")
        self.model = model
        self.corpus = corpus
        self.rag_config = rag_config or RAGConfig()
        self.gen_config = gen_config
        device = "cuda" if torch.cuda.is_available() and self.rag_config.use_gpu else "cpu"
        print(f"Loading embedder: {self.rag_config.embedding_model} (device={device})")
        self.embedder = SentenceTransformer(
            self.rag_config.embedding_model,
            device=device,
            cache_folder=self.rag_config.cache_dir,
        )
        self.index = self._build_index(corpus)

    def _build_index(self, corpus: List[str]) -> faiss.IndexFlatIP:
        print(f"Building RAG index for {len(corpus)} passages")
        embeddings = self.embedder.encode(
            corpus,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        if self.rag_config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def _retrieve(self, query: str) -> List[str]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, self.rag_config.top_k)
        return [self.corpus[i] for i in indices[0] if i >= 0]

    def run(self, instance: TaskInstance) -> AgentResult:
        passages = self._retrieve(instance.input)
        context = "\n\n".join(passages)
        prompt = f"Context:\n{context}\n\nQuestion:\n{instance.input}\n\nAnswer:"
        result = self.model.generate(prompt, self.gen_config)
        return AgentResult(
            text=result["text"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"],
            metadata={"agent": self.name, "top_k": self.rag_config.top_k},
        )
