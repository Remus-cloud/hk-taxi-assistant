"""Current-corpus retrieval with the approved lexical and ranking algorithms."""
from pathlib import Path
import json
import numpy as np
from rank_bm25 import BM25Okapi
from .retrieval import Retriever, LexicalTokenizer, read_jsonl
from .runtime_assets import verify_inputs, verify_asset, verify_model


class RuntimeRetriever(Retriever):
    def __init__(self, root):
        self.root = Path(root)
        verify_inputs(self.root)
        verify_model(self.root, 'embedding')
        self.config = json.loads((self.root/'configs/retrieval_v1.json').read_text(encoding='utf-8'))
        index = self.root/'artifacts/retrieval_index'
        for filename in ['manifest.json', 'chunks.jsonl', 'embeddings.npy']:
            verify_asset(self.root, 'artifacts/retrieval_index/'+filename)
        self.manifest = json.loads((index/'manifest.json').read_text(encoding='utf-8'))
        self.chunks = read_jsonl(index/'chunks.jsonl')
        self.vectors = np.load(index/'embeddings.npy', allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), self.manifest['dimension']):
            raise ValueError('Index dimensions do not match the corpus')
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid or unnormalized corpus vectors')
        ids = [c['chunk_id'] for c in self.chunks]
        if len(set(ids)) != len(ids) or not all(c.get('default_retrievable') is True for c in self.chunks):
            raise ValueError('Duplicate or unapproved corpus item')
        self.tokenize = LexicalTokenizer()
        self.bm25 = BM25Okapi([self.tokenize(c['text']) for c in self.chunks],
                             k1=self.config['bm25_k1'], b=self.config['bm25_b'])
        self.encoder = None
