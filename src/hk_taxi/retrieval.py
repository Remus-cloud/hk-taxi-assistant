"""Exact dense, Chinese lexical BM25, and rank fusion. No answering model."""
from pathlib import Path
from collections import defaultdict
import json, re, hashlib, time
import numpy as np
from opencc import OpenCC
from rank_bm25 import BM25Okapi


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LexicalTokenizer:
    def __init__(self):
        self.converter = OpenCC("t2s")

    def __call__(self, text):
        text = self.converter.convert(text).lower()
        tokens = []
        for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z]+(?:[-'][a-z]+)*|[0-9]+(?:\.[0-9]+)*", text):
            if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", run):
                tokens.extend(run[i:i+2] for i in range(len(run)-1))
                if len(run) == 1:
                    tokens.append(run)
            else:
                tokens.append(run)
        return tokens


def ranked(scores, chunks, limit):
    indices = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))[:limit]
    return [dict(chunk=chunks[i], score=float(scores[i]), rank=n+1) for n, i in enumerate(indices)]


def fuse(dense, lexical, rrf_k=60):
    scores = defaultdict(float)
    docs = {}
    ranks = defaultdict(dict)
    for name, hits in [("dense", dense), ("bm25", lexical)]:
        for hit in hits:
            cid = hit["chunk"]["chunk_id"]
            scores[cid] += 1 / (rrf_k + hit["rank"])
            docs[cid] = hit["chunk"]
            ranks[cid][name] = hit["rank"]
    return [dict(chunk=docs[cid], score=scores[cid], rank=i+1, component_ranks=ranks[cid])
            for i, cid in enumerate(sorted(scores, key=lambda cid: (-scores[cid], cid)))]


class DenseEncoder:
    def __init__(self, local_model):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(local_model), device="cpu", local_files_only=True, trust_remote_code=False)
        self.model.float()
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_length = self.model.max_seq_length
        self.tokenizer = self.model.tokenizer

    def encode(self, texts):
        lengths = [len(self.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) for text in texts]
        if max(lengths, default=0) > self.max_length:
            raise ValueError(f"Input exceeds model limit; silent truncation prohibited: {max(lengths)} > {self.max_length}")
        start = time.perf_counter()
        vectors = self.model.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding contains nonfinite values")
        return vectors, (time.perf_counter()-start)*1000, lengths


class Retriever:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root/"configs/retrieval_v1.json").read_text(encoding="utf-8"))
        index = self.root/"artifacts/retrieval_index_v1"
        self.manifest = json.loads((index/"manifest.json").read_text(encoding="utf-8"))
        corpus = self.root/self.config["corpus"]
        if fingerprint(corpus) != self.manifest["corpus_sha256"]:
            raise ValueError("Corpus changed; rebuild index")
        if fingerprint(self.root/"configs/retrieval_v1.json") != self.manifest["config_sha256"]:
            raise ValueError("Retrieval configuration changed; rebuild index")
        # Exact corpus allowlist: no directory crawl and no evaluation files.
        self.chunks = read_jsonl(index/"chunks.jsonl")
        self.vectors = np.load(index/"embeddings.npy", allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), self.manifest["dimension"]):
            raise ValueError("Index row/dimension mismatch")
        tok = LexicalTokenizer()
        self.tokenize = tok
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks], k1=self.config["bm25_k1"], b=self.config["bm25_b"])
        self.encoder = None

    def encode_queries(self, texts):
        if self.encoder is None:
            self.encoder = DenseEncoder(self.root/self.manifest["local_model"])
        return self.encoder.encode(texts)

    def search(self, query, method="hybrid", top_k=None, vector=None):
        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Unknown retrieval method")
        top_k = self.config["final_top_k"] if top_k is None else top_k
        if not 1 <= top_k <= self.config["candidate_per_retriever"]:
            raise ValueError("top_k must be within configured candidate count")
        encoding_ms = 0
        if method != "bm25" and vector is None:
            v, encoding_ms, _ = self.encode_queries([query]); vector = v[0]
        start = time.perf_counter()
        n = self.config["candidate_per_retriever"]
        dense = ranked(self.vectors @ vector, self.chunks, n) if method != "bm25" else []
        lexical = []
        if method != "dense":
            tokens = self.tokenize(query)
            # Zero-score/negative-score hits do not supply a lexical contribution.
            scores = self.bm25.get_scores(tokens)
            lexical = [h for h in ranked(scores, self.chunks, n) if h["score"] > 0]
            for i, h in enumerate(lexical): h["rank"] = i+1
        hits = dense if method == "dense" else lexical if method == "bm25" else fuse(dense, lexical, self.config["rrf_k"])
        return dict(query=query, method=method, hits=hits[:top_k], candidate_count=len(hits), encoding_ms=encoding_ms, ranking_ms=(time.perf_counter()-start)*1000)


def evidence_groups(question):
    # Each group is one required evidence item; alternative chunk IDs satisfy that group.
    groups = []
    for e in question["gold_evidence"]:
        ids = e.get("relevant_chunk_ids") or [e["chunk_id"]]
        groups.append(set(ids))
    return groups


def score_evidence(question, hits):
    groups = evidence_groups(question)
    seen = {h["chunk"]["chunk_id"] for h in hits}
    found = sum(bool(g & seen) for g in groups)
    positions = [i+1 for i,h in enumerate(hits) if any(h["chunk"]["chunk_id"] in g for g in groups)]
    return dict(evidence_recall=found/len(groups), complete_evidence=int(found==len(groups)), reciprocal_rank=1/min(positions) if positions else 0.0)
