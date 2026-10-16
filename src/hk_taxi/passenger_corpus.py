"""Composition adapter for the current corpus, without an experimental snapshot."""
from pathlib import Path
import json
from .rag_answer import FrozenRagSession


class PassengerCorpusSession:
    def __init__(self, root, frozen_session=None):
        self.root = Path(root)
        self.frozen = frozen_session or FrozenRagSession(self.root, with_generator=False)
        self._owns_frozen = frozen_session is None
        self.base = self.frozen.base
        self.retriever = self.frozen.retriever
        self.manifest = self.base.manifest
        self.source_scopes = json.loads((self.root/'data/curated/source_scope.annotations.json').read_text(encoding='utf-8'))

    def prepare_vectors(self, questions):
        return self.frozen.prepare_vectors(questions)

    def retrieve(self, question, vectors):
        return self.frozen.retrieve(question, vectors)

    def close(self):
        if self._owns_frozen:
            self.frozen.close()
