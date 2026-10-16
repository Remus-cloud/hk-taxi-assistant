from .router import PolicyRetriever,QueryRouter
import json

class ExpandedPolicyRetriever(PolicyRetriever):
    def __init__(self,base):
        self.base=base
        self.tags=json.loads((base.root/'data/curated/content_types.json').read_text(encoding='utf-8'))
        self.policy=json.loads((base.root/'configs/router_v2.json').read_text(encoding='utf-8'))
        self.router=QueryRouter(self.policy)
        assert set(self.tags)=={c['chunk_id'] for c in base.chunks}
