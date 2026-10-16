"""Passenger web runtime: expanded corpus + pretrained reranker, session-local state."""
from copy import deepcopy
from pathlib import Path
import json
from .web_chat import WebChatSession as OriginalWebChatSession
from .unified_passenger_v2 import UnifiedAssistant
from .passenger_tools_v2 import upgrade_state

class WebChatSession(OriginalWebChatSession):
    runtime_version='passenger-web-pretrained-v1'

    def __init__(self,root,engine_factory=None):
        root=Path(root)
        if engine_factory is None:
            config=json.loads((root/'configs/web_runtime_v2.json').read_text(encoding='utf-8'))
            if config['reranker']!='pretrained':raise ValueError('Unexpected default web retrieval profile')
            engine_factory=lambda:UnifiedAssistant(root,reranker='pretrained')
        super().__init__(root,engine_factory=engine_factory)
        self.state=upgrade_state(self.state)

    def new_chat(self):
        changed=super().new_chat()
        if changed:self.state=upgrade_state(self.state)
        return changed
