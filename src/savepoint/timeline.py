"""Timeline: the DAG of saves that a forkable world induces.

Every ``.wsave`` file is a node; ``parent_id`` edges record where a timeline
was branched. A linear play-through is a chain; rewinding and playing again
adds a sibling; sharing a save lets someone else grow your tree on their
machine. The registry is a plain JSON file next to the saves — no database,
no pickle, human-inspectable.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class SaveNode:
    save_id: str
    parent_id: Optional[str]          # None for a world's root save
    path: str                         # .wsave file, relative to registry dir
    latent_frame: int                 # cursor at capture time
    mode: str                         # "state" | "reprime"
    created_at: float                 # unix seconds (wall time, informational)
    label: str = ""                   # user-facing name ("before the jump")
    meta: Dict = field(default_factory=dict)


class Timeline:
    """A forest of save nodes persisted as ``timeline.json`` in one directory."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._index_path = os.path.join(root_dir, "timeline.json")
        self.nodes: Dict[str, SaveNode] = {}
        os.makedirs(root_dir, exist_ok=True)
        if os.path.exists(self._index_path):
            self._load()

    # -- mutation ------------------------------------------------------------

    def add(
        self,
        *,
        path: str,
        latent_frame: int,
        mode: str,
        parent_id: Optional[str] = None,
        label: str = "",
        meta: Optional[Dict] = None,
        created_at: Optional[float] = None,
    ) -> SaveNode:
        if parent_id is not None and parent_id not in self.nodes:
            raise KeyError(f"unknown parent {parent_id}")
        node = SaveNode(
            save_id=uuid.uuid4().hex[:12],
            parent_id=parent_id,
            path=os.path.relpath(path, self.root_dir) if os.path.isabs(path) else path,
            latent_frame=latent_frame,
            mode=mode,
            created_at=time.time() if created_at is None else created_at,
            label=label,
            meta=dict(meta or {}),
        )
        self.nodes[node.save_id] = node
        self._flush()
        return node

    def relabel(self, save_id: str, label: str) -> None:
        self.nodes[save_id].label = label
        self._flush()

    # -- queries ---------------------------------------------------------------

    def children(self, save_id: Optional[str]) -> List[SaveNode]:
        return sorted((n for n in self.nodes.values() if n.parent_id == save_id),
                      key=lambda n: n.created_at)

    def roots(self) -> List[SaveNode]:
        return self.children(None)

    def lineage(self, save_id: str) -> List[SaveNode]:
        """Root → ... → node path (the timeline you'd replay to get here)."""
        chain: List[SaveNode] = []
        cur: Optional[str] = save_id
        while cur is not None:
            node = self.nodes[cur]
            chain.append(node)
            cur = node.parent_id
        return list(reversed(chain))

    def wsave_path(self, save_id: str) -> str:
        return os.path.join(self.root_dir, self.nodes[save_id].path)

    def to_tree(self) -> List[Dict]:
        """Nested dict form for the browser UI's timeline-tree widget."""
        def build(node: SaveNode) -> Dict:
            d = asdict(node)
            d["children"] = [build(c) for c in self.children(node.save_id)]
            return d
        return [build(r) for r in self.roots()]

    # -- persistence -------------------------------------------------------------

    def _flush(self) -> None:
        tmp = self._index_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"version": 0, "nodes": [asdict(n) for n in self.nodes.values()]},
                      fh, indent=2)
        os.replace(tmp, self._index_path)

    def _load(self) -> None:
        with open(self._index_path) as fh:
            data = json.load(fh)
        self.nodes = {n["save_id"]: SaveNode(**n) for n in data["nodes"]}
