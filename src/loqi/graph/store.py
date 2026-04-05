"""SQLite-backed graph storage for Loqi.

Provides persistent storage for nodes, edges, and triggers with an
adjacency table design. Embeddings are stored as binary blobs.

Usage:
    store = GraphStore(":memory:")  # or a file path
    store.add_node(node)
    store.add_edge(edge)
    neighbors = store.get_neighbors(node_id, edge_type=EdgeType.HARD)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from loqi.graph.models import Edge, EdgeType, Node, NodeType, Trigger, TriggerOrigin

_MAX_ID_LENGTH = 256


def _validate_id(value: str, label: str = "id") -> None:
    """Validate that an ID string is safe for storage."""
    if not value:
        raise ValueError(f"{label} cannot be empty")
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{label} exceeds max length ({len(value)} > {_MAX_ID_LENGTH})")


class GraphStore:
    """SQLite-backed storage for the memory graph."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                node_type TEXT NOT NULL DEFAULT 'section',
                parent_id TEXT,
                embedding BLOB,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.5,
                edge_type TEXT NOT NULL DEFAULT 'diffuse',
                co_activation_count INTEGER NOT NULL DEFAULT 0,
                last_strengthened TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE TABLE IF NOT EXISTS triggers (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                pattern_embedding BLOB,
                associated_node_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                fire_count INTEGER NOT NULL DEFAULT 0,
                useful_count INTEGER NOT NULL DEFAULT 0,
                origin TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                FOREIGN KEY (associated_node_id) REFERENCES nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_triggers_node ON triggers(associated_node_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        _validate_id(node.id, "node.id")
        embedding_blob = node.embedding.tobytes() if node.embedding is not None else None
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(id, title, content, node_type, parent_id, embedding, access_count, last_accessed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node.id,
                node.title,
                node.content,
                node.node_type.value,
                node.parent_id,
                embedding_blob,
                node.access_count,
                node.last_accessed.isoformat(),
                node.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_all_nodes(self) -> list[Node]:
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    def update_node_access(self, node_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE nodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, node_id),
        )
        self._conn.commit()

    def get_node_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        _validate_id(edge.source_id, "edge.source_id")
        _validate_id(edge.target_id, "edge.target_id")
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (source_id, target_id, weight, edge_type, co_activation_count, last_strengthened) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                edge.source_id,
                edge.target_id,
                edge.weight,
                edge.edge_type.value,
                edge.co_activation_count,
                edge.last_strengthened.isoformat(),
            ),
        )
        self._conn.commit()

    def get_edge(self, source_id: str, target_id: str) -> Edge | None:
        row = self._conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        min_weight: float = 0.0,
    ) -> list[tuple[str, Edge]]:
        """Get neighboring node IDs and their edges.

        Returns (neighbor_id, edge) pairs sorted by weight descending.
        """
        query = "SELECT * FROM edges WHERE source_id = ? AND weight >= ?"
        params: list = [node_id, min_weight]

        if edge_type is not None:
            query += " AND edge_type = ?"
            params.append(edge_type.value)

        query += " ORDER BY weight DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [(r["target_id"], self._row_to_edge(r)) for r in rows]

    def strengthen_edge(self, source_id: str, target_id: str, amount: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE edges SET weight = MIN(weight + ?, 1.0), "
            "co_activation_count = co_activation_count + 1, "
            "last_strengthened = ? "
            "WHERE source_id = ? AND target_id = ?",
            (amount, now, source_id, target_id),
        )
        self._conn.commit()

    def decay_edge(self, source_id: str, target_id: str, amount: float) -> None:
        self._conn.execute(
            "UPDATE edges SET weight = MAX(weight - ?, 0.0) "
            "WHERE source_id = ? AND target_id = ?",
            (amount, source_id, target_id),
        )
        self._conn.commit()

    def promote_edge(self, source_id: str, target_id: str, new_type: EdgeType) -> None:
        self._conn.execute(
            "UPDATE edges SET edge_type = ? WHERE source_id = ? AND target_id = ?",
            (new_type.value, source_id, target_id),
        )
        self._conn.commit()

    def get_edge_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def add_trigger(self, trigger: Trigger) -> None:
        _validate_id(trigger.id, "trigger.id")
        _validate_id(trigger.associated_node_id, "trigger.associated_node_id")
        pattern_json = json.dumps(trigger.pattern)
        embedding_blob = (
            trigger.pattern_embedding.tobytes()
            if trigger.pattern_embedding is not None
            else None
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO triggers "
            "(id, pattern, pattern_embedding, associated_node_id, confidence, fire_count, useful_count, origin, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trigger.id,
                pattern_json,
                embedding_blob,
                trigger.associated_node_id,
                trigger.confidence,
                trigger.fire_count,
                trigger.useful_count,
                trigger.origin.value,
                trigger.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_all_triggers(self) -> list[Trigger]:
        rows = self._conn.execute("SELECT * FROM triggers").fetchall()
        return [self._row_to_trigger(r) for r in rows]

    def update_trigger_fire(self, trigger_id: str, was_useful: bool) -> None:
        if was_useful:
            self._conn.execute(
                "UPDATE triggers SET fire_count = fire_count + 1, useful_count = useful_count + 1 WHERE id = ?",
                (trigger_id,),
            )
        else:
            self._conn.execute(
                "UPDATE triggers SET fire_count = fire_count + 1 WHERE id = ?",
                (trigger_id,),
            )
        self._conn.commit()

    def decay_trigger(self, trigger_id: str, amount: float) -> None:
        self._conn.execute(
            "UPDATE triggers SET confidence = MAX(confidence - ?, 0.0) WHERE id = ?",
            (amount, trigger_id),
        )
        self._conn.commit()

    def get_trigger_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM triggers").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._conn.executescript(
            "DELETE FROM triggers; DELETE FROM edges; DELETE FROM nodes;"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Row converters
    # ------------------------------------------------------------------

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        embedding = None
        if row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()

        return Node(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            parent_id=row["parent_id"],
            embedding=embedding,
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            weight=row["weight"],
            edge_type=EdgeType(row["edge_type"]),
            co_activation_count=row["co_activation_count"],
            last_strengthened=datetime.fromisoformat(row["last_strengthened"]),
        )

    def _row_to_trigger(self, row: sqlite3.Row) -> Trigger:
        embedding = None
        if row["pattern_embedding"] is not None:
            embedding = np.frombuffer(
                row["pattern_embedding"], dtype=np.float32
            ).copy()

        return Trigger(
            id=row["id"],
            pattern=json.loads(row["pattern"]),
            pattern_embedding=embedding,
            associated_node_id=row["associated_node_id"],
            confidence=row["confidence"],
            fire_count=row["fire_count"],
            useful_count=row["useful_count"],
            origin=TriggerOrigin(row["origin"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
