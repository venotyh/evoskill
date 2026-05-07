"""Lineage tree — tracks skill inheritance relationships."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .skill import Skill
from .storage import load_lineage, load_skill, save_lineage


@dataclass
class LineageNode:
    """A node in the lineage tree."""
    skill_id: str
    name: str
    parent_ids: list[str]
    generation: int
    fitness: float
    mutation_type: str
    mutation_desc: str
    created_at: str


class LineageTree:
    """DAG tracking skill inheritance from primordial seed to current generation.

    Maintains an internal child index so that ancestor/descendant/tree-render
    queries are O(depth) instead of O(n^2) full scans.
    """

    def __init__(self):
        self.nodes: dict[str, LineageNode] = {}
        self._children: dict[str, list[str]] = defaultdict(list)
        self._roots: list[str] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        data = load_lineage()
        for node_id, node_data in data.get("nodes", {}).items():
            self.nodes[node_id] = LineageNode(**node_data)
        self._rebuild_index()

    def _save(self) -> None:
        data = {
            "nodes": {
                nid: {
                    "skill_id": node.skill_id,
                    "name": node.name,
                    "parent_ids": node.parent_ids,
                    "generation": node.generation,
                    "fitness": node.fitness,
                    "mutation_type": node.mutation_type,
                    "mutation_desc": node.mutation_desc,
                    "created_at": node.created_at,
                }
                for nid, node in self.nodes.items()
            },
        }
        save_lineage(data)

    def _rebuild_index(self) -> None:
        """Rebuild children index and roots list. O(n), called once per batch."""
        self._children = defaultdict(list)
        self._roots = []
        for nid, node in self.nodes.items():
            if not node.parent_ids:
                self._roots.append(nid)
            for pid in node.parent_ids:
                self._children[pid].append(nid)

    # ── mutation ───────────────────────────────────────────────────────

    def add_skill(self, skill: Skill, *, persist: bool = False) -> LineageNode:
        """Register a skill in the lineage tree.

        Set persist=True to immediately write to disk (single additions).
        For batch additions prefer add_skills() which saves once.
        """
        node = LineageNode(
            skill_id=skill.id,
            name=skill.name,
            parent_ids=skill.parent_ids,
            generation=skill.generation,
            fitness=skill.fitness,
            mutation_type=skill.mutation_type.value if skill.mutation_type else "seed",
            mutation_desc=skill.mutation_desc,
            created_at=skill.created_at,
        )
        self.nodes[skill.id] = node
        if not skill.parent_ids:
            self._roots.append(skill.id)
        for pid in skill.parent_ids:
            self._children[pid].append(skill.id)

        if persist:
            self._save()
        return node

    def add_skills(self, skills: list[Skill]) -> None:
        """Batch-register many skills, then persist once."""
        for s in skills:
            self.add_skill(s, persist=False)
        self._save()

    # ── queries (all O(depth) thanks to children index) ────────────────

    def ancestors(self, skill_id: str) -> list[LineageNode]:
        """Get the full ancestor chain of a skill (BFS up)."""
        result = []
        visited = set()
        queue = [skill_id]
        while queue:
            sid = queue.pop(0)
            if sid in visited:
                continue
            visited.add(sid)
            node = self.nodes.get(sid)
            if node:
                result.append(node)
                queue.extend(node.parent_ids)
        result.sort(key=lambda n: n.generation)
        return result

    def descendants(self, skill_id: str) -> list[LineageNode]:
        """Get all descendants of a skill (BFS down via child index, O(depth))."""
        result = []
        visited = set()
        queue = [skill_id]
        while queue:
            sid = queue.pop(0)
            if sid in visited:
                continue
            visited.add(sid)
            if sid != skill_id:
                node = self.nodes.get(sid)
                if node:
                    result.append(node)
            queue.extend(self._children.get(sid, []))
        return result

    def children(self, skill_id: str) -> list[LineageNode]:
        """Get direct children of a skill."""
        return [self.nodes[cid] for cid in self._children.get(skill_id, []) if cid in self.nodes]

    def by_generation(self) -> dict[int, list[LineageNode]]:
        """Group nodes by generation number."""
        grouped = defaultdict(list)
        for node in self.nodes.values():
            grouped[node.generation].append(node)
        return dict(sorted(grouped.items()))

    def roots(self) -> list[LineageNode]:
        """Get all root skills (no parents)."""
        return [self.nodes[nid] for nid in self._roots if nid in self.nodes]

    # ── rendering ──────────────────────────────────────────────────────

    def to_ascii_tree(self, root_id: str | None = None, max_depth: int = 5) -> str:
        """Render the lineage as an ASCII tree."""
        if root_id is None:
            roots = self.roots()
            if not roots:
                return "(empty lineage)"
            lines = []
            for root in roots:
                lines.append(self._render_subtree(root.skill_id, "", max_depth, True))
            return "\n".join(lines)
        return self._render_subtree(root_id, "", max_depth, True)

    def _render_subtree(self, skill_id: str, prefix: str, max_depth: int, is_last: bool) -> str:
        if max_depth <= 0:
            return ""

        node = self.nodes.get(skill_id)
        if not node:
            return f"{prefix}(unknown: {skill_id})"

        connector = "└──" if is_last else "├──"
        fitness_str = f"[{node.fitness:.1f}]" if node.fitness else "[?]"
        mut_str = f" ({node.mutation_type})" if node.mutation_type else ""
        line = f"{prefix}{connector} {node.name} {fitness_str}{mut_str}"

        children = self.children(skill_id)
        children.sort(key=lambda n: n.generation)

        if not children or max_depth <= 1:
            return line

        lines = [line]
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(
                self._render_subtree(child.skill_id, child_prefix, max_depth - 1, is_child_last)
            )
        return "\n".join(lines)

    # ── stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics."""
        if not self.nodes:
            return {}
        gens = [n.generation for n in self.nodes.values()]
        fits = [n.fitness for n in self.nodes.values() if n.fitness > 0]
        mut_types = defaultdict(int)
        for n in self.nodes.values():
            mut_types[n.mutation_type] += 1

        return {
            "total_skills": len(self.nodes),
            "max_generation": max(gens) if gens else 0,
            "best_fitness": max(fits) if fits else 0,
            "avg_fitness": round(sum(fits) / len(fits), 2) if fits else 0,
            "mutation_distribution": dict(mut_types),
            "roots": len(self._roots),
        }


def sync_lineage_from_disk() -> LineageTree:
    """Rebuild the lineage tree from all skill files on disk.

    Loads all skills, then persists lineage.json once (O(n) reads, 1 write).
    """
    from .storage import list_skills
    tree = LineageTree()
    tree.nodes.clear()
    skills = list_skills()
    for skill in skills:
        tree.add_skill(skill, persist=False)
    tree._rebuild_index()
    tree._save()
    return tree
