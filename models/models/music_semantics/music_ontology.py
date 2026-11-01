"""Music ontology: knowledge graph for musical concepts, instruments, genres, scales.

Implements a directed graph of musical concepts with typed edges (is_a, part_of,
used_in, related_to, derived_from, influences, contrasts_with) and standard
graph algorithms (BFS shortest path, topological sort, subgraph extraction).

MusicKnowledgeBase provides a pre-built ontology covering Western instrument
families, genre hierarchies, scale types, chord qualities, and techniques.
"""
from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class NodeType(enum.Enum):
    INSTRUMENT = "instrument"
    GENRE = "genre"
    TECHNIQUE = "technique"
    CONCEPT = "concept"
    EMOTION = "emotion"
    SCALE = "scale"
    CHORD = "chord"
    FORM = "form"
    FAMILY = "family"


class RelationType(enum.Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    USED_IN = "used_in"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    INFLUENCES = "influences"
    CONTRASTS_WITH = "contrasts_with"
    HAS_MEMBER = "has_member"
    PRODUCES = "produces"


@dataclass
class OntologyNode:
    id: str
    name: str
    node_type: NodeType
    attributes: Dict[str, object] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)


@dataclass
class OntologyEdge:
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class MusicOntology:
    """Directed graph of musical concepts with typed edges."""

    def __init__(self) -> None:
        self._nodes: Dict[str, OntologyNode] = {}
        self._edges: List[OntologyEdge] = []
        self._adj: Dict[str, List[OntologyEdge]] = {}
        self._rev: Dict[str, List[OntologyEdge]] = {}

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def add_node(self, node: OntologyNode) -> None:
        self._nodes[node.id] = node
        self._adj.setdefault(node.id, [])
        self._rev.setdefault(node.id, [])

    def add_edge(self, edge: OntologyEdge) -> None:
        self._edges.append(edge)
        self._adj.setdefault(edge.source_id, []).append(edge)
        self._rev.setdefault(edge.target_id, []).append(edge)

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> List[OntologyNode]:
        out = []
        for e in self._adj.get(node_id, []):
            n = self._nodes.get(e.target_id)
            if n is not None:
                out.append(n)
        return out

    def get_parents(self, node_id: str) -> List[OntologyNode]:
        out = []
        for e in self._rev.get(node_id, []):
            n = self._nodes.get(e.source_id)
            if n is not None:
                out.append(n)
        return out

    def get_related(
        self, node_id: str, relation: RelationType
    ) -> List[OntologyNode]:
        out = []
        for e in self._adj.get(node_id, []):
            if e.relation == relation:
                n = self._nodes.get(e.target_id)
                if n is not None:
                    out.append(n)
        return out

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return [src]
        visited: Set[str] = {src}
        queue: deque[List[str]] = deque([[src]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            for e in self._adj.get(current, []):
                nxt = e.target_id
                if nxt == dst:
                    return path + [nxt]
                if nxt not in visited and nxt in self._nodes:
                    visited.add(nxt)
                    queue.append(path + [nxt])
            for e in self._rev.get(current, []):
                nxt = e.source_id
                if nxt == dst:
                    return path + [nxt]
                if nxt not in visited and nxt in self._nodes:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def semantic_distance(self, src: str, dst: str) -> float:
        path = self.shortest_path(src, dst)
        if path is None:
            return float("inf")
        length = len(path) - 1
        if length == 0:
            return 0.0
        return 1.0 / length

    def subgraph(self, node_ids: Set[str]) -> "MusicOntology":
        g = MusicOntology()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                g.add_node(node)
        for e in self._edges:
            if e.source_id in node_ids and e.target_id in node_ids:
                g.add_edge(e)
        return g

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        ids = sorted(self._nodes.keys())
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        mat = np.zeros((n, n))
        for e in self._edges:
            i = idx.get(e.source_id)
            j = idx.get(e.target_id)
            if i is not None and j is not None:
                mat[i, j] = e.weight
        return mat, ids

    def topological_sort(self) -> List[str]:
        in_deg: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            if e.target_id in in_deg:
                in_deg[e.target_id] += 1
        queue = deque(nid for nid, d in in_deg.items() if d == 0)
        result: List[str] = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for e in self._adj.get(nid, []):
                if e.target_id in in_deg:
                    in_deg[e.target_id] -= 1
                    if in_deg[e.target_id] == 0:
                        queue.append(e.target_id)
        return result

    def find_by_type(self, node_type: NodeType) -> List[OntologyNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def find_by_name(self, pattern: str) -> List[OntologyNode]:
        pat = pattern.lower()
        return [n for n in self._nodes.values() if pat in n.name.lower()]


class MusicKnowledgeBase:
    """Pre-built ontology of common Western music concepts."""

    @classmethod
    def build(cls) -> MusicOntology:
        g = MusicOntology()
        cls._add_instrument_families(g)
        cls._add_genres(g)
        cls._add_scales(g)
        cls._add_chords(g)
        cls._add_techniques(g)
        cls._add_cross_links(g)
        return g

    @classmethod
    def _add_instrument_families(cls, g: MusicOntology) -> None:
        families = {
            "strings": ["violin", "viola", "cello", "double_bass", "harp", "guitar"],
            "woodwinds": ["flute", "oboe", "clarinet", "bassoon", "saxophone"],
            "brass": ["trumpet", "french_horn", "trombone", "tuba"],
            "percussion": ["timpani", "snare_drum", "bass_drum", "cymbals", "xylophone", "marimba"],
            "keyboard": ["piano", "harpsichord", "organ", "celesta"],
            "electronic": ["synthesizer", "drum_machine", "sampler", "theremin"],
        }
        root = OntologyNode("instruments", "Instruments", NodeType.CONCEPT)
        g.add_node(root)
        for fam_name, members in families.items():
            fam = OntologyNode(f"fam_{fam_name}", fam_name.title(), NodeType.FAMILY)
            g.add_node(fam)
            g.add_edge(OntologyEdge("instruments", fam.id, RelationType.HAS_MEMBER))
            for m in members:
                inst = OntologyNode(f"inst_{m}", m.replace("_", " ").title(), NodeType.INSTRUMENT, parent_ids=[fam.id])
                g.add_node(inst)
                g.add_edge(OntologyEdge(fam.id, inst.id, RelationType.HAS_MEMBER))

    @classmethod
    def _add_genres(cls, g: MusicOntology) -> None:
        tree = {
            "classical": ["baroque", "classical_period", "romantic", "modern_classical", "impressionist"],
            "jazz": ["bebop", "cool_jazz", "fusion", "smooth_jazz", "free_jazz", "swing"],
            "electronic": ["house", "techno", "ambient", "drum_and_bass", "trance", "dubstep"],
            "rock": ["classic_rock", "punk", "metal", "alternative", "progressive_rock", "indie"],
            "folk": ["country", "bluegrass", "celtic", "world_music"],
            "pop": ["synth_pop", "dance_pop", "art_pop", "indie_pop"],
            "hip_hop": ["trap", "boom_bap", "lo_fi_hip_hop", "experimental_hip_hop"],
            "blues": ["delta_blues", "chicago_blues", "electric_blues"],
        }
        root = OntologyNode("genres", "Genres", NodeType.CONCEPT)
        g.add_node(root)
        for parent, children in tree.items():
            pnode = OntologyNode(f"genre_{parent}", parent.replace("_", " ").title(), NodeType.GENRE)
            g.add_node(pnode)
            g.add_edge(OntologyEdge("genres", pnode.id, RelationType.HAS_MEMBER))
            for ch in children:
                cnode = OntologyNode(f"genre_{ch}", ch.replace("_", " ").title(), NodeType.GENRE, parent_ids=[pnode.id])
                g.add_node(cnode)
                g.add_edge(OntologyEdge(pnode.id, cnode.id, RelationType.IS_A))

    @classmethod
    def _add_scales(cls, g: MusicOntology) -> None:
        scales = [
            "major", "natural_minor", "harmonic_minor", "melodic_minor",
            "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian",
            "pentatonic_major", "pentatonic_minor", "blues",
            "whole_tone", "chromatic", "diminished", "augmented",
            "hungarian_minor", "spanish_phrygian", "japanese_hirajoshi",
        ]
        root = OntologyNode("scales", "Scales", NodeType.CONCEPT)
        g.add_node(root)
        for s in scales:
            node = OntologyNode(f"scale_{s}", s.replace("_", " ").title(), NodeType.SCALE, parent_ids=["scales"])
            g.add_node(node)
            g.add_edge(OntologyEdge("scales", node.id, RelationType.HAS_MEMBER))
        g.add_edge(OntologyEdge("scale_natural_minor", "scale_harmonic_minor", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("scale_harmonic_minor", "scale_melodic_minor", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("scale_major", "scale_pentatonic_major", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("scale_pentatonic_minor", "scale_blues", RelationType.DERIVED_FROM))

    @classmethod
    def _add_chords(cls, g: MusicOntology) -> None:
        chords = [
            "major", "minor", "diminished", "augmented",
            "dominant7", "major7", "minor7", "half_diminished7", "diminished7",
            "sus2", "sus4", "add9", "major9", "minor9",
            "power_chord", "augmented6",
        ]
        root = OntologyNode("chords", "Chords", NodeType.CONCEPT)
        g.add_node(root)
        for c in chords:
            node = OntologyNode(f"chord_{c}", c.replace("_", " ").title(), NodeType.CHORD, parent_ids=["chords"])
            g.add_node(node)
            g.add_edge(OntologyEdge("chords", node.id, RelationType.HAS_MEMBER))
        g.add_edge(OntologyEdge("chord_major", "chord_dominant7", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("chord_major", "chord_major7", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("chord_minor", "chord_minor7", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("chord_diminished", "chord_half_diminished7", RelationType.DERIVED_FROM))
        g.add_edge(OntologyEdge("chord_diminished", "chord_diminished7", RelationType.DERIVED_FROM))

    @classmethod
    def _add_techniques(cls, g: MusicOntology) -> None:
        techniques = [
            "legato", "staccato", "vibrato", "tremolo", "pizzicato", "arco",
            "glissando", "portamento", "trill", "mordent", "turn",
            "harmonics", "mute", "bend", "slide", "hammer_on", "pull_off",
            "palm_mute", "sweep_picking", "fingerpicking",
        ]
        root = OntologyNode("techniques", "Techniques", NodeType.CONCEPT)
        g.add_node(root)
        for t in techniques:
            node = OntologyNode(f"tech_{t}", t.replace("_", " ").title(), NodeType.TECHNIQUE, parent_ids=["techniques"])
            g.add_node(node)
            g.add_edge(OntologyEdge("techniques", node.id, RelationType.HAS_MEMBER))
        g.add_edge(OntologyEdge("tech_legato", "tech_portamento", RelationType.RELATED_TO))
        g.add_edge(OntologyEdge("tech_staccato", "tech_legato", RelationType.CONTRASTS_WITH))
        g.add_edge(OntologyEdge("tech_pizzicato", "tech_arco", RelationType.CONTRASTS_WITH))
        g.add_edge(OntologyEdge("tech_vibrato", "tech_tremolo", RelationType.RELATED_TO))
        g.add_edge(OntologyEdge("tech_hammer_on", "tech_pull_off", RelationType.CONTRASTS_WITH))

    @classmethod
    def _add_cross_links(cls, g: MusicOntology) -> None:
        g.add_edge(OntologyEdge("genre_blues", "scale_blues", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_jazz", "chord_dominant7", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_jazz", "chord_major7", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_classical", "tech_legato", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_classical", "tech_vibrato", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_rock", "chord_power_chord", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_electronic", "inst_synthesizer", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_electronic", "inst_drum_machine", RelationType.USED_IN))
        g.add_edge(OntologyEdge("genre_blues", "genre_jazz", RelationType.INFLUENCES))
        g.add_edge(OntologyEdge("genre_blues", "genre_rock", RelationType.INFLUENCES))
        g.add_edge(OntologyEdge("genre_jazz", "genre_hip_hop", RelationType.INFLUENCES))
