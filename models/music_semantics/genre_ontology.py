"""Hierarchical genre taxonomy with distance metrics and feature-based classification."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GenreNode:
    name: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    attributes: dict[str, float] = field(default_factory=dict)


def _build_tree() -> dict[str, GenreNode]:
    t: dict[str, GenreNode] = {}

    def add(name: str, parent: str | None, **attrs: float):
        t[name] = GenreNode(name, parent, [], attrs)
        if parent and parent in t:
            t[parent].children.append(name)

    add("root", None)
    for top in ("rock", "pop", "electronic", "jazz", "classical", "hip_hop", "folk", "world", "metal", "rnb", "country", "blues", "reggae", "latin", "funk", "soul"):
        add(top, "root")

    add("indie_rock", "rock"); add("punk_rock", "rock"); add("alt_rock", "rock"); add("prog_rock", "rock"); add("hard_rock", "rock"); add("garage_rock", "rock")
    add("synth_pop", "pop"); add("dream_pop", "pop"); add("art_pop", "pop"); add("k_pop", "pop"); add("j_pop", "pop")
    add("house", "electronic"); add("techno", "electronic"); add("drum_and_bass", "electronic"); add("ambient", "electronic"); add("trance", "electronic"); add("dubstep", "electronic"); add("idm", "electronic")
    add("bebop", "jazz"); add("cool_jazz", "jazz"); add("fusion", "jazz"); add("swing", "jazz"); add("free_jazz", "jazz"); add("latin_jazz", "jazz")
    add("baroque", "classical"); add("romantic", "classical"); add("modern_classical", "classical"); add("minimalist", "classical"); add("chamber", "classical")
    add("trap", "hip_hop"); add("boom_bap", "hip_hop"); add("conscious_rap", "hip_hop"); add("lo_fi_hip_hop", "hip_hop")
    add("americana", "folk"); add("celtic", "folk"); add("singer_songwriter", "folk")
    add("afrobeat", "world"); add("bossa_nova", "world"); add("flamenco", "world"); add("klezmer", "world"); add("fado", "world")
    add("death_metal", "metal"); add("black_metal", "metal"); add("thrash_metal", "metal"); add("doom_metal", "metal"); add("prog_metal", "metal")
    add("neo_soul", "rnb"); add("contemporary_rnb", "rnb")
    add("alt_country", "country"); add("bluegrass", "country"); add("outlaw_country", "country")
    add("delta_blues", "blues"); add("chicago_blues", "blues"); add("electric_blues", "blues")
    add("dancehall", "reggae"); add("dub", "reggae"); add("ska", "reggae")
    add("salsa", "latin"); add("cumbia", "latin"); add("reggaeton", "latin"); add("bachata", "latin")
    add("p_funk", "funk"); add("electro_funk", "funk")
    add("northern_soul", "soul"); add("motown", "soul")
    return t


GENRE_TREE: dict[str, GenreNode] = _build_tree()


def genre_ancestors(genre: str) -> list[str]:
    path = []
    current = genre
    while current and current in GENRE_TREE:
        path.append(current)
        current = GENRE_TREE[current].parent
    return path


def genre_descendants(genre: str) -> list[str]:
    if genre not in GENRE_TREE:
        return []
    result = []
    stack = [genre]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(GENRE_TREE[node].children if node in GENRE_TREE else [])
    return result


def common_ancestor(g1: str, g2: str) -> str | None:
    ancestors1 = set(genre_ancestors(g1))
    for a in genre_ancestors(g2):
        if a in ancestors1:
            return a
    return None


def genre_distance(g1: str, g2: str) -> float:
    if g1 == g2:
        return 0.0
    a1 = genre_ancestors(g1)
    a2 = genre_ancestors(g2)
    ca = common_ancestor(g1, g2)
    if ca is None:
        return 1.0
    d1 = a1.index(ca) if ca in a1 else len(a1)
    d2 = a2.index(ca) if ca in a2 else len(a2)
    max_depth = max(len(a1), len(a2), 1)
    return (d1 + d2) / (2 * max_depth)


def features_to_genre(bpm: float, energy: float, valence: float, acousticness: float) -> str:
    if acousticness > 0.8 and energy < 0.3:
        return "ambient" if bpm < 100 else "folk"
    if acousticness > 0.7:
        if valence > 0.6:
            return "country" if bpm > 100 else "bossa_nova"
        return "classical" if energy < 0.4 else "blues"
    if energy > 0.8:
        if bpm > 150:
            return "metal" if valence < 0.4 else "drum_and_bass"
        if bpm > 120:
            return "rock" if acousticness > 0.3 else "electronic"
        return "funk" if valence > 0.6 else "hip_hop"
    if bpm > 120 and energy > 0.5:
        return "pop" if valence > 0.5 else "techno"
    if valence > 0.6 and energy > 0.4:
        return "rnb" if bpm < 110 else "reggae"
    if energy < 0.4:
        return "jazz" if acousticness > 0.5 else "lo_fi_hip_hop"
    return "pop"
