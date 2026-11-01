"""Cross-modal sensory associations: music-colour, music-texture, music-shape synesthesia models."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ColourAssociation:
    hue: float
    saturation: float
    lightness: float


PITCH_HUE_MAP = {
    0: 0.0,    1: 30.0,   2: 60.0,   3: 90.0,
    4: 120.0,  5: 150.0,  6: 180.0,  7: 210.0,
    8: 240.0,  9: 270.0,  10: 300.0, 11: 330.0,
}


def pitch_to_colour(pitch_class: int, velocity: float = 0.7,
                     register: float = 0.5) -> ColourAssociation:
    pc = pitch_class % 12
    hue = PITCH_HUE_MAP[pc]
    saturation = np.clip(velocity, 0.0, 1.0)
    lightness = 0.3 + 0.4 * np.clip(register, 0.0, 1.0)
    return ColourAssociation(hue=float(hue), saturation=float(saturation),
                             lightness=float(lightness))


def chord_colour(pitch_classes: list[int], velocity: float = 0.7,
                  register: float = 0.5) -> ColourAssociation:
    if not pitch_classes:
        return ColourAssociation(0.0, 0.0, 0.5)
    colours = [pitch_to_colour(pc, velocity, register) for pc in pitch_classes]
    avg_hue = float(np.mean([c.hue for c in colours]))
    avg_sat = float(np.mean([c.saturation for c in colours]))
    avg_lit = float(np.mean([c.lightness for c in colours]))
    return ColourAssociation(avg_hue, avg_sat, avg_lit)


def colour_distance(a: ColourAssociation, b: ColourAssociation) -> float:
    dh = min(abs(a.hue - b.hue), 360.0 - abs(a.hue - b.hue)) / 180.0
    ds = abs(a.saturation - b.saturation)
    dl = abs(a.lightness - b.lightness)
    return float(np.sqrt(dh ** 2 + ds ** 2 + dl ** 2))


@dataclass
class TextureAssociation:
    roughness: float
    density: float
    warmth: float


def spectral_to_texture(spectral_centroid: float, spectral_flatness: float,
                         low_energy_ratio: float) -> TextureAssociation:
    roughness = float(np.clip(spectral_flatness, 0.0, 1.0))
    density = float(np.clip(1.0 - spectral_flatness, 0.0, 1.0))
    warmth = float(np.clip(low_energy_ratio, 0.0, 1.0))
    return TextureAssociation(roughness=roughness, density=density, warmth=warmth)


def texture_distance(a: TextureAssociation, b: TextureAssociation) -> float:
    return float(np.sqrt(
        (a.roughness - b.roughness) ** 2 +
        (a.density - b.density) ** 2 +
        (a.warmth - b.warmth) ** 2
    ))


@dataclass
class ShapeAssociation:
    angularity: float
    size: float
    symmetry: float


BOUBA_KIKI_MAP = {
    "staccato": ShapeAssociation(angularity=0.9, size=0.3, symmetry=0.5),
    "legato": ShapeAssociation(angularity=0.1, size=0.7, symmetry=0.7),
    "marcato": ShapeAssociation(angularity=0.8, size=0.6, symmetry=0.4),
    "tenuto": ShapeAssociation(angularity=0.2, size=0.5, symmetry=0.8),
}


def articulation_to_shape(articulation: str) -> ShapeAssociation:
    return BOUBA_KIKI_MAP.get(articulation,
                               ShapeAssociation(angularity=0.5, size=0.5, symmetry=0.5))


def dynamics_to_shape(loudness: float, attack_time: float) -> ShapeAssociation:
    angularity = float(np.clip(1.0 - attack_time * 10.0, 0.0, 1.0))
    size = float(np.clip(loudness, 0.0, 1.0))
    symmetry = 0.5
    return ShapeAssociation(angularity=angularity, size=size, symmetry=symmetry)


def shape_distance(a: ShapeAssociation, b: ShapeAssociation) -> float:
    return float(np.sqrt(
        (a.angularity - b.angularity) ** 2 +
        (a.size - b.size) ** 2 +
        (a.symmetry - b.symmetry) ** 2
    ))


def cross_modal_coherence(colour: ColourAssociation, texture: TextureAssociation,
                           shape: ShapeAssociation) -> float:
    bright_warm = 1.0 - abs(colour.lightness - texture.warmth)
    sat_dense = 1.0 - abs(colour.saturation - texture.density)
    angular_bright = 1.0 - abs(shape.angularity - (1.0 - colour.lightness))
    return float(np.mean([bright_warm, sat_dense, angular_bright]))


def synesthesia_profile(pitch_classes: list[int], velocity: float,
                         register: float, spectral_centroid: float,
                         spectral_flatness: float, low_energy_ratio: float,
                         articulation: str = "legato") -> dict:
    colour = chord_colour(pitch_classes, velocity, register)
    texture = spectral_to_texture(spectral_centroid, spectral_flatness, low_energy_ratio)
    shape = articulation_to_shape(articulation)
    coherence = cross_modal_coherence(colour, texture, shape)
    return {
        "colour": colour,
        "texture": texture,
        "shape": shape,
        "coherence": coherence,
    }
