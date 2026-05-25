"""
matcher.py
Compares a query color palette against a library of Rothko painting profiles
and returns the closest matches using weighted nearest-neighbor distance in LAB space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def load_profiles(profiles_path: str | Path) -> list[dict]:
    """Load profiles.json into memory."""
    with open(profiles_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lab_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two LAB triplets."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compare_palettes(palette_a: list[dict], palette_b: list[dict]) -> float:
    """
    Weighted greedy nearest-neighbor distance between two palettes in LAB space.

    For each color in palette_a (sorted by weight descending), finds its nearest
    neighbor in palette_b by LAB Euclidean distance, then multiplies by palette_a's
    weight. The sum of these weighted distances is the score (lower = more similar).

    Asymmetric: palette_a's weights drive the penalty, so we measure how well
    palette_b covers the dominant colors of palette_a.
    """
    sorted_a = sorted(palette_a, key=lambda c: c["weight"], reverse=True)

    total_score = 0.0
    for color_a in sorted_a:
        best_dist = min(
            _lab_distance(color_a["lab"], color_b["lab"]) for color_b in palette_b
        )
        total_score += best_dist * color_a["weight"]

    return total_score


def find_matches(
    query_palette: list[dict],
    profiles: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    Rank all profiles by color similarity to the query palette.

    Returns a list of up to top_n dicts, each being a copy of the profile with
    an extra "score" (raw distance, lower = better) and "similarity" (0-100 display value).
    """
    scored = []
    for profile in profiles:
        score = compare_palettes(query_palette, profile["palette"])
        scored.append((score, profile))

    scored.sort(key=lambda x: x[0])

    results = []
    for score, profile in scored[:top_n]:
        entry = dict(profile)
        entry["score"] = score
        # Cosmetic 0–100 similarity: clip raw score at 100 and invert
        entry["similarity"] = max(0.0, 100.0 - score)
        results.append(entry)

    return results
