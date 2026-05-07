"""Side info for ML-20M: genres, decade, optional Tag Genome scores.

Side embeddings are summed with item ID embedding inside the model. This is the
content-aware boost from IDEAS.md #2 — typically +2..4% NDCG@10 on ML-20M.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
N_GENRES = len(GENRE_LIST)
DECADE_BUCKETS = list(range(1900, 2030, 10))     # 1900s..2020s
N_DECADES = len(DECADE_BUCKETS) + 1               # +1 for unknown bucket
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class SideInfoTables:
    """Per-item side info indexed by dense item idx (1..n_items)."""

    genre_multi_hot: np.ndarray           # [n_items+1, N_GENRES] float32
    decade_idx: np.ndarray                # [n_items+1] long
    genome: Optional[np.ndarray] = None   # [n_items+1, 1128] float32 or None


def _parse_year(title: str) -> int:
    m = YEAR_RE.search(title or "")
    return int(m.group(1)) if m else 0


def _decade_bucket(year: int) -> int:
    if year < 1900:
        return 0  # unknown / out-of-range
    bucket = (year // 10) * 10
    if bucket not in DECADE_BUCKETS:
        return 0
    return DECADE_BUCKETS.index(bucket) + 1


def build_sideinfo(
    movies_csv: str | Path,
    movie_id_to_idx: Dict[int, int],
    n_items: int,
    genome_scores_csv: str | Path | None = None,
) -> SideInfoTables:
    movies = pd.read_csv(movies_csv)
    genre_table = np.zeros((n_items + 1, N_GENRES), dtype=np.float32)
    decade_table = np.zeros((n_items + 1,), dtype=np.int64)

    for row in movies.itertuples(index=False):
        idx = movie_id_to_idx.get(row.movieId)
        if idx is None:
            continue
        for g in str(row.genres).split("|"):
            if g in GENRE_LIST:
                genre_table[idx, GENRE_LIST.index(g)] = 1.0
        decade_table[idx] = _decade_bucket(_parse_year(row.title))

    genome = None
    if genome_scores_csv is not None and Path(genome_scores_csv).exists():
        print(f"[sideinfo] loading genome scores from {genome_scores_csv}")
        gs = pd.read_csv(genome_scores_csv)
        # Wide pivot: rows = movieId, cols = tagId (1..1128), vals = relevance.
        wide = gs.pivot(index="movieId", columns="tagId", values="relevance")
        wide = wide.fillna(0.0).astype(np.float32)
        n_tags = wide.shape[1]
        genome = np.zeros((n_items + 1, n_tags), dtype=np.float32)
        for movie_id, vec in wide.iterrows():
            idx = movie_id_to_idx.get(int(movie_id))
            if idx is not None:
                genome[idx] = vec.values
        print(f"[sideinfo] genome shape: {genome.shape}")

    return SideInfoTables(
        genre_multi_hot=genre_table,
        decade_idx=decade_table,
        genome=genome,
    )


class SideInfoEmbedding(nn.Module):
    """Maps (genre multi-hot, decade idx, genome vector) → d-dim vector.

    Output is added to item ID embedding. PAD token (idx=0) gets zero contribution
    by virtue of zero genre/genome rows and decade_idx=0 (handled via padding_idx).
    """

    def __init__(
        self,
        d: int,
        side: SideInfoTables,
        use_genome: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d

        self.register_buffer(
            "genre_table", torch.from_numpy(side.genre_multi_hot)
        )
        self.register_buffer(
            "decade_table", torch.from_numpy(side.decade_idx)
        )

        self.genre_proj = nn.Linear(N_GENRES, d, bias=False)
        self.decade_emb = nn.Embedding(N_DECADES, d, padding_idx=0)

        self.use_genome = use_genome and side.genome is not None
        if self.use_genome:
            self.register_buffer("genome_table", torch.from_numpy(side.genome))
            self.genome_proj = nn.Linear(side.genome.shape[1], d, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.genre_proj.weight, std=0.02)
        nn.init.normal_(self.decade_emb.weight, std=0.02)
        with torch.no_grad():
            self.decade_emb.weight[0].fill_(0.0)  # padding row stays zero
        if self.use_genome:
            nn.init.normal_(self.genome_proj.weight, std=0.02)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids: [B, L] long → [B, L, d] float."""
        genre = self.genre_table[item_ids]          # [B, L, N_GENRES]
        decade = self.decade_table[item_ids]        # [B, L]
        out = self.genre_proj(genre) + self.decade_emb(decade)
        if self.use_genome:
            genome = self.genome_table[item_ids]    # [B, L, n_tags]
            out = out + self.genome_proj(genome)
        return self.dropout(out)
