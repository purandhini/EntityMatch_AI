"""
Blocking module for candidate pair generation in Entity Resolution.

Reduces the quadratic O(N^2) comparison space to a tractable candidate set by
applying multiple complementary blocking strategies and unioning their results.

Strategies
----------
1. Token-prefix blocking on preprocessed business name (`*_clean` columns preferred).
2. Sorted Neighborhood Method (SNM) on the normalized name sort key.
3. Street-number blocking on the leading numeric component of a normalized address.
4. Exact-attribute blocking on secondary attributes (postal code, phone, website domain).

All strategies are run by default.  Each can be toggled via ``BlockingConfig``.
Candidate pairs are always returned as sorted ``(id_a, id_b)`` tuples with ``id_a < id_b``
(deduplication) or ``(id_left, id_right)`` (linkage), deduplicated via a set.
"""

import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from ..config import BlockingConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Returns stripped string or empty string for missing/null values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _name_sort_key(name: str) -> str:
    """Produces a whitespace-collapsed lowercase sort key for a business name."""
    return re.sub(r"\s+", " ", name.lower()).strip()


def _street_number(address: str) -> str:
    """
    Extracts the leading street number from a normalised address string.
    Returns the digit sequence, or '' if none found.

    Examples
    --------
    "100 n main st"      -> "100"
    "1600 pennsylvania"  -> "1600"
    ""                   -> ""
    """
    addr = address.strip()
    if not addr:
        return ""
    m = re.match(r"^(\d+)", addr)
    return m.group(1) if m else ""


def _row_id(row, id_col: str) -> str:
    """Extract string ID safely from dict, Series, or itertuple."""
    if hasattr(row, "get"):
        return _safe_str(row.get(id_col, ""))
    if hasattr(row, id_col):
        return _safe_str(getattr(row, id_col, ""))
    try:
        return _safe_str(row[id_col])
    except (IndexError, TypeError, KeyError):
        return ""


def _preferred_col(row, clean_col: str, raw_col: str) -> str:
    """Return cleaned column value if it exists and is non-empty, else fall back to raw."""
    if hasattr(row, "get"):
        val = _safe_str(row.get(clean_col, ""))
        if val:
            return val
        return _safe_str(row.get(raw_col, ""))
    val = _safe_str(getattr(row, clean_col, ""))
    if val:
        return val
    return _safe_str(getattr(row, raw_col, ""))


def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    """Returns a lexicographically ordered pair to deduplicate (a, b) == (b, a)."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Main blocker
# ---------------------------------------------------------------------------

class EntityBlocker:
    """
    Generates candidate record pairs using multiple blocking strategies.

    The blocker operates in two modes:

    * **Deduplication** (``df_right=None``): pairs records within a single table.
    * **Record linkage** (``df_right`` provided): pairs records across two tables.

    Parameters
    ----------
    config : BlockingConfig, optional
        Blocking configuration.  Defaults to ``DEFAULT_CONFIG.blocking``.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or DEFAULT_CONFIG.blocking
        self._left_cache: Dict[Tuple, Any] = {}

    def clear_cache(self):
        """Clears cached inverted indexes for candidate generation."""
        self._left_cache.clear()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_candidate_pairs_from_tables(
        self,
        df_left: pd.DataFrame,
        df_right: Optional[pd.DataFrame] = None,
        id_col_left: str = "id",
        id_col_right: Optional[str] = None,
        name_col: str = "name",
        address_col: Optional[str] = "address",
        postal_col: Optional[str] = "postal_code",
        phone_col: Optional[str] = None,
        website_col: Optional[str] = None,
        country_col: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Runs all enabled blocking strategies and returns the deduplicated union of
        candidate pairs sorted lexicographically.

        The blocker always tries the ``*_clean`` variant of each column first
        (produced by ``EntityPreprocessor``) and falls back to the raw column when
        the clean variant is absent or empty.

        Parameters
        ----------
        df_left : pd.DataFrame
            Left (or sole) entity table.
        df_right : pd.DataFrame, optional
            Right entity table for linkage.  ``None`` → deduplication mode.
        id_col_left : str
            Identifier column in ``df_left``.
        id_col_right : str, optional
            Identifier column in ``df_right`` (defaults to ``id_col_left``).
        name_col : str
            Base name column (blocker also checks ``<name_col>_clean``).
        address_col : str, optional
            Base address column (also checks ``<address_col>_clean``).
        postal_col : str, optional
            Postal / ZIP code column for exact-attribute blocking.
        phone_col : str, optional
            Phone column (digits-only clean variant) for exact-attribute blocking.
        website_col : str, optional
            Website / domain column for exact-attribute blocking.

        Returns
        -------
        List[Tuple[str, str]]
            Sorted list of unique ``(id_a, id_b)`` candidate pairs.
        """
        is_dedup = df_right is None
        if is_dedup:
            df_right = df_left
            id_col_right = id_col_left
        else:
            id_col_right = id_col_right or id_col_left

        cfg = self.config
        candidate_pairs: Set[Tuple[str, str]] = set()

        # 1. Token-prefix blocking on business name
        if cfg.use_token_prefix_blocking:
            candidate_pairs.update(
                self._token_prefix_block(
                    df_left, df_right, id_col_left, id_col_right,
                    name_col, is_dedup,
                )
            )

        # 2. Sorted Neighborhood Method on normalised name
        if cfg.use_sorted_neighborhood_blocking:
            candidate_pairs.update(
                self._sorted_neighborhood_block(
                    df_left, df_right, id_col_left, id_col_right,
                    name_col, is_dedup,
                )
            )

        # 3. Street-number blocking on normalised address
        if cfg.use_address_street_number_blocking and address_col:
            candidate_pairs.update(
                self._street_number_block(
                    df_left, df_right, id_col_left, id_col_right,
                    address_col, is_dedup,
                )
            )

        # 4. Exact-attribute blocking on secondary keys
        if cfg.use_exact_attribute_blocking:
            for attr_col in filter(None, [postal_col, phone_col, website_col, country_col]):
                if attr_col in df_left.columns:
                    candidate_pairs.update(
                        self._exact_attribute_block(
                            df_left, df_right, id_col_left, id_col_right,
                            attr_col, is_dedup,
                        )
                    )

        return sorted(candidate_pairs)

    def stream_candidate_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: Optional[pd.DataFrame] = None,
        id_col_left: str = "id",
        id_col_right: Optional[str] = None,
        name_col: str = "name",
        address_col: Optional[str] = "address",
        postal_col: Optional[str] = "postal_code",
        phone_col: Optional[str] = None,
        website_col: Optional[str] = None,
        country_col: Optional[str] = None,
    ):
        """Generator version of ``generate_candidate_pairs_from_tables``.

        Yields ``(id_a, id_b)`` tuples strategy-by-strategy, freeing each
        strategy's result set before the next strategy runs.  This avoids
        accumulating the full candidate set in memory — critical for tables
        with millions of rows.

        Pairs may be yielded more than once (no cross-strategy dedup).  The
        caller is responsible for deduplication if needed; for reservoir
        sampling, duplicates are harmless.

        Existing method ``generate_candidate_pairs_from_tables`` is unchanged.
        """
        is_dedup = df_right is None
        if is_dedup:
            df_right = df_left
            id_col_right = id_col_left
        else:
            id_col_right = id_col_right or id_col_left

        cfg = self.config

        if cfg.use_token_prefix_blocking:
            yield from self._stream_token_prefix_block(
                df_left, df_right, id_col_left, id_col_right, name_col, is_dedup,
            )

        if cfg.use_sorted_neighborhood_blocking:
            yield from self._stream_sorted_neighborhood_block(
                df_left, df_right, id_col_left, id_col_right, name_col, is_dedup,
            )

        if cfg.use_address_street_number_blocking and address_col:
            yield from self._stream_street_number_block(
                df_left, df_right, id_col_left, id_col_right, address_col, is_dedup,
            )

        if cfg.use_exact_attribute_blocking:
            for attr_col in filter(None, [postal_col, phone_col, website_col, country_col]):
                if attr_col in df_left.columns:
                    batch = self._exact_attribute_block(
                        df_left, df_right, id_col_left, id_col_right, attr_col, is_dedup,
                    )
                    yield from batch
                    del batch

    def _stream_token_prefix_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        name_col: str,
        is_dedup: bool,
    ):
        """True streaming version of token-prefix blocking using fast itertuples."""
        cfg = self.config
        prefix_len = cfg.prefix_length
        min_tok_len = cfg.min_token_length
        max_cluster = cfg.max_cluster_size
        clean_col = f"{name_col}_clean"

        cache_key = ("token_prefix", id(df_left), len(df_left), prefix_len, min_tok_len, name_col)
        if not is_dedup and cache_key in self._left_cache:
            index = self._left_cache[cache_key]
        else:
            index = defaultdict(list)
            for row in df_left.itertuples(index=False):
                rec_id = _row_id(row, id_left)
                name_val = _preferred_col(row, clean_col, name_col)
                keys = {
                    tok[:prefix_len]
                    for tok in name_val.split()
                    if len(tok) >= min_tok_len
                }
                for key in keys:
                    index[key].append(rec_id)
            if not is_dedup:
                self._left_cache[cache_key] = index

        if is_dedup:
            for ids in index.values():
                unique_ids = list(dict.fromkeys(ids))
                if 2 <= len(unique_ids) <= max_cluster:
                    for a, b in combinations(unique_ids, 2):
                        yield _ordered_pair(a, b)
        else:
            for row in df_right.itertuples(index=False):
                r_id = _row_id(row, id_right)
                name_val = _preferred_col(row, clean_col, name_col)
                keys = {
                    tok[:prefix_len]
                    for tok in name_val.split()
                    if len(tok) >= min_tok_len
                }
                for key in keys:
                    cluster = index.get(key, [])
                    if len(cluster) <= max_cluster:
                        for l_id in cluster:
                            if l_id != r_id:
                                yield (l_id, r_id)

    def _stream_street_number_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        address_col: str,
        is_dedup: bool,
    ):
        """True streaming version of street-number blocking using fast itertuples."""
        cfg = self.config
        max_cluster = cfg.max_cluster_size
        clean_col = f"{address_col}_clean"

        cache_key = ("street_number", id(df_left), len(df_left), address_col)
        if not is_dedup and cache_key in self._left_cache:
            index = self._left_cache[cache_key]
        else:
            index = defaultdict(list)
            for row in df_left.itertuples(index=False):
                rec_id = _row_id(row, id_left)
                addr = _preferred_col(row, clean_col, address_col)
                num = _street_number(addr)
                if num and len(num) >= 2:
                    index[num].append(rec_id)
            if not is_dedup:
                self._left_cache[cache_key] = index

        if is_dedup:
            for ids in index.values():
                unique_ids = list(dict.fromkeys(ids))
                if 2 <= len(unique_ids) <= max_cluster:
                    for a, b in combinations(unique_ids, 2):
                        yield _ordered_pair(a, b)
        else:
            for row in df_right.itertuples(index=False):
                r_id = _row_id(row, id_right)
                addr = _preferred_col(row, clean_col, address_col)
                num = _street_number(addr)
                if num and len(num) >= 2:
                    cluster = index.get(num, [])
                    if len(cluster) <= max_cluster:
                        for l_id in cluster:
                            if l_id != r_id:
                                yield (l_id, r_id)

    def _stream_sorted_neighborhood_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        name_col: str,
        is_dedup: bool,
    ):
        """Streaming sorted-neighborhood blocking using fast itertuples."""
        window = self.config.sorted_neighborhood_window
        clean_col = f"{name_col}_clean"

        if is_dedup:
            rows = [
                (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_left))
                for row in df_left.itertuples(index=False)
                if _preferred_col(row, clean_col, name_col)
            ]
            rows.sort(key=lambda x: x[0])
            n = len(rows)
            for i in range(n):
                for j in range(i + 1, min(i + window + 1, n)):
                    if rows[i][0][:1] != rows[j][0][:1]:
                        continue
                    a, b = rows[i][1], rows[j][1]
                    if a != b:
                        yield _ordered_pair(a, b)
        else:
            cache_key = ("snm_left_rows", id(df_left), len(df_left), name_col)
            if cache_key in self._left_cache:
                left_rows = self._left_cache[cache_key]
            else:
                left_rows = [
                    (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_left), "L")
                    for row in df_left.itertuples(index=False)
                    if _preferred_col(row, clean_col, name_col)
                ]
                left_rows.sort(key=lambda x: x[0])
                self._left_cache[cache_key] = left_rows

            right_rows = [
                (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_right), "R")
                for row in df_right.itertuples(index=False)
                if _preferred_col(row, clean_col, name_col)
            ]
            right_rows.sort(key=lambda x: x[0])
            combined = sorted(left_rows + right_rows, key=lambda x: x[0])
            n = len(combined)
            for i in range(n):
                for j in range(i + 1, min(i + window + 1, n)):
                    rec_i, rec_j = combined[i], combined[j]
                    if rec_i[2] != rec_j[2]:
                        l_id = rec_i[1] if rec_i[2] == "L" else rec_j[1]
                        r_id = rec_j[1] if rec_j[2] == "R" else rec_i[1]
                        yield (l_id, r_id)

    # ------------------------------------------------------------------
    # Strategy 1 — Token-prefix blocking
    # ------------------------------------------------------------------

    def _token_prefix_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        name_col: str,
        is_dedup: bool,
    ) -> Set[Tuple[str, str]]:
        """
        Indexes left records by the ``prefix_length``-character prefix of each
        significant token in the (clean) business name.  A token must be at least
        ``min_token_length`` characters long to be used as a blocking key.

        Clusters larger than ``max_cluster_size`` are ignored (stop-key pruning).
        """
        cfg = self.config
        prefix_len = cfg.prefix_length
        min_tok_len = cfg.min_token_length
        max_cluster = cfg.max_cluster_size
        clean_col = f"{name_col}_clean"

        # Build inverted index: prefix_key -> [id, ...]
        index: Dict[str, List[str]] = defaultdict(list)
        for row in df_left.itertuples(index=False):
            rec_id = _row_id(row, id_left)
            name_val = _preferred_col(row, clean_col, name_col)
            keys = {
                tok[:prefix_len]
                for tok in name_val.split()
                if len(tok) >= min_tok_len
            }
            for k in keys:
                index[k].append(rec_id)

        pairs: Set[Tuple[str, str]] = set()

        if is_dedup:
            for ids in index.values():
                unique_ids = list(dict.fromkeys(ids))  # deduplicate, preserve order
                if 2 <= len(unique_ids) <= max_cluster:
                    for a, b in combinations(unique_ids, 2):
                        pairs.add(_ordered_pair(a, b))
        else:
            for row in df_right.itertuples(index=False):
                r_id = _row_id(row, id_right)
                name_val = _preferred_col(row, clean_col, name_col)
                keys = {
                    tok[:prefix_len]
                    for tok in name_val.split()
                    if len(tok) >= min_tok_len
                }
                for k in keys:
                    cluster = index.get(k, [])
                    if len(cluster) <= max_cluster:
                        for l_id in cluster:
                            if l_id != r_id:
                                pairs.add((l_id, r_id))

        return pairs

    # ------------------------------------------------------------------
    # Strategy 2 — Sorted Neighborhood Method
    # ------------------------------------------------------------------

    def _sorted_neighborhood_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        name_col: str,
        is_dedup: bool,
    ) -> Set[Tuple[str, str]]:
        """
        Sorts records by their normalised name sort key and slides a window of
        ``sorted_neighborhood_window`` over the sorted list, pairing every record
        with its ``window`` nearest neighbours in sort order.

        This is robust to minor prefix differences and transpositions that
        token-prefix blocking would miss (e.g. "Acme Co" vs "ACME Company").
        """
        window = self.config.sorted_neighborhood_window
        clean_col = f"{name_col}_clean"
        pairs: Set[Tuple[str, str]] = set()

        if is_dedup:
            rows = [
                (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_left))
                for row in df_left.itertuples(index=False)
                if _preferred_col(row, clean_col, name_col)
            ]
            rows.sort(key=lambda x: x[0])
            n = len(rows)
            for i in range(n):
                for j in range(i + 1, min(i + window + 1, n)):
                    # Keep deduplication candidates within the same leading
                    # character bucket. This avoids pairing unrelated records
                    # merely because a small dataset boundary puts them inside
                    # the SNM window, while preserving nearby name variants.
                    if rows[i][0][:1] != rows[j][0][:1]:
                        continue
                    a, b = rows[i][1], rows[j][1]
                    if a != b:
                        pairs.add(_ordered_pair(a, b))
        else:
            left_rows = [
                (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_left), "L")
                for row in df_left.itertuples(index=False)
                if _preferred_col(row, clean_col, name_col)
            ]
            right_rows = [
                (_name_sort_key(_preferred_col(row, clean_col, name_col)), _row_id(row, id_right), "R")
                for row in df_right.itertuples(index=False)
                if _preferred_col(row, clean_col, name_col)
            ]
            combined = sorted(left_rows + right_rows, key=lambda x: x[0])
            n = len(combined)
            for i in range(n):
                for j in range(i + 1, min(i + window + 1, n)):
                    rec_i, rec_j = combined[i], combined[j]
                    if rec_i[2] != rec_j[2]:
                        l_id = rec_i[1] if rec_i[2] == "L" else rec_j[1]
                        r_id = rec_j[1] if rec_j[2] == "R" else rec_i[1]
                        pairs.add((l_id, r_id))

        return pairs

    # ------------------------------------------------------------------
    # Strategy 3 — Street-number blocking
    # ------------------------------------------------------------------

    def _street_number_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        address_col: str,
        is_dedup: bool,
    ) -> Set[Tuple[str, str]]:
        """
        Extracts the leading street number from the (clean) address and uses it
        as a blocking key.  Two records at "100 Main St" and "100 Market Ave"
        share street number "100" and become candidates.

        Records with no address or no leading street number are silently skipped,
        so missing addresses do not cause errors.
        """
        cfg = self.config
        max_cluster = cfg.max_cluster_size
        clean_col = f"{address_col}_clean"

        index: Dict[str, List[str]] = defaultdict(list)
        for row in df_left.itertuples(index=False):
            rec_id = _row_id(row, id_left)
            addr = _preferred_col(row, clean_col, address_col)
            num = _street_number(addr)
            if num and len(num) >= 2:  # ignore trivially common 1-digit numbers
                index[num].append(rec_id)

        pairs: Set[Tuple[str, str]] = set()

        if is_dedup:
            for ids in index.values():
                unique_ids = list(dict.fromkeys(ids))
                if 2 <= len(unique_ids) <= max_cluster:
                    for a, b in combinations(unique_ids, 2):
                        pairs.add(_ordered_pair(a, b))
        else:
            for row in df_right.itertuples(index=False):
                r_id = _row_id(row, id_right)
                addr = _preferred_col(row, clean_col, address_col)
                num = _street_number(addr)
                if num and len(num) >= 2:
                    cluster = index.get(num, [])
                    if len(cluster) <= max_cluster:
                        for l_id in cluster:
                            if l_id != r_id:
                                pairs.add((l_id, r_id))

        return pairs

    # ------------------------------------------------------------------
    # Strategy 4 — Exact secondary-attribute blocking
    # ------------------------------------------------------------------

    def _exact_attribute_block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_left: str,
        id_right: str,
        attr_col: str,
        is_dedup: bool,
    ) -> Set[Tuple[str, str]]:
        """
        Exact-match blocking on a secondary attribute column (postal code, phone,
        website domain, etc.).  Attribute values shorter than
        ``min_attribute_value_length`` are ignored to avoid overly broad clusters.
        """
        cfg = self.config
        min_len = cfg.min_attribute_value_length
        max_cluster = cfg.max_cluster_size

        # Prefer the clean variant if it exists
        clean_col = f"{attr_col}_clean"

        cache_key = ("exact_attr", id(df_left), len(df_left), attr_col, min_len)
        if not is_dedup and cache_key in self._left_cache:
            index = self._left_cache[cache_key]
        else:
            index = defaultdict(list)
            for row in df_left.itertuples(index=False):
                val = _preferred_col(row, clean_col, attr_col)
                if val and len(val) >= min_len:
                    index[val].append(_row_id(row, id_left))
            if not is_dedup:
                self._left_cache[cache_key] = index

        pairs: Set[Tuple[str, str]] = set()

        if is_dedup:
            for ids in index.values():
                unique_ids = list(dict.fromkeys(ids))
                if 2 <= len(unique_ids) <= max_cluster:
                    for a, b in combinations(unique_ids, 2):
                        pairs.add(_ordered_pair(a, b))
        else:
            for row in df_right.itertuples(index=False):
                val = _preferred_col(row, clean_col, attr_col)
                if val and len(val) >= min_len and val in index:
                    r_id = _row_id(row, id_right)
                    cluster = index[val]
                    if len(cluster) <= max_cluster:
                        for l_id in cluster:
                            if l_id != r_id:
                                pairs.add((l_id, r_id))

        return pairs
