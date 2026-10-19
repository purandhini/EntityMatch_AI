"""Utilities that connect entity-table blocking to pairwise ML inference."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from .blocking import EntityBlocker
from .ingestion import DetectedSchema, SchemaValidator, SchemaValidationError
from .preprocessing import EntityPreprocessor

CANONICAL_ENTITY_FIELDS = [
    "id", "name", "address", "city", "state", "postal_code", "phone", "website", "country"
]


def canonicalize_entity_dataframe(
    df: pd.DataFrame,
    schema: Optional[DetectedSchema] = None,
) -> pd.DataFrame:
    """Map a single entity table's detected aliases to canonical entity columns."""
    validator = SchemaValidator()
    detected = schema or validator.inspect_headers(list(df.columns))
    if detected.dataset_format != "SINGLE_TABLE":
        raise SchemaValidationError(
            "Entity-table pipeline requires a SINGLE_TABLE TSV.",
            missing_fields=["single-table schema"],
            found_columns=list(df.columns),
        )
    validator.validate_schema(detected)

    out = df.copy()
    for canonical in CANONICAL_ENTITY_FIELDS:
        source = detected.column_mapping.get(canonical)
        if source and source != canonical:
            out[canonical] = out[source]

    if "id" not in out.columns:
        # Stable row identifiers are a fallback only; if the challenge supplies
        # an ID column, the detected ID is preserved above.
        out["id"] = [str(i) for i in range(len(out))]
    else:
        out["id"] = out["id"].astype(str)

    return out


def prepare_entity_table(
    df: pd.DataFrame,
    schema: Optional[DetectedSchema] = None,
    preprocessor: Optional[EntityPreprocessor] = None,
) -> pd.DataFrame:
    """Canonicalize and preprocess a single entity table."""
    canonical = canonicalize_entity_dataframe(df, schema=schema)
    proc = preprocessor or EntityPreprocessor()
    return proc.preprocess_dataframe(
        canonical,
        name_cols=["name"],
        address_cols=["address"],
        phone_cols=["phone"],
        website_cols=["website"],
        postal_cols=["postal_code"],
    )


def build_pairwise_dataframe_from_candidates(
    entity_df: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Materialize blocked candidate IDs into the canonical pairwise schema."""
    if "id" not in entity_df.columns:
        raise ValueError("Entity table must contain canonical 'id' before candidate materialization.")

    indexed = entity_df.copy()
    indexed["id"] = indexed["id"].astype(str)
    indexed = indexed.drop_duplicates(subset=["id"], keep="first").set_index("id", drop=False)

    rows = []
    entity_fields = [c for c in CANONICAL_ENTITY_FIELDS if c != "id"]
    for left_id, right_id in candidate_pairs:
        left_id, right_id = str(left_id), str(right_id)
        if left_id == right_id or left_id not in indexed.index or right_id not in indexed.index:
            continue
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        row = {"id_1": left_id, "id_2": right_id}
        for field in entity_fields:
            row[f"{field}_1"] = left.get(field, "")
            row[f"{field}_2"] = right.get(field, "")
        # Preserve the preprocessor's cleaned fields when available.
        for field in ["name", "address", "postal_code", "phone", "website"]:
            for side, record in [("1", left), ("2", right)]:
                clean_col = f"{field}_clean"
                row[f"{field}_{side}_clean"] = record.get(clean_col, "")
        rows.append(row)

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Amazon-challenge multi-source support
# ---------------------------------------------------------------------------

def _canonicalize_chunk(
    chunk: pd.DataFrame,
    schema,
    label: str,
    row_offset: int,
) -> pd.DataFrame:
    """Apply alias canonicalization, source tagging, and global-id assignment to one chunk."""
    out = chunk.fillna("").copy()

    for canonical in CANONICAL_ENTITY_FIELDS:
        source_col = schema.column_mapping.get(canonical)
        if source_col and source_col != canonical and source_col in out.columns:
            out[canonical] = out[source_col]

    # Preserve original entity_id
    if "entity_id" in chunk.columns:
        out["entity_id"] = chunk["entity_id"].astype(str)
    elif "id" in chunk.columns:
        out["entity_id"] = chunk["id"].astype(str)
    else:
        out["entity_id"] = [str(row_offset + i) for i in range(len(out))]

    out["source"] = label
    out["id"] = label + "_" + out["entity_id"]
    return out


def load_amazon_sources(
    source_paths: List[Union[str, Path]],
    source_labels: Optional[List[str]] = None,
    preprocessor: Optional[EntityPreprocessor] = None,
    chunksize: Optional[int] = 50_000,
) -> pd.DataFrame:
    """Load and combine multiple Amazon-challenge entity source TSV files.

    Each source file is expected to have columns:
        entity_id, business_name, business_address, country
    (and optionally other columns).

    A ``source`` column is added to record which file each entity came from, and
    the ``id`` column is set to ``<source>_<entity_id>`` to guarantee global
    uniqueness across the combined table.

    Parameters
    ----------
    source_paths:
        Ordered list of paths to source TSV files
        (e.g. train_source1.tsv, train_source2.tsv, train_source3.tsv).
    source_labels:
        Human-readable labels for each source (e.g. ["source1", "source2", "source3"]).
        Defaults to "source1", "source2", ... when omitted.
    preprocessor:
        Optional pre-constructed EntityPreprocessor.
    chunksize:
        Number of rows to read per chunk when streaming large files.
        Set to ``None`` to read each file entirely into memory (legacy behaviour).
        Default is 50 000 rows, which keeps peak RAM well under 1 GB for typical
        Amazon-challenge TSVs.

    Returns
    -------
    pd.DataFrame
        Combined, preprocessed entity table with columns:
        id, source, entity_id, name, address, country, (+ clean variants).
    """
    validator = SchemaValidator()
    proc = preprocessor or EntityPreprocessor()
    frames: List[pd.DataFrame] = []

    for idx, path in enumerate(source_paths):
        label = (source_labels[idx] if source_labels and idx < len(source_labels)
                 else f"source{idx + 1}")

        if chunksize is None:
            # Legacy: read the whole file at once
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            schema = validator.inspect_headers(list(df.columns))
            out = _canonicalize_chunk(df, schema, label, row_offset=0)
            frames.append(out)
        else:
            # Streaming: detect schema from header row, then iterate chunks
            header_df = pd.read_csv(path, sep="\t", dtype=str, nrows=0)
            schema = validator.inspect_headers(list(header_df.columns))

            row_offset = 0
            for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
                out = _canonicalize_chunk(chunk, schema, label, row_offset=row_offset)
                frames.append(out)
                row_offset += len(chunk)

    combined = pd.concat(frames, ignore_index=True)

    # Preprocess name and address columns (vectorised — no per-row memory spike)
    combined = proc.preprocess_dataframe(
        combined,
        name_cols=["name"] if "name" in combined.columns else [],
        address_cols=["address"] if "address" in combined.columns else [],
        phone_cols=["phone"] if "phone" in combined.columns else [],
        website_cols=["website"] if "website" in combined.columns else [],
        postal_cols=["postal_code"] if "postal_code" in combined.columns else [],
    )

    return combined


def load_amazon_ground_truth(
    ground_truth_path: Union[str, Path],
    source_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parse an Amazon-challenge ground-truth TSV into a long-format labeled pair table.

    Expected ground-truth format (two-column TSV, no header assumed but also
    handles a header row)::

        source1_entity_id    matched_entity_ids

    where ``matched_entity_ids`` can be a comma-separated list of IDs from
    other source tables.

    Returns a DataFrame with columns:
        left_id, right_id, label
    where ``left_id`` and ``right_id`` are raw entity IDs as they appear in
    the ground-truth file.  Callers that use ``load_amazon_sources`` should
    prefix these IDs with the appropriate source label to obtain global IDs.
    """
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, header=None).fillna("")

    # Skip a real header row by checking whether the first cell looks like a
    # column label rather than an entity ID.  We cannot use isdigit() because
    # real IDs like "S1-965667" are also non-numeric.  Instead we recognise the
    # known header words used in the Amazon challenge ("entity_id", "matched").
    if gt_df.shape[1] >= 2:
        first_cell = str(gt_df.iloc[0, 0]).strip().lower()
        _HEADER_MARKERS = {"entity_id", "matched_entity_ids", "matched", "source1_entity_id"}
        if first_cell in _HEADER_MARKERS or first_cell.endswith("entity_id"):
            gt_df = gt_df.iloc[1:].reset_index(drop=True)

    pairs: List[Dict] = []
    for _, row in gt_df.iterrows():
        left_id = str(row.iloc[0]).strip()
        matched_raw = str(row.iloc[1]).strip() if gt_df.shape[1] > 1 else ""
        if not left_id or not matched_raw:
            continue
        for right_id in matched_raw.split(","):
            right_id = right_id.strip()
            if right_id and right_id != left_id:
                pairs.append({"left_id": left_id, "right_id": right_id, "label": 1})

    return pd.DataFrame(pairs, columns=["left_id", "right_id", "label"])


def generate_cross_source_candidates(
    combined_df: pd.DataFrame,
    blocker: Optional[EntityBlocker] = None,
    source_col: str = "source",
    id_col: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate cross-source candidate pairs from a combined entity table.

    Pairs are only generated between entities from *different* sources,
    avoiding within-source deduplication pairs entirely.

    Parameters
    ----------
    combined_df:
        Output of ``load_amazon_sources`` — must have ``id``, ``source``,
        ``name``, ``address`` columns.
    blocker:
        Optional pre-configured EntityBlocker.
    source_col:
        Column name holding the source label (default ``"source"``).
    id_col:
        Column name holding the global unique id (default ``"id"``).

    Returns
    -------
    Tuple[pair_df, candidate_df]
        pair_df: pairwise DataFrame ready for feature extraction.
        candidate_df: DataFrame with columns id_1, id_2.
    """
    sources = combined_df[source_col].unique().tolist()
    if len(sources) < 2:
        raise ValueError(
            f"Cross-source blocking requires at least 2 sources; found: {sources}"
        )

    active_blocker = blocker or EntityBlocker()
    all_candidates: List[Tuple[str, str]] = []

    # Generate candidates across every pair of distinct sources
    seen_source_pairs = set()
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            key = tuple(sorted([src_a, src_b]))
            if key in seen_source_pairs:
                continue
            seen_source_pairs.add(key)

            left = combined_df[combined_df[source_col] == src_a].copy()
            right = combined_df[combined_df[source_col] == src_b].copy()

            country_col = "country" if "country" in left.columns else None
            cross_candidates = active_blocker.generate_candidate_pairs_from_tables(
                left,
                right,
                id_col_left=id_col,
                id_col_right=id_col,
                name_col="name",
                address_col="address",
                postal_col="postal_code" if "postal_code" in left.columns else None,
                phone_col="phone" if "phone" in left.columns else None,
                website_col="website" if "website" in left.columns else None,
                country_col=country_col,
            )
            all_candidates.extend(cross_candidates)

    # Deduplicate
    unique_candidates = sorted(set(all_candidates))
    pair_df = build_pairwise_dataframe_from_candidates(combined_df, unique_candidates)
    candidate_df = pd.DataFrame(unique_candidates, columns=["id_1", "id_2"])
    return pair_df, candidate_df


# ---------------------------------------------------------------------------
# Shared helpers for ground-truth ID resolution
# ---------------------------------------------------------------------------

def _build_id_map(combined_df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping from bare entity_id to global id for every row in combined_df."""
    if "entity_id" not in combined_df.columns or "id" not in combined_df.columns:
        raise ValueError(
            "combined_df must have 'entity_id' and 'id' columns. "
            "Use load_amazon_sources() to create it."
        )
    return dict(zip(combined_df["entity_id"].astype(str), combined_df["id"].astype(str)))


def _resolve_gt_pairs(
    ground_truth_df: pd.DataFrame,
    id_map: Dict[str, str],
) -> Tuple[set, int]:
    """Resolve ground-truth bare IDs to global IDs using id_map.

    Returns
    -------
    positive_global_pairs : set of sorted (id_a, id_b) tuples
    n_skipped : number of pairs that could not be resolved
    """
    positive_global_pairs: set = set()
    skipped = 0
    for _, row in ground_truth_df.iterrows():
        left_bare = str(row["left_id"]).strip()
        right_bare = str(row["right_id"]).strip()
        left_global = id_map.get(left_bare)
        right_global = id_map.get(right_bare)
        if left_global is None or right_global is None:
            skipped += 1
            continue
        positive_global_pairs.add(tuple(sorted([left_global, right_global])))
    return positive_global_pairs, skipped


def build_amazon_training_dataframe(
    combined_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    blocker: Optional[EntityBlocker] = None,
    negative_ratio: int = 5,
    positive_sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assemble a labeled pairwise training DataFrame from Amazon-challenge inputs.

    Combines the output of ``load_amazon_sources`` with the output of
    ``load_amazon_ground_truth`` to produce a DataFrame that the existing
    ``PairwiseFeatureExtractor`` and ``ModelTrainer`` can consume directly.

    Parameters
    ----------
    combined_df:
        Entity table produced by ``load_amazon_sources``.  Must contain columns
        ``id`` (global id, e.g. "source1_S1-965667") and ``entity_id`` (bare id,
        e.g. "S1-965667").
    ground_truth_df:
        Pair table produced by ``load_amazon_ground_truth``.  Must contain
        columns ``left_id``, ``right_id`` (bare entity IDs) and ``label`` (all 1).
    blocker:
        Optional pre-configured ``EntityBlocker``.  Defaults to a new instance
        with default settings.
    negative_ratio:
        Target ratio of negative to positive pairs in the returned DataFrame.
        Blocked candidates not in the ground truth are downsampled to
        ``negative_ratio * n_positives``.  Pass 0 to keep all negatives.
        Known positives are NEVER removed regardless of this setting.
    positive_sample_size:
        If set, randomly sample this many rows from ``ground_truth_df`` before
        resolving IDs.  Use to cap memory when the GT is very large (e.g. 7.6M
        pairs on the Amazon dataset).  ``None`` (default) uses all GT rows and
        preserves existing behaviour.
    random_state:
        Random seed for reproducible GT sampling and negative downsampling.

    Returns
    -------
    pd.DataFrame
        Pairwise DataFrame with all attribute columns (``name_1``, ``name_2``,
        ``address_1``, ``address_2``, … ) plus a ``label`` column (1 or 0).
        Compatible with ``PairwiseFeatureExtractor.extract_features_from_pair_df``.

    Notes
    -----
    Missing-ID handling: if a ground-truth bare ID cannot be resolved to a
    global ID in ``combined_df``, the pair is skipped and a count of skipped
    pairs is printed.  This never raises; it guarantees all returned labels are
    trustworthy.
    """
    import random as _random

    # ------------------------------------------------------------------
    # Step A0: Optionally sample GT positives to limit memory usage
    # ------------------------------------------------------------------
    if positive_sample_size is not None and positive_sample_size < len(ground_truth_df):
        ground_truth_df = ground_truth_df.sample(
            n=positive_sample_size, random_state=random_state
        ).reset_index(drop=True)
        print(
            f"[build_amazon_training_dataframe] GT sampled to "
            f"{positive_sample_size:,} rows (random_state={random_state})."
        )

    # ------------------------------------------------------------------
    # Step A: Build bare-entity-id → global-id lookup from combined_df
    # ------------------------------------------------------------------
    id_map = _build_id_map(combined_df)

    # ------------------------------------------------------------------
    # Step B: Map ground-truth bare IDs → global IDs; report missing
    # ------------------------------------------------------------------
    positive_global_pairs, skipped = _resolve_gt_pairs(ground_truth_df, id_map)

    if skipped:
        print(
            f"[build_amazon_training_dataframe] WARNING: {skipped} ground-truth pair(s) "
            f"skipped because one or both bare entity IDs were not found in combined_df."
        )
    if not positive_global_pairs:
        raise ValueError(
            "No ground-truth pairs could be resolved to global IDs. "
            "Check that source_labels in load_amazon_sources match the ID prefixes "
            "in train_ground_truth.tsv."
        )

    # ------------------------------------------------------------------
    # Step C: Reservoir-sampled negative generation
    #
    # Instead of materialising all blocked candidates and downsampling
    # afterwards (which causes OOM at 12M+ entities), we:
    #   1. Determine n_pos from the resolved positive set.
    #   2. Run blocking source-pair by source-pair.
    #   3. For each candidate that is NOT a known positive, apply
    #      Algorithm R reservoir sampling inline.
    #   4. Materialise only the reservoir (≤ n_neg_target IDs) plus all
    #      positives at the end.
    # Peak extra RAM = reservoir size * 2 strings ≈ 100 MB for 500k pairs.
    # ------------------------------------------------------------------
    n_pos = len(positive_global_pairs)
    n_neg_target = negative_ratio * n_pos if negative_ratio > 0 else None
    # n_neg_target=None means keep all negatives (legacy behaviour)

    # For large Amazon training runs, disable sorted-neighborhood blocking.
    # The SNM implementation must scan the full 12M+ row tables and is too slow
    # for the time-constrained training path. Other blocking strategies remain active.
    if blocker is not None:
        active_blocker = blocker
    else:
        from backend.config import DEFAULT_CONFIG
        import copy as _copy
        blocking_config = _copy.deepcopy(DEFAULT_CONFIG.blocking)
        blocking_config.use_sorted_neighborhood_blocking = False
        active_blocker = EntityBlocker(config=blocking_config)

    sources = combined_df["source"].unique().tolist() if "source" in combined_df.columns else []
    if len(sources) < 2:
        raise ValueError(
            "build_amazon_training_dataframe requires at least 2 sources in combined_df."
        )

    # Reservoir: list of (id_1, id_2) negative candidate tuples
    reservoir: List[Tuple[str, str]] = []
    n_neg_seen = 0  # total non-positive candidates seen so far

    rng = _random.Random(random_state)

    seen_source_pairs: set = set()
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            key = tuple(sorted([src_a, src_b]))
            if key in seen_source_pairs:
                continue
            seen_source_pairs.add(key)

            left  = combined_df[combined_df["source"] == src_a]
            right = combined_df[combined_df["source"] == src_b]
            country_col = "country" if "country" in left.columns else None

            cross_candidates = active_blocker.stream_candidate_pairs(
                left, right,
                id_col_left="id", id_col_right="id",
                name_col="name",
                address_col="address" if "address" in left.columns else None,
                postal_col="postal_code" if "postal_code" in left.columns else None,
                phone_col="phone" if "phone" in left.columns else None,
                website_col="website" if "website" in left.columns else None,
                country_col=country_col,
            )

            for pair in cross_candidates:
                canonical = tuple(sorted([str(pair[0]), str(pair[1])]))
                if canonical in positive_global_pairs:
                    continue  # positive — handled separately below

                n_neg_seen += 1
                if n_neg_target is None:
                    reservoir.append(canonical)
                elif len(reservoir) < n_neg_target:
                    reservoir.append(canonical)
                else:
                    # Algorithm R: replace a random earlier element
                    j = rng.randint(0, n_neg_seen - 1)
                    if j < n_neg_target:
                        reservoir[j] = canonical

    if n_pos == 0 and len(reservoir) == 0:
        raise ValueError(
            "Blocking produced no candidate pairs. The combined entity table may be "
            "too small or blocking thresholds too strict."
        )

    # ------------------------------------------------------------------
    # Step D: Materialise positives + reservoir negatives
    # ------------------------------------------------------------------
    # Positives: look up attributes directly from combined_df
    positive_list = list(positive_global_pairs)
    positive_rows = build_pairwise_dataframe_from_candidates(combined_df, positive_list)
    if not positive_rows.empty:
        positive_rows["label"] = 1

    # Negatives: materialise only the reservoir
    negative_rows = build_pairwise_dataframe_from_candidates(combined_df, reservoir)
    if not negative_rows.empty:
        negative_rows["label"] = 0

    result = pd.concat(
        [df for df in [positive_rows, negative_rows] if not df.empty],
        ignore_index=True,
    )

    print(
        f"[build_amazon_training_dataframe] "
        f"positives={len(positive_rows)}, negatives={len(negative_rows)}, "
        f"total={len(result)}, "
        f"gt_pairs_resolved={len(positive_global_pairs)}, "
        f"gt_pairs_skipped={skipped}, "
        f"neg_candidates_seen={n_neg_seen}"
    )
    return result


def evaluate_blocking_recall(
    combined_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    blocker: Optional[EntityBlocker] = None,
) -> Dict[str, object]:
    """Measure how many ground-truth positive pairs survive the blocking step.

    Blocking recall is the fraction of known positive pairs that blocking
    generates as candidates.  A value below 1.0 means the ML model will never
    see those missed positives during training or evaluation.

    Parameters
    ----------
    combined_df:
        Entity table produced by ``load_amazon_sources``.
    ground_truth_df:
        Pair table produced by ``load_amazon_ground_truth``.
    blocker:
        Optional pre-configured ``EntityBlocker``.

    Returns
    -------
    dict with keys:
        total_gt_pairs        – number of resolvable ground-truth positive pairs
        gt_pairs_in_candidates – positives found by blocking
        gt_pairs_missed       – positives missed by blocking
        blocking_recall       – gt_pairs_in_candidates / total_gt_pairs
        total_candidates      – total candidate pairs generated by blocking
        reduction_ratio       – fraction of all possible pairs pruned away
        gt_pairs_skipped      – ground-truth pairs whose IDs were not in combined_df
    """
    # --- Resolve ground-truth IDs ---
    id_map = _build_id_map(combined_df)
    positive_global_pairs, skipped = _resolve_gt_pairs(ground_truth_df, id_map)
    n_total_gt = len(positive_global_pairs)

    if n_total_gt == 0:
        return {
            "total_gt_pairs": 0,
            "gt_pairs_in_candidates": 0,
            "gt_pairs_missed": 0,
            "blocking_recall": 0.0,
            "total_candidates": 0,
            "reduction_ratio": 0.0,
            "gt_pairs_skipped": skipped,
        }

    # --- Run blocking ---
    _, candidate_df = generate_cross_source_candidates(combined_df, blocker=blocker)
    candidate_set = {
        tuple(sorted([str(r["id_1"]), str(r["id_2"])]))
        for _, r in candidate_df.iterrows()
    }

    # --- Measure recall ---
    n_found = sum(1 for p in positive_global_pairs if p in candidate_set)
    n_missed = n_total_gt - n_found
    blocking_recall = n_found / n_total_gt

    # --- Reduction ratio: fraction of all possible cross-source pairs pruned ---
    n_entities = len(combined_df)
    n_all_possible = n_entities * (n_entities - 1) / 2
    reduction_ratio = (
        1.0 - len(candidate_df) / n_all_possible if n_all_possible > 0 else 0.0
    )

    return {
        "total_gt_pairs": n_total_gt,
        "gt_pairs_in_candidates": n_found,
        "gt_pairs_missed": n_missed,
        "blocking_recall": round(blocking_recall, 6),
        "total_candidates": len(candidate_df),
        "reduction_ratio": round(reduction_ratio, 6),
        "gt_pairs_skipped": skipped,
    }



def estimate_blocking_recall_from_gt(
    combined_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    config=None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Estimate blocking recall WITHOUT materializing the full candidate set.

    For each ground-truth pair, directly checks whether the existing blocking
    logic *would* generate that pair — using only O(1) lookups per pair after
    a one-time O(N log N) sorted-name-index build for SNM.

    This is safe for 12M+ entity tables where full candidate generation would
    exhaust RAM.

    Parameters
    ----------
    combined_df:
        Entity table from ``load_amazon_sources``.  Must have columns
        ``entity_id``, ``id``, ``name`` (or ``name_clean``), ``address``
        (or ``address_clean``), and optionally ``postal_code``, ``phone``,
        ``website``, ``country``.
    ground_truth_df:
        Pair table from ``load_amazon_ground_truth``.
    config:
        ``BlockingConfig`` instance.  Defaults to ``DEFAULT_CONFIG.blocking``.
    sample_size:
        If given, randomly sample this many GT pairs for a fast estimate.
        ``None`` checks every pair.  50 000 gives ±0.5 % accuracy at 95 % CI.
    random_state:
        Seed for reproducible sampling.

    Returns
    -------
    dict with keys:
        total_gt_pairs         – GT pairs checked (after sampling)
        gt_pairs_found         – pairs at least one strategy would emit
        gt_pairs_missed        – pairs no strategy would emit
        estimated_recall       – gt_pairs_found / total_gt_pairs
        sampled               – True when sample_size was applied
        sample_size_used      – actual number of pairs checked
        gt_pairs_skipped      – GT pairs whose IDs were absent from combined_df
        strategy_hits          – dict: how many pairs each strategy alone would find
    """
    import bisect
    import random as _random

    from backend.config import DEFAULT_CONFIG
    from backend.pipeline.blocking import _safe_str, _name_sort_key, _street_number

    cfg = config or DEFAULT_CONFIG.blocking

    # ------------------------------------------------------------------
    # 1. Build id_map and resolve GT pairs
    # ------------------------------------------------------------------
    id_map = _build_id_map(combined_df)
    positive_global_pairs, skipped = _resolve_gt_pairs(ground_truth_df, id_map)
    pairs_list = list(positive_global_pairs)

    # Optional sampling
    sampled = False
    if sample_size is not None and sample_size < len(pairs_list):
        rng = _random.Random(random_state)
        pairs_list = rng.sample(pairs_list, sample_size)
        sampled = True

    n_to_check = len(pairs_list)
    if n_to_check == 0:
        return {
            "total_gt_pairs": 0,
            "gt_pairs_found": 0,
            "gt_pairs_missed": 0,
            "estimated_recall": 0.0,
            "sampled": sampled,
            "sample_size_used": 0,
            "gt_pairs_skipped": skipped,
            "strategy_hits": {},
        }

    # ------------------------------------------------------------------
    # 2. Build lookup index: global_id -> entity row (as dict)
    # ------------------------------------------------------------------
    indexed = (
        combined_df.copy()
        .drop_duplicates(subset=["id"], keep="first")
        .set_index("id", drop=False)
    )

    def _row(gid: str) -> dict:
        try:
            return indexed.loc[gid].to_dict()
        except KeyError:
            return {}

    def _name_val(row: dict) -> str:
        return _safe_str(row.get("name_clean") or row.get("name", ""))

    def _addr_val(row: dict) -> str:
        return _safe_str(row.get("address_clean") or row.get("address", ""))

    # ------------------------------------------------------------------
    # 3. Build SNM sorted-name index once (O(N log N), ~500 MB for 12M rows)
    # ------------------------------------------------------------------
    snm_sorted: list = []  # list of (sort_key, global_id)
    if cfg.use_sorted_neighborhood_blocking:
        print("[estimate_blocking_recall_from_gt] Building SNM name index...", flush=True)
        snm_sorted = sorted(
            (
                (_name_sort_key(_safe_str(row.get("name_clean") or row.get("name", ""))), gid)
                for gid, row in indexed.iterrows()
                if _safe_str(row.get("name_clean") or row.get("name", ""))
            ),
            key=lambda x: x[0],
        )
        snm_keys = [t[0] for t in snm_sorted]
        snm_ids  = [t[1] for t in snm_sorted]
        id_to_snm_pos: Dict[str, int] = {gid: i for i, gid in enumerate(snm_ids)}
        window = cfg.sorted_neighborhood_window
        print(f"[estimate_blocking_recall_from_gt] SNM index built ({len(snm_sorted):,} entries).", flush=True)
    else:
        snm_keys = []
        snm_ids  = []
        id_to_snm_pos = {}
        window = 0

    # ------------------------------------------------------------------
    # 4. Per-pair strategy checks
    # ------------------------------------------------------------------
    prefix_len  = cfg.prefix_length
    min_tok_len = cfg.min_token_length
    min_attr    = cfg.min_attribute_value_length

    strategy_hits = {
        "token_prefix": 0,
        "snm": 0,
        "street_number": 0,
        "exact_attribute": 0,
    }
    found = 0

    for left_id, right_id in pairs_list:
        row_l = _row(left_id)
        row_r = _row(right_id)
        if not row_l or not row_r:
            continue

        hit = False

        # Strategy 1: token-prefix
        if cfg.use_token_prefix_blocking:
            name_l = _name_val(row_l)
            name_r = _name_val(row_r)
            prefixes_l = {
                tok[:prefix_len]
                for tok in name_l.split()
                if len(tok) >= min_tok_len
            }
            prefixes_r = {
                tok[:prefix_len]
                for tok in name_r.split()
                if len(tok) >= min_tok_len
            }
            if prefixes_l & prefixes_r:
                strategy_hits["token_prefix"] += 1
                hit = True

        # Strategy 2: SNM — check if positions are within window
        if cfg.use_sorted_neighborhood_blocking and not hit:
            pos_l = id_to_snm_pos.get(left_id)
            pos_r = id_to_snm_pos.get(right_id)
            if pos_l is not None and pos_r is not None:
                if abs(pos_l - pos_r) <= window:
                    # Also apply the leading-char guard from the real SNM impl
                    key_l = snm_keys[pos_l]
                    key_r = snm_keys[pos_r]
                    if key_l[:1] == key_r[:1]:
                        strategy_hits["snm"] += 1
                        hit = True

        # Strategy 3: street-number
        if cfg.use_address_street_number_blocking and not hit:
            num_l = _street_number(_addr_val(row_l))
            num_r = _street_number(_addr_val(row_r))
            if num_l and num_r and len(num_l) >= 2 and num_l == num_r:
                strategy_hits["street_number"] += 1
                hit = True

        # Strategy 4: exact-attribute (postal_code, phone, website, country)
        if cfg.use_exact_attribute_blocking and not hit:
            for attr in ["postal_code", "phone", "website", "country"]:
                clean = f"{attr}_clean"
                val_l = _safe_str(row_l.get(clean) or row_l.get(attr, ""))
                val_r = _safe_str(row_r.get(clean) or row_r.get(attr, ""))
                if val_l and val_r and len(val_l) >= min_attr and val_l == val_r:
                    strategy_hits["exact_attribute"] += 1
                    hit = True
                    break

        if hit:
            found += 1

    missed = n_to_check - found
    recall = found / n_to_check if n_to_check > 0 else 0.0

    return {
        "total_gt_pairs": n_to_check,
        "gt_pairs_found": found,
        "gt_pairs_missed": missed,
        "estimated_recall": round(recall, 6),
        "sampled": sampled,
        "sample_size_used": n_to_check,
        "gt_pairs_skipped": skipped,
        "strategy_hits": strategy_hits,
    }
