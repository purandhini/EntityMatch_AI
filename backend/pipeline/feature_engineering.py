"""Feature engineering module for pairwise entity similarity calculations."""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from ..config import FeatureConfig, DEFAULT_CONFIG

# Optional C-accelerated library import with pure-Python fallback
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def pure_jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Jaro-Winkler similarity (1.0 = identical).
    Accelerated with RapidFuzz when available, with pure-Python fallback.
    """
    if s1 is None or s2 is None:
        return 0.0
    str1 = str(s1)
    str2 = str(s2)
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    if HAS_RAPIDFUZZ and p == 0.1 and max_l == 4:
        return float(distance.JaroWinkler.similarity(str1, str2))

    len1, len2 = len(str1), len(str2)
    match_distance = (max(len1, len2) // 2) - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if str1[i] == str2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Transpositions
    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if str1[i] != str2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    # Jaro calculation
    jaro = (
        (matches / len1) + (matches / len2) + ((matches - transpositions) / matches)
    ) / 3.0

    # Common prefix up to max_l characters
    prefix = 0
    for i in range(min(len1, len2, max_l)):
        if str1[i] == str2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * p * (1.0 - jaro))


def pure_levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance ratio (1.0 = identical).
    Accelerated with RapidFuzz when available, with pure-Python fallback.
    """
    if s1 is None or s2 is None:
        return 0.0
    str1 = str(s1)
    str2 = str(s2)
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(distance.Levenshtein.normalized_similarity(str1, str2))

    len1, len2 = len(str1), len(str2)
    current_row = list(range(len1 + 1))
    for i in range(1, len2 + 1):
        previous_row = current_row
        current_row = [i] + [0] * len1
        for j in range(1, len1 + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1]
            if str1[j - 1] != str2[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    dist = current_row[len1]
    return 1.0 - (dist / max(len1, len2))


def token_sort_ratio(s1: str, s2: str) -> float:
    """Sorts tokens alphabetically and computes normalized Levenshtein ratio."""
    toks1 = " ".join(sorted(s1.split()))
    toks2 = " ".join(sorted(s2.split()))
    return pure_levenshtein_ratio(toks1, toks2)


def token_set_ratio(s1: str, s2: str) -> float:
    """Computes similarity of shared intersection tokens vs remaining tokens."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = " ".join(sorted(set1 & set2))
    diff1to2 = " ".join(sorted(set1 - set2))
    diff2to1 = " ".join(sorted(set2 - set1))
    t0 = intersection.strip()
    t1 = f"{intersection} {diff1to2}".strip()
    t2 = f"{intersection} {diff2to1}".strip()
    scores = [
        pure_levenshtein_ratio(t0, t1),
        pure_levenshtein_ratio(t0, t2),
        pure_levenshtein_ratio(t1, t2),
    ]
    return max(scores) if scores else 0.0


def n_gram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity over character n-grams."""
    if not s1 or not s2:
        return 0.0
    ngrams1 = {s1[i : i + n] for i in range(max(0, len(s1) - n + 1))}
    ngrams2 = {s2[i : i + n] for i in range(max(0, len(s2) - n + 1))}
    if not ngrams1 or not ngrams2:
        return 1.0 if s1 == s2 else 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


class PairwiseFeatureExtractor:
    """
    Computes fine-grained entity similarity features for candidate pairs.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features

    def compute_string_similarities(
        self, s1: Optional[str], s2: Optional[str], prefix: str = ""
    ) -> Dict[str, float]:
        """Calculates multi-metric string distances between two text attributes."""
        str1 = (str(s1) if s1 is not None and not pd.isna(s1) else "").strip().lower()
        str2 = (str(s2) if s2 is not None and not pd.isna(s2) else "").strip().lower()

        features: Dict[str, float] = {}
        tag = f"{prefix}_" if prefix else ""

        # Both missing or empty
        both_empty = (len(str1) == 0) and (len(str2) == 0)
        one_empty = (len(str1) == 0) ^ (len(str2) == 0)

        if self.config.use_missingness_indicator:
            features[f"{tag}both_missing"] = 1.0 if both_empty else 0.0
            features[f"{tag}one_missing"] = 1.0 if one_empty else 0.0

        if both_empty:
            features[f"{tag}exact_match"] = 0.0
            features[f"{tag}jaro_winkler"] = 0.0
            features[f"{tag}levenshtein"] = 0.0
            features[f"{tag}token_sort_ratio"] = 0.0
            features[f"{tag}token_set_ratio"] = 0.0
            features[f"{tag}ngram_jaccard"] = 0.0
            features[f"{tag}len_diff"] = 0.0
            features[f"{tag}len_ratio"] = 0.0
            return features

        if one_empty:
            features[f"{tag}exact_match"] = 0.0
            features[f"{tag}jaro_winkler"] = 0.0
            features[f"{tag}levenshtein"] = 0.0
            features[f"{tag}token_sort_ratio"] = 0.0
            features[f"{tag}token_set_ratio"] = 0.0
            features[f"{tag}ngram_jaccard"] = 0.0
            features[f"{tag}len_diff"] = float(abs(len(str1) - len(str2)))
            features[f"{tag}len_ratio"] = 0.0
            return features

        # Exact match
        features[f"{tag}exact_match"] = 1.0 if str1 == str2 else 0.0

        # Jaro-Winkler
        if self.config.use_jaro_winkler:
            if HAS_RAPIDFUZZ:
                features[f"{tag}jaro_winkler"] = distance.JaroWinkler.similarity(str1, str2)
            else:
                features[f"{tag}jaro_winkler"] = pure_jaro_winkler(str1, str2)

        # Normalized Levenshtein ratio
        if self.config.use_levenshtein:
            if HAS_RAPIDFUZZ:
                features[f"{tag}levenshtein"] = distance.Levenshtein.normalized_similarity(str1, str2)
            else:
                features[f"{tag}levenshtein"] = pure_levenshtein_ratio(str1, str2)

        # Token Sort and Set ratios
        if self.config.use_token_sort_ratio:
            features[f"{tag}token_sort_ratio"] = token_sort_ratio(str1, str2)
        if self.config.use_token_set_ratio:
            features[f"{tag}token_set_ratio"] = token_set_ratio(str1, str2)

        # Character N-gram Jaccard
        if self.config.use_jaccard_ngram:
            features[f"{tag}ngram_jaccard"] = n_gram_jaccard(str1, str2, n=self.config.ngram_size)

        # Length metrics
        if self.config.use_length_ratio:
            l1, l2 = len(str1), len(str2)
            features[f"{tag}len_diff"] = float(abs(l1 - l2))
            features[f"{tag}len_ratio"] = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0

        return features

    def extract_pair_features(
        self,
        record_left: Dict[str, Any],
        record_right: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extracts comprehensive pairwise feature vector for two entity records.
        """
        feats: Dict[str, float] = {}

        def _get_val(rec: Dict[str, Any], clean_key: str, raw_key: str) -> str:
            v = rec.get(clean_key)
            if v is not None and not pd.isna(v) and str(v).strip():
                return str(v).strip()
            v = rec.get(raw_key, "")
            return "" if v is None or pd.isna(v) else str(v).strip()

        # 1. Company Name similarities
        name_l = _get_val(record_left, "name_clean", "name")
        name_r = _get_val(record_right, "name_clean", "name")
        feats.update(self.compute_string_similarities(name_l, name_r, prefix="name"))

        # 2. Address similarities
        addr_l = _get_val(record_left, "address_clean", "address")
        addr_r = _get_val(record_right, "address_clean", "address")
        feats.update(self.compute_string_similarities(addr_l, addr_r, prefix="address"))

        # 3. City & State match
        city_raw_l = record_left.get("city", "")
        city_l = ("" if city_raw_l is None or pd.isna(city_raw_l) else str(city_raw_l).strip()).lower()
        city_raw_r = record_right.get("city", "")
        city_r = ("" if city_raw_r is None or pd.isna(city_raw_r) else str(city_raw_r).strip()).lower()
        feats["city_exact_match"] = 1.0 if city_l and city_l == city_r else 0.0
        if HAS_RAPIDFUZZ:
            feats["city_jaro"] = float(distance.JaroWinkler.similarity(city_l, city_r)) if city_l and city_r else 0.0
        else:
            feats["city_jaro"] = pure_jaro_winkler(city_l, city_r) if city_l and city_r else 0.0

        state_raw_l = record_left.get("state", "")
        state_l = ("" if state_raw_l is None or pd.isna(state_raw_l) else str(state_raw_l).strip()).lower()
        state_raw_r = record_right.get("state", "")
        state_r = ("" if state_raw_r is None or pd.isna(state_raw_r) else str(state_raw_r).strip()).lower()
        feats["state_exact_match"] = 1.0 if state_l and state_l == state_r else 0.0

        # 4. Postal Code exact match
        zip_l = _get_val(record_left, "postal_code_clean", "postal_code")
        zip_r = _get_val(record_right, "postal_code_clean", "postal_code")
        feats["zip_exact_match"] = 1.0 if zip_l and zip_l == zip_r else 0.0
        feats["zip_prefix_match"] = 1.0 if zip_l and zip_r and zip_l[:3] == zip_r[:3] else 0.0

        # 5. Phone match
        phone_l = _get_val(record_left, "phone_clean", "phone")
        phone_r = _get_val(record_right, "phone_clean", "phone")
        feats["phone_exact_match"] = 1.0 if phone_l and phone_l == phone_r else 0.0
        feats["phone_both_present"] = 1.0 if phone_l and phone_r else 0.0

        # 6. Website Domain match
        web_l = _get_val(record_left, "website_clean", "website")
        web_r = _get_val(record_right, "website_clean", "website")
        feats["website_exact_match"] = 1.0 if web_l and web_l == web_r else 0.0
        feats["website_both_present"] = 1.0 if web_l and web_r else 0.0

        return feats

    def extract_features_from_pair_df(
        self,
        pairs_df: pd.DataFrame,
        left_suffix: str = "_1",
        right_suffix: str = "_2",
    ) -> pd.DataFrame:
        """
        Computes feature matrix for a dataframe containing pairwise records.
        Vectorized column extraction avoids per-row pandas Series instantiation.
        """
        n_rows = len(pairs_df)
        if n_rows == 0:
            dummy = self.extract_pair_features({}, {})
            return pd.DataFrame(columns=list(dummy.keys()))

        cols = set(pairs_df.columns)

        def _get_col_data(attr: str, suf: str) -> List[str]:
            clean_candidates = [f"{attr}{suf}_clean", f"{attr}_clean{suf}"]
            clean_col = next((c for c in clean_candidates if c in cols), None)
            raw_candidates = [f"{attr}{suf}", attr]
            raw_col = next((c for c in raw_candidates if c in cols), None)

            clean_list = pairs_df[clean_col].tolist() if clean_col else None
            raw_list = pairs_df[raw_col].tolist() if raw_col else None

            if clean_list is not None and raw_list is not None:
                out = []
                for c, r in zip(clean_list, raw_list):
                    c_str = "" if (c is None or pd.isna(c)) else str(c).strip()
                    if c_str:
                        out.append(c_str)
                    else:
                        out.append("" if (r is None or pd.isna(r)) else str(r).strip())
                return out
            elif clean_list is not None:
                return ["" if (c is None or pd.isna(c)) else str(c).strip() for c in clean_list]
            elif raw_list is not None:
                return ["" if (r is None or pd.isna(r)) else str(r).strip() for r in raw_list]
            else:
                return ["" for _ in range(n_rows)]

        n1, n2 = _get_col_data("name", left_suffix), _get_col_data("name", right_suffix)
        a1, a2 = _get_col_data("address", left_suffix), _get_col_data("address", right_suffix)
        c1, c2 = _get_col_data("city", left_suffix), _get_col_data("city", right_suffix)
        s1, s2 = _get_col_data("state", left_suffix), _get_col_data("state", right_suffix)
        z1, z2 = _get_col_data("postal_code", left_suffix), _get_col_data("postal_code", right_suffix)
        p1, p2 = _get_col_data("phone", left_suffix), _get_col_data("phone", right_suffix)
        w1, w2 = _get_col_data("website", left_suffix), _get_col_data("website", right_suffix)

        rows = []
        for i in range(n_rows):
            rec_l = {
                "name": n1[i],
                "address": a1[i],
                "city": c1[i],
                "state": s1[i],
                "postal_code": z1[i],
                "phone": p1[i],
                "website": w1[i],
            }
            rec_r = {
                "name": n2[i],
                "address": a2[i],
                "city": c2[i],
                "state": s2[i],
                "postal_code": z2[i],
                "phone": p2[i],
                "website": w2[i],
            }
            rows.append(self.extract_pair_features(rec_l, rec_r))

        return pd.DataFrame(rows)
