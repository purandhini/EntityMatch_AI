"""Evaluation metrics module for Entity Resolution challenges."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EvaluationMetrics:
    """
    Computes Precision, Recall, F1, F-beta (specifically F0.5),
    and optimizes decision thresholds for entity resolution.
    """

    @staticmethod
    def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
        """
        Calculates F-beta score:
        F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
        """
        if precision + recall == 0:
            return 0.0
        beta_sq = beta ** 2
        numerator = (1.0 + beta_sq) * (precision * recall)
        denominator = (beta_sq * precision) + recall
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def compute_confusion_matrix(
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
    ) -> Dict[str, int]:
        """Calculates TP, FP, TN, FN counts."""
        yt = np.asarray(y_true).astype(int)
        yp = np.asarray(y_pred).astype(int)

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @classmethod
    def compute_metrics(
        cls,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        beta: float = 0.5,
    ) -> Dict[str, float]:
        """Computes comprehensive classification metrics."""
        cm = cls.compute_confusion_matrix(y_true, y_pred)
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        f1 = cls.calculate_f_beta(precision, recall, beta=1.0)
        f_beta = cls.calculate_f_beta(precision, recall, beta=beta)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "accuracy": round(accuracy, 6),
            "f1": round(f1, 6),
            f"f_{beta:.1f}".replace(".", "_"): round(f_beta, 6),
        }

    @classmethod
    def find_optimal_threshold(
        cls,
        y_true: Union[List[int], np.ndarray],
        y_probs: Union[List[float], np.ndarray],
        beta: float = 0.5,
        steps: int = 100,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Sweeps decision threshold from 0.01 to 0.99 to maximize F_beta (default F0.5).

        Returns:
            Tuple of (optimal_threshold, best_f_beta_score, best_metrics_dict)
        """
        yt = np.asarray(y_true).astype(int)
        yp_probs = np.asarray(y_probs).astype(float)

        thresholds = np.linspace(0.01, 0.99, steps)
        best_threshold = 0.5
        best_f_beta = -1.0
        best_metrics: Dict[str, float] = {}

        for th in thresholds:
            preds = (yp_probs >= th).astype(int)
            metrics = cls.compute_metrics(yt, preds, beta=beta)
            score = metrics[f"f_{beta:.1f}".replace(".", "_")]
            if score > best_f_beta:
                best_f_beta = score
                best_threshold = float(th)
                best_metrics = metrics

        return round(best_threshold, 4), round(best_f_beta, 6), best_metrics
