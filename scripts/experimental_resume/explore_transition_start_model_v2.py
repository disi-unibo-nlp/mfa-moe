"""Small discovery-only, family-fold prefix text feasibility diagnostic.

This never trains a production detector.  Same-model start ratings may be
noisy and the sampled nonfires have a family-spread rather than uniform design.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moe_exp.routing_control.transitions_v2 import classify_sentence_v2
from audit_transition_detector_v2 import ratings, judged


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    frame, by_uid, _ = ratings()
    output = {}
    for transition in ("candidate_to_verify", "approach_to_commit"):
        selected = [(r, judged(by_uid[r["uid"]])) for r in frame["records"]
                    if r["transition"] == transition]
        selected = [(r, v) for r, v in selected if v is not None]
        texts = [r["reader_input"]["previous_sentence"][-250:] + " [CURRENT] " +
                 r["reader_input"]["triggering_sentence"] for r, _ in selected]
        y = np.asarray([int(v["start_both"]) for _, v in selected])
        groups = [r["analysis_meta"]["family"] for r, _ in selected]
        scores = np.zeros(len(y))
        for train, test in GroupKFold(n_splits=5).split(texts, y, groups):
            model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                min_df=2, max_features=12000, sublinear_tf=True),
                LogisticRegression(max_iter=300, C=1.0, class_weight="balanced"))
            model.fit([texts[i] for i in train], y[train])
            scores[test] = model.predict_proba([texts[i] for i in test])[:, 1]
        p, recall, thresholds = precision_recall_curve(y, scores)
        rules = np.asarray([transition in classify_sentence_v2(r["reader_input"]["triggering_sentence"])
                            for r, _ in selected])
        output[transition] = {
            "n": len(y), "positives": int(y.sum()), "families": len(set(groups)),
            "group_fold_pr_auc": float(average_precision_score(y, scores)),
            "group_fold_roc_auc": float(roc_auc_score(y, scores)),
            "rule_precision": float(y[rules].mean()) if rules.any() else None,
            "rule_recall": float(rules[y == 1].mean()) if y.any() else None,
            "rule_fires": int(rules.sum()),
            "threshold_grid": [
                {"threshold": float(t), "precision": float(p[i]), "recall": float(recall[i]),
                 "fired": int((scores >= t).sum())}
                for t in (.45, .5, .55, .6, .65, .7)
                for i in [max(0, np.searchsorted(thresholds, t, side="left")-1)]
            ],
        }
    print(json.dumps({"schema": "transition-prefix-text-v2-exploratory-folds",
                      "results": output,
                      "limits": "Only 619 previously inspected discovery rows; family folds do not erase protocol adaptation or LLM-rater noise; no validation or trigger qualification"},
                     indent=2))


if __name__ == "__main__":
    main()
