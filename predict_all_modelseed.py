"""Batch-predict dG for all feasible ModelSEED reactions.

The single-reaction predict() loop is O(N * p^2) when return_std=True
because BayesianRidge recomputes the predictive covariance per call.
Building one big (N, p) feature matrix and calling predict() once is
massively faster — sklearn computes the std for all rows simultaneously.
"""

import os
import json
import logging
import numpy as np

from dg_prediction_modelseed import ModelSEEDdGPredictor, SKIP_CPDS, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def predict_all_batch(predictor, pH=7.0, I=0.25, T=298.15, batch_size=2000):
    """Build the full feature matrix in chunks then call predict() per chunk."""
    n_r1 = len(predictor.group_names_r1)
    n_r2 = len(predictor.group_names_r2)

    predictable = predictor.get_predictable_reactions()
    n_total = len(predictable)
    logger.info(f"Featurizing {n_total} predictable reactions...")

    # Pre-compute compound vectors for all compounds used by predictable reactions
    used_compounds = set()
    for rxn_id in predictable:
        for cpd_id in predictor.rxn_stoich[rxn_id]:
            used_compounds.add(cpd_id)
    logger.info(f"  {len(used_compounds)} unique compounds across these reactions")

    for cpd_id in used_compounds:
        if cpd_id in SKIP_CPDS:
            continue
        predictor._cpd_to_vec(cpd_id, 1)
        predictor._cpd_to_vec(cpd_id, 2)
    logger.info(f"  Compound vectors cached")

    # Build feature matrix
    pad = np.zeros(44)
    X_full = np.zeros((n_total, n_r1 + 44 + n_r2 + 44), dtype=np.float64)

    for idx, rxn_id in enumerate(predictable):
        if idx % 5000 == 0 and idx > 0:
            logger.info(f"  Featurizing: {idx}/{n_total}")
        cpd_stoic = predictor.rxn_stoich[rxn_id]
        rule_r1 = np.zeros(n_r1)
        rule_r2 = np.zeros(n_r2)
        for cpd_id, stoic in cpd_stoic.items():
            if cpd_id in SKIP_CPDS:
                continue
            rule_r1 += predictor._r1_vec_cache[cpd_id] * stoic
            rule_r2 += predictor._r2_vec_cache[cpd_id] * stoic
        X_full[idx, :n_r1] = rule_r1
        X_full[idx, n_r1 + 44 : n_r1 + 44 + n_r2] = rule_r2

    logger.info(f"Feature matrix built: {X_full.shape}")
    logger.info(f"Calling model.predict() in batches of {batch_size}...")

    means = np.zeros(n_total)
    stds = np.zeros(n_total)
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        m, s = predictor.model.predict(X_full[start:end], return_std=True)
        means[start:end] = m
        stds[start:end] = s
        logger.info(f"  Predicted batch {start}–{end}")

    logger.info(f"Computing pH/ionic-strength corrections...")
    results = {}
    for idx, rxn_id in enumerate(predictable):
        cpd_stoic = predictor.rxn_stoich[rxn_id]
        ddG0 = 0.0
        for cpd_id, coeff in cpd_stoic.items():
            if cpd_id in predictor.compound_dict:
                ddG0 += coeff * predictor.compound_dict[cpd_id].transform_pH7(pH, I, T)
        results[rxn_id] = {
            'dG_mean': float(means[idx] + ddG0),
            'dG_std': float(stds[idx]),
            'dG_model_only': float(means[idx]),
            'ddG0_pH_correction': float(ddG0),
        }

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch predict dG for all ModelSEED reactions')
    parser.add_argument('--output', type=str,
                        default=os.path.join(DATA_DIR, 'modelseed_all_reaction_dG_predictions.json'))
    parser.add_argument('--pH', type=float, default=7.0)
    parser.add_argument('--ionic_strength', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=2000)
    args = parser.parse_args()

    predictor = ModelSEEDdGPredictor()
    results = predict_all_batch(predictor, pH=args.pH, I=args.ionic_strength,
                                 batch_size=args.batch_size)

    with open(args.output, 'w') as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Saved {len(results)} predictions to {args.output}")
