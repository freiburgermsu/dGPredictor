"""Compare TECRDB experimental dG measurements against new ModelSEED predictions.

For each TECRDB reaction that matched a ModelSEED reaction, extract the experimental
Keq/dG measurements, convert to dG' at standard conditions, and compare against
the predicted dG from our retrained model.

Output: data/tecrdb_vs_predicted_dG.json
"""

import os
import json
import logging
import math
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TECRDB_FILE = 'CC/data_cc/TECRDB.tsv'
RXN_MAP_FILE = 'data/tecrdb_reactions_to_modelseed.json'
PREDICTIONS_FILE = 'data/modelseed_all_reaction_dG_predictions.json'
OUT_FILE = 'data/tecrdb_vs_predicted_dG.json'

R = 8.314e-3  # kJ/(K*mol)


def keq_to_dG(keq, T=298.15):
    """Convert equilibrium constant to dG' in kJ/mol."""
    if keq is None or keq <= 0:
        return None
    return -R * T * math.log(keq)


def load_tecrdb_measurements():
    """Load TECRDB measurements and parse the dG/Keq values.

    Column layout (0-indexed):
      0: URL
      1: Source reference
      2: Method (spectrophotometry, etc.)
      3: Confidence (A/B/C/D)
      4: EC number
      5: Enzyme name
      6: Reaction equation (KEGG compound IDs)
      7: Reaction in words
      8: K' (apparent equilibrium constant) — may be NaN
      9: Keq or dG value (varies) — the main measurement column
     10: Temperature (K)
     11: Ionic strength (M)
     12: pH
     13: Buffer/notes
    """
    tecrdb = pd.read_csv(TECRDB_FILE, sep='\t', header=None)
    logger.info(f"Loaded {len(tecrdb)} TECRDB entries")

    measurements = []
    for _, row in tecrdb.iterrows():
        eq = row.iloc[6]
        if pd.isna(eq):
            continue

        keq_prime = row.iloc[8] if not pd.isna(row.iloc[8]) else None
        keq_or_val = row.iloc[9] if not pd.isna(row.iloc[9]) else None
        T = float(row.iloc[10]) if not pd.isna(row.iloc[10]) else 298.15
        pH = float(row.iloc[12]) if not pd.isna(row.iloc[12]) else None
        I = float(row.iloc[11]) if not pd.isna(row.iloc[11]) else None
        confidence = str(row.iloc[3]) if not pd.isna(row.iloc[3]) else None
        ec = str(row.iloc[4]) if not pd.isna(row.iloc[4]) else None

        # Convert K' to dG'
        dG_experimental = None
        keq_used = None
        if keq_prime is not None and isinstance(keq_prime, (int, float)):
            try:
                keq_val = float(keq_prime)
                if keq_val > 0:
                    dG_experimental = keq_to_dG(keq_val, T)
                    keq_used = keq_val
            except (ValueError, TypeError):
                pass

        # If column 8 was empty, try column 9 as Keq
        if dG_experimental is None and keq_or_val is not None:
            try:
                keq_val = float(keq_or_val)
                if keq_val > 0:
                    dG_experimental = keq_to_dG(keq_val, T)
                    keq_used = keq_val
            except (ValueError, TypeError):
                pass

        measurements.append({
            'equation': str(eq),
            'dG_experimental': dG_experimental,
            'keq': keq_used,
            'T_K': T,
            'pH': pH,
            'ionic_strength': I,
            'confidence': confidence,
            'ec': ec,
        })

    logger.info(f"Parsed {len(measurements)} measurements, "
                f"{sum(1 for m in measurements if m['dG_experimental'] is not None)} have dG values")
    return measurements


def main():
    # Load all data
    measurements = load_tecrdb_measurements()

    with open(RXN_MAP_FILE) as fh:
        rxn_map = json.load(fh)
    tecrdb_to_ms = rxn_map['tecrdb_to_modelseed']

    with open(PREDICTIONS_FILE) as fh:
        predictions = json.load(fh)
    logger.info(f"Loaded {len(predictions)} ModelSEED dG predictions")

    # Group measurements by equation (same key as the mapping uses)
    import re
    eq_to_measurements = {}
    for m in measurements:
        key = tuple(sorted(re.findall(r'C\d{5}', m['equation'])))
        if key not in eq_to_measurements:
            eq_to_measurements[key] = []
        eq_to_measurements[key].append(m)

    # Build a reverse lookup: equation key -> tecrdb mapping entry
    eq_key_to_tecrdb_id = {}
    for tkey, entry in tecrdb_to_ms.items():
        kegg_stoich = entry.get('tecrdb_stoichiometry_kegg')
        if kegg_stoich is None:
            continue
        eq_key = tuple(sorted(kegg_stoich.keys()))
        eq_key_to_tecrdb_id[eq_key] = tkey

    # For each mapped TECRDB reaction, compare experimental vs predicted
    comparisons = []
    n_with_data = 0
    n_no_prediction = 0

    for eq_key, meas_list in eq_to_measurements.items():
        tkey = eq_key_to_tecrdb_id.get(eq_key)
        if tkey is None:
            continue
        entry = tecrdb_to_ms[tkey]
        ms_rxn_ids = entry.get('modelseed_reaction_ids', [])
        method = entry.get('method', 'unmapped')
        if method in ('unmapped', 'unmappable_compound') or not ms_rxn_ids:
            continue

        # Get prediction for the primary ModelSEED reaction
        primary_id = entry.get('primary_modelseed_id', ms_rxn_ids[0])
        pred = predictions.get(primary_id)
        if pred is None:
            # Try other matched reaction IDs
            for alt_id in ms_rxn_ids:
                pred = predictions.get(alt_id)
                if pred is not None:
                    primary_id = alt_id
                    break

        if pred is None:
            n_no_prediction += 1
            continue

        # Collect valid experimental dG measurements
        valid_meas = [m for m in meas_list if m['dG_experimental'] is not None]
        if not valid_meas:
            continue

        dG_exps = [m['dG_experimental'] for m in valid_meas]
        pHs = [m['pH'] for m in valid_meas if m['pH'] is not None]
        Ts = [m['T_K'] for m in valid_meas]

        # Direction: if matched as "exact_reversed", flip experimental sign
        direction_flip = -1.0 if method == 'exact_reversed' else 1.0

        comp = {
            'tecrdb_id': tkey,
            'tecrdb_equation': entry['tecrdb_equation'],
            'modelseed_rxn_id': primary_id,
            'match_method': method,
            'predicted_dG_mean': pred['dG_mean'],
            'predicted_dG_std': pred['dG_std'],
            'experimental_dG_values': [d * direction_flip for d in dG_exps],
            'experimental_dG_mean': float(np.mean(dG_exps)) * direction_flip,
            'experimental_dG_std': float(np.std(dG_exps)) if len(dG_exps) > 1 else None,
            'n_measurements': len(valid_meas),
            'experimental_pH_range': [min(pHs), max(pHs)] if pHs else None,
            'experimental_T_range': [min(Ts), max(Ts)] if Ts else None,
            'delta_dG': pred['dG_mean'] - float(np.mean(dG_exps)) * direction_flip,
            'abs_delta_dG': abs(pred['dG_mean'] - float(np.mean(dG_exps)) * direction_flip),
        }
        comparisons.append(comp)
        n_with_data += 1

    logger.info(f"Built {n_with_data} comparisons (experimental vs predicted)")
    logger.info(f"Skipped {n_no_prediction} reactions without predictions")

    # Compute summary statistics
    if comparisons:
        deltas = np.array([c['delta_dG'] for c in comparisons])
        abs_deltas = np.abs(deltas)
        pred_vals = np.array([c['predicted_dG_mean'] for c in comparisons])
        exp_vals = np.array([c['experimental_dG_mean'] for c in comparisons])
        correlation = float(np.corrcoef(pred_vals, exp_vals)[0, 1]) if len(comparisons) > 2 else None

        summary = {
            'n_reactions_compared': len(comparisons),
            'n_measurements_total': sum(c['n_measurements'] for c in comparisons),
            'delta_dG_kJ_per_mol': {
                'mean': float(np.mean(deltas)),
                'median': float(np.median(deltas)),
                'std': float(np.std(deltas)),
                'MAE': float(np.mean(abs_deltas)),
                'RMSE': float(np.sqrt(np.mean(deltas**2))),
            },
            'abs_delta_dG_kJ_per_mol': {
                'median': float(np.median(abs_deltas)),
                'p25': float(np.percentile(abs_deltas, 25)),
                'p75': float(np.percentile(abs_deltas, 75)),
                'p90': float(np.percentile(abs_deltas, 90)),
                'p95': float(np.percentile(abs_deltas, 95)),
            },
            'pearson_correlation': correlation,
            'within_5_kJ': int(np.sum(abs_deltas < 5)),
            'within_10_kJ': int(np.sum(abs_deltas < 10)),
            'within_20_kJ': int(np.sum(abs_deltas < 20)),
            'within_50_kJ': int(np.sum(abs_deltas < 50)),
            'note': 'Experimental dG derived from K_eq via dG = -RT*ln(Keq). '
                    'Predicted dG at pH 7.0, I=0.25M, T=298.15K. '
                    'Experimental conditions vary (see per-reaction pH/T ranges).',
        }
    else:
        summary = {'n_reactions_compared': 0}

    output = {'_summary': summary, 'comparisons': comparisons}
    with open(OUT_FILE, 'w') as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Saved to {OUT_FILE}")

    # Print results
    print()
    print("=" * 60)
    print("TECRDB Experimental vs. ModelSEED Predicted dG")
    print("=" * 60)
    if comparisons:
        print(f"  Reactions compared:       {summary['n_reactions_compared']}")
        print(f"  Total measurements:       {summary['n_measurements_total']}")
        print()
        print(f"  Predicted - Experimental (kJ/mol):")
        print(f"    Mean error (bias):      {summary['delta_dG_kJ_per_mol']['mean']:8.2f}")
        print(f"    Median error:           {summary['delta_dG_kJ_per_mol']['median']:8.2f}")
        print(f"    MAE:                    {summary['delta_dG_kJ_per_mol']['MAE']:8.2f}")
        print(f"    RMSE:                   {summary['delta_dG_kJ_per_mol']['RMSE']:8.2f}")
        print()
        print(f"  Pearson correlation:      {summary['pearson_correlation']:.4f}")
        print()
        n = summary['n_reactions_compared']
        print(f"  Accuracy buckets:")
        print(f"    Within 5 kJ/mol:        {summary['within_5_kJ']:4d} ({100*summary['within_5_kJ']/n:.1f}%)")
        print(f"    Within 10 kJ/mol:       {summary['within_10_kJ']:4d} ({100*summary['within_10_kJ']/n:.1f}%)")
        print(f"    Within 20 kJ/mol:       {summary['within_20_kJ']:4d} ({100*summary['within_20_kJ']/n:.1f}%)")
        print(f"    Within 50 kJ/mol:       {summary['within_50_kJ']:4d} ({100*summary['within_50_kJ']/n:.1f}%)")
    else:
        print("  No comparisons could be made.")


if __name__ == '__main__':
    main()
