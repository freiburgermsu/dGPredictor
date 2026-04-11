"""Build a JSON file comparing new ModelSEED dG predictions to the old
reference predictions shipped in ModelSEEDDatabase.

Reference data: 61 sharded files in
    ../ModelSEEDDatabase/Biochemistry/Thermodynamics/dGPredictor/json_files/
each with format:
    {"rxn00001": {"R00004": {"dG_mean": ..., "dG_uncer": ...}}, ...}

New predictions:
    data/modelseed_all_reaction_dG_predictions.json

Output:
    data/modelseed_dG_comparison.json
with format:
    {
      "_summary": {...aggregate statistics...},
      "reactions": {
        "rxn00001": {
          "kegg_id": "R00004",
          "old": {"dG_mean": -15.98, "dG_std": 0.10},
          "new": {"dG_mean": -15.76, "dG_std": 3.63},
          "delta_dG": 0.22,
          "abs_delta_dG": 0.22
        },
        ...
      }
    }
"""

import os
import glob
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REF_DIR = '../ModelSEEDDatabase/Biochemistry/Thermodynamics/dGPredictor/json_files'
NEW_FILE = 'data/modelseed_all_reaction_dG_predictions.json'
OUT_FILE = 'data/modelseed_dG_comparison.json'


def load_reference_predictions(ref_dir):
    """Load and merge all reference dG json shards. Skip null/NaN entries."""
    ref = {}
    files = sorted(glob.glob(os.path.join(ref_dir, 'reaction_*_dG.json')))
    logger.info(f"Loading {len(files)} reference files...")
    for f in files:
        with open(f) as fh:
            shard = json.load(fh)
        for rxn_id, payload in shard.items():
            if not isinstance(payload, dict) or not payload:
                continue
            kegg_id = next(iter(payload))
            entry = payload[kegg_id]
            if 'dG_mean' not in entry:
                continue
            try:
                dG_mean = float(entry['dG_mean'])
                dG_std = float(entry.get('dG_uncer', entry.get('dG_std', 0.0)))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(dG_mean) and np.isfinite(dG_std)):
                continue
            ref[rxn_id] = {
                'kegg_id': kegg_id,
                'dG_mean': dG_mean,
                'dG_std': dG_std,
            }
    logger.info(f"Loaded {len(ref)} reference predictions")
    return ref


def load_new_predictions(new_file):
    with open(new_file) as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(data)} new predictions")
    return data


def build_comparison(ref, new):
    comparison = {}
    for rxn_id, ref_entry in ref.items():
        if rxn_id not in new:
            continue
        new_entry = new[rxn_id]
        delta = new_entry['dG_mean'] - ref_entry['dG_mean']
        comparison[rxn_id] = {
            'kegg_id': ref_entry['kegg_id'],
            'old': {
                'dG_mean': ref_entry['dG_mean'],
                'dG_std': ref_entry['dG_std'],
            },
            'new': {
                'dG_mean': new_entry['dG_mean'],
                'dG_std': new_entry['dG_std'],
            },
            'delta_dG': delta,
            'abs_delta_dG': abs(delta),
        }
    logger.info(f"Built comparison for {len(comparison)} reactions in both sets")
    return comparison


def compute_summary(comparison, ref, new):
    deltas = np.array([c['delta_dG'] for c in comparison.values()])
    abs_deltas = np.abs(deltas)
    old_dGs = np.array([c['old']['dG_mean'] for c in comparison.values()])
    new_dGs = np.array([c['new']['dG_mean'] for c in comparison.values()])
    old_stds = np.array([c['old']['dG_std'] for c in comparison.values()])
    new_stds = np.array([c['new']['dG_std'] for c in comparison.values()])

    only_in_old = set(ref) - set(new)
    only_in_new = set(new) - set(ref)
    in_both = set(ref) & set(new)

    return {
        'n_reactions_in_both': len(in_both),
        'n_reactions_only_in_old': len(only_in_old),
        'n_reactions_only_in_new': len(only_in_new),
        'n_reference_total': len(ref),
        'n_new_total': len(new),
        'delta_dG_kJ_per_mol': {
            'mean': float(np.mean(deltas)),
            'median': float(np.median(deltas)),
            'std': float(np.std(deltas)),
            'min': float(np.min(deltas)),
            'max': float(np.max(deltas)),
        },
        'abs_delta_dG_kJ_per_mol': {
            'mean': float(np.mean(abs_deltas)),
            'median': float(np.median(abs_deltas)),
            'p90': float(np.percentile(abs_deltas, 90)),
            'p95': float(np.percentile(abs_deltas, 95)),
            'p99': float(np.percentile(abs_deltas, 99)),
            'max': float(np.max(abs_deltas)),
        },
        'agreement_buckets': {
            'within_1_kJ': int(np.sum(abs_deltas < 1)),
            'within_5_kJ': int(np.sum(abs_deltas < 5)),
            'within_10_kJ': int(np.sum(abs_deltas < 10)),
            'within_50_kJ': int(np.sum(abs_deltas < 50)),
            'within_100_kJ': int(np.sum(abs_deltas < 100)),
            'within_500_kJ': int(np.sum(abs_deltas < 500)),
            'beyond_500_kJ': int(np.sum(abs_deltas >= 500)),
        },
        'pearson_correlation_old_vs_new': float(np.corrcoef(old_dGs, new_dGs)[0, 1]),
        'std_comparison': {
            'old_std_median': float(np.median(old_stds)),
            'new_std_median': float(np.median(new_stds)),
            'old_std_mean': float(np.mean(old_stds)),
            'new_std_mean': float(np.mean(new_stds)),
        },
        'units': 'kJ/mol',
        'conditions': {'pH': 7.0, 'ionic_strength': 0.25, 'temperature_K': 298.15},
    }


def main():
    ref = load_reference_predictions(REF_DIR)
    new = load_new_predictions(NEW_FILE)
    comparison = build_comparison(ref, new)
    summary = compute_summary(comparison, ref, new)

    output = {'_summary': summary, 'reactions': comparison}
    with open(OUT_FILE, 'w') as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Saved comparison to {OUT_FILE}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Reactions in both old and new:     {summary['n_reactions_in_both']:,}")
    print(f"  Only in old (reference):           {summary['n_reactions_only_in_old']:,}")
    print(f"  Only in new (ModelSEED retrain):   {summary['n_reactions_only_in_new']:,}")
    print()
    print(f"  Pearson correlation (old vs new):  {summary['pearson_correlation_old_vs_new']:.4f}")
    print()
    print("  |Δ dG| distribution (kJ/mol):")
    for k, v in summary['abs_delta_dG_kJ_per_mol'].items():
        print(f"    {k:8s}: {v:10.2f}")
    print()
    print("  Agreement buckets (count of reactions in both):")
    for k, v in summary['agreement_buckets'].items():
        pct = 100 * v / summary['n_reactions_in_both']
        print(f"    {k:18s}: {v:6,} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
