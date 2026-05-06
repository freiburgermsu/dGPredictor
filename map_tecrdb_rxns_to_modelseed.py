"""Map TECRDB reactions to ModelSEED reaction IDs.

TECRDB reactions come as KEGG-compound equations (e.g.
"C05662 + C00003 = C00322 + C00288 + C00004") with no KEGG/ModelSEED reaction ID.
We translate each equation's compound IDs to ModelSEED IDs, then search the
ModelSEED reaction stoichiometry for a matching reaction.

Strategies, tried in order (most strict -> most permissive):
  1. exact           — same compounds + same coefficients, same direction
  2. exact_reversed  — same stoichiometry but products/reactants swapped
  3. stoich_any_dir  — same compounds + abs(coefficients) match (ignores direction)
  4. compound_set    — same compound set, ignoring H2O (cpd00001) and H+ (cpd00067)

Output: data/tecrdb_reactions_to_modelseed.json
"""

import os
import re
import json
import logging
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TECRDB_FILE = 'CC/data_cc/TECRDB.tsv'
CPD_MAP_FILE = 'data/tecrdb_to_modelseed_map.json'
MS_STOICH_FILE = 'data/modelseed_reaction_stoich.json'
OUT_FILE = 'data/tecrdb_reactions_to_modelseed.json'

PROTONATION_CPDS = {'cpd00001', 'cpd00067'}  # H2O and H+


def parse_tecrdb_equation(eq_str):
    """Parse a TECRDB equation string into {kegg_cpd: coefficient} with
    negative coefficients for reactants and positive for products.

    Examples:
      "C05662 + C00003 = C00322 + C00288 + C00004"
      "2 C00001 + C00002 = C00008 + C00009"
    """
    if '=' not in eq_str:
        return None
    left, right = eq_str.split('=', 1)

    def parse_side(side):
        out = {}
        for term in re.split(r'\s*\+\s*', side.strip()):
            term = term.strip()
            if not term:
                continue
            m = re.match(r'(?:(\d+(?:\.\d+)?)\s+)?([A-Z]\d{4,5})', term)
            if m:
                coeff = float(m.group(1)) if m.group(1) else 1.0
                cid = m.group(2)
                out[cid] = out.get(cid, 0) + coeff
        return out

    reactants = parse_side(left)
    products = parse_side(right)
    stoich = {}
    for cid, c in reactants.items():
        stoich[cid] = stoich.get(cid, 0) - c
    for cid, c in products.items():
        stoich[cid] = stoich.get(cid, 0) + c
    return stoich


def translate_to_modelseed(kegg_stoich, cpd_map):
    """Translate {kegg_cpd: coeff} to {modelseed_cpd: coeff} using the mapping."""
    ms_stoich = {}
    for kegg_id, coeff in kegg_stoich.items():
        entry = cpd_map.get(kegg_id)
        if entry is None or entry['primary'] is None:
            return None  # unmappable compound → can't translate
        ms_id = entry['primary']
        ms_stoich[ms_id] = ms_stoich.get(ms_id, 0) + coeff
    return ms_stoich


def stoich_signature(stoich, ignore_protonation=False, abs_coeff=False):
    """Normalize a stoichiometry dict into a tuple signature for comparison."""
    items = stoich.items()
    if ignore_protonation:
        items = [(k, v) for k, v in items if k not in PROTONATION_CPDS]
    if abs_coeff:
        return tuple(sorted((k, abs(v)) for k, v in items if v != 0))
    return tuple(sorted((k, v) for k, v in items if v != 0))


def build_ms_indices(ms_stoich):
    """Build three indices over ModelSEED reactions for fast matching."""
    by_exact = defaultdict(list)        # full stoichiometry (signed)
    by_abs = defaultdict(list)          # unsigned stoichiometry (direction-agnostic)
    by_cpd_set = defaultdict(list)      # compound set only, ignoring H2O/H+

    for rxn_id, cpd_stoic in ms_stoich.items():
        stoich = {k: float(v) for k, v in cpd_stoic.items()}
        sig_exact = stoich_signature(stoich)
        sig_abs = stoich_signature(stoich, abs_coeff=True)
        sig_set = tuple(sorted(set(stoich.keys()) - PROTONATION_CPDS))
        by_exact[sig_exact].append(rxn_id)
        by_abs[sig_abs].append(rxn_id)
        by_cpd_set[sig_set].append(rxn_id)

    logger.info(f"Built indices: {len(by_exact)} exact, "
                f"{len(by_abs)} direction-agnostic, {len(by_cpd_set)} compound-set")
    return by_exact, by_abs, by_cpd_set


def match_reaction(ms_rxn_stoich, by_exact, by_abs, by_cpd_set):
    """Try to match a translated TECRDB reaction to a ModelSEED reaction.

    Returns (rxn_ids, method) or ([], 'unmapped').
    """
    sig_exact = stoich_signature(ms_rxn_stoich)
    if sig_exact in by_exact:
        return by_exact[sig_exact], 'exact'

    # Try reversed direction (negate all coefficients)
    sig_rev = stoich_signature({k: -v for k, v in ms_rxn_stoich.items()})
    if sig_rev in by_exact:
        return by_exact[sig_rev], 'exact_reversed'

    # Direction-agnostic
    sig_abs = stoich_signature(ms_rxn_stoich, abs_coeff=True)
    if sig_abs in by_abs:
        return by_abs[sig_abs], 'stoich_any_dir'

    # Compound set only, ignoring protonation
    sig_set = tuple(sorted(set(ms_rxn_stoich.keys()) - PROTONATION_CPDS))
    if sig_set in by_cpd_set:
        return by_cpd_set[sig_set], 'compound_set'

    return [], 'unmapped'


def main():
    # Load inputs
    tecrdb = pd.read_csv(TECRDB_FILE, sep='\t', header=None)
    logger.info(f"Loaded {len(tecrdb)} TECRDB measurements")

    with open(CPD_MAP_FILE) as fh:
        cpd_map_full = json.load(fh)
    cpd_map = cpd_map_full['mappings']

    with open(MS_STOICH_FILE) as fh:
        ms_rxn_stoich = json.load(fh)
    logger.info(f"Loaded {len(ms_rxn_stoich)} ModelSEED reaction stoichiometries")

    # Build lookup indices
    by_exact, by_abs, by_cpd_set = build_ms_indices(ms_rxn_stoich)

    # Process TECRDB entries and group by unique reaction
    unique_rxns = {}  # signature -> {equation, measurements: [...]}
    for idx, row in tecrdb.iterrows():
        eq = row.iloc[6]
        if pd.isna(eq):
            continue
        kegg_stoich = parse_tecrdb_equation(str(eq))
        if kegg_stoich is None or not kegg_stoich:
            continue

        # Use the compound-set signature as the uniqueness key for TECRDB rxns
        key = tuple(sorted(kegg_stoich.keys()))
        if key not in unique_rxns:
            unique_rxns[key] = {
                'tecrdb_equation': str(eq),
                'tecrdb_stoichiometry': kegg_stoich,
                'measurements': [],
            }
        unique_rxns[key]['measurements'].append({
            'source_url': str(row.iloc[0]) if not pd.isna(row.iloc[0]) else None,
            'ec': str(row.iloc[4]) if not pd.isna(row.iloc[4]) else None,
            'name': str(row.iloc[5]) if not pd.isna(row.iloc[5]) else None,
            'keq_or_dG': row.iloc[8] if not pd.isna(row.iloc[8]) else None,
            'T_K': row.iloc[10] if not pd.isna(row.iloc[10]) else None,
            'pH': row.iloc[12] if not pd.isna(row.iloc[12]) else None,
        })

    logger.info(f"Unique TECRDB reactions (by compound set): {len(unique_rxns)}")

    # Map each unique reaction
    mappings = {}
    stats = {
        'exact': 0, 'exact_reversed': 0, 'stoich_any_dir': 0,
        'compound_set': 0, 'unmapped': 0, 'unmappable_compound': 0,
    }
    ambiguous_count = 0

    for i, (key, info) in enumerate(unique_rxns.items()):
        kegg_stoich = info['tecrdb_stoichiometry']

        # Translate to ModelSEED compound IDs
        ms_stoich = translate_to_modelseed(kegg_stoich, cpd_map)
        if ms_stoich is None:
            # At least one compound unmappable
            unmappable = [kid for kid in kegg_stoich
                          if kid not in cpd_map or cpd_map[kid]['primary'] is None]
            mappings[f'tecrdb_{i:04d}'] = {
                'tecrdb_equation': info['tecrdb_equation'],
                'tecrdb_stoichiometry_kegg': kegg_stoich,
                'modelseed_stoichiometry': None,
                'modelseed_reaction_ids': [],
                'method': 'unmappable_compound',
                'unmappable_kegg_compounds': unmappable,
                'n_measurements': len(info['measurements']),
            }
            stats['unmappable_compound'] += 1
            continue

        rxn_ids, method = match_reaction(ms_stoich, by_exact, by_abs, by_cpd_set)
        mappings[f'tecrdb_{i:04d}'] = {
            'tecrdb_equation': info['tecrdb_equation'],
            'tecrdb_stoichiometry_kegg': kegg_stoich,
            'modelseed_stoichiometry': ms_stoich,
            'modelseed_reaction_ids': rxn_ids,
            'primary_modelseed_id': rxn_ids[0] if rxn_ids else None,
            'method': method,
            'n_measurements': len(info['measurements']),
        }
        stats[method] += 1
        if len(rxn_ids) > 1:
            ambiguous_count += 1

    # Also build a ModelSEED-keyed inverse lookup
    ms_to_tecrdb = defaultdict(list)
    for tkey, m in mappings.items():
        for ms_id in m.get('modelseed_reaction_ids', []) or []:
            ms_to_tecrdb[ms_id].append(tkey)

    summary = {
        'n_tecrdb_measurements': int(len(tecrdb)),
        'n_unique_tecrdb_reactions': len(unique_rxns),
        'n_mapped_any_method': sum(stats[k] for k in
            ['exact', 'exact_reversed', 'stoich_any_dir', 'compound_set']),
        'n_unmapped': stats['unmapped'],
        'n_unmappable_compound': stats['unmappable_compound'],
        'n_ambiguous_multi_match': ambiguous_count,
        'n_distinct_modelseed_reactions_hit': len(ms_to_tecrdb),
        'by_method': stats,
    }

    output = {
        '_summary': summary,
        'tecrdb_to_modelseed': mappings,
        'modelseed_to_tecrdb': dict(ms_to_tecrdb),
    }

    with open(OUT_FILE, 'w') as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Saved mapping to {OUT_FILE}")

    print()
    print("=" * 60)
    print("TECRDB reactions -> ModelSEED reactions")
    print("=" * 60)
    print(f"  TECRDB measurements:                  {summary['n_tecrdb_measurements']:6,}")
    print(f"  Unique TECRDB reactions:              {summary['n_unique_tecrdb_reactions']:6,}")
    print()
    print(f"  Matched by method:")
    print(f"    exact direction & coefficients:     {stats['exact']:6,}")
    print(f"    exact reversed direction:           {stats['exact_reversed']:6,}")
    print(f"    same stoich, either direction:      {stats['stoich_any_dir']:6,}")
    print(f"    compound-set only (ignore H+/H2O):  {stats['compound_set']:6,}")
    print(f"    unmappable compound:                {stats['unmappable_compound']:6,}")
    print(f"    unmapped (no match):                {stats['unmapped']:6,}")
    print()
    total_matched = summary['n_mapped_any_method']
    pct = 100 * total_matched / summary['n_unique_tecrdb_reactions']
    print(f"  Total matched:                        {total_matched:6,} ({pct:.1f}%)")
    print(f"  Multi-match (ambiguous):              {ambiguous_count:6,}")
    print(f"  Distinct ModelSEED reactions covered: {summary['n_distinct_modelseed_reactions_hit']:6,}")


if __name__ == '__main__':
    main()
