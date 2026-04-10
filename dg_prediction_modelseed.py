"""
dG prediction using ModelSEED compound/reaction identifiers.

This module provides the same functionality as dg_prediction.py but uses
ModelSEED IDs (cpd#####, rxn#####) natively, removing the dependency on
KEGG identifiers and enabling predictions for the full ModelSEED biochemistry.

Usage:
    from dg_prediction_modelseed import ModelSEEDdGPredictor

    predictor = ModelSEEDdGPredictor()
    dG, std = predictor.predict_reaction('rxn00001', pH=7.0, I=0.25)
    results = predictor.predict_all(pH=7.0, I=0.25)
"""

import os
import sys
import json
import gzip
import re
import logging

import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CC'))
from compound import Compound
from thermodynamic_constants import R

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
CC_DATA_DIR = os.path.join(SCRIPT_DIR, 'CC', 'data_cc')

# Compounds to skip in fingerprint calculation (zero group contribution)
SKIP_CPDS = {'cpd00067', 'cpd11640'}  # H+, H2


class ModelSEEDdGPredictor:
    """Predict reaction dG using ModelSEED identifiers."""

    def __init__(self, model_path=None, decompose_r1_path=None,
                 decompose_r2_path=None, group_names_r1_path=None,
                 group_names_r2_path=None, compound_cache_path=None,
                 rxn_stoich_path=None, smiles_path=None):

        model_path = model_path or os.path.join(MODEL_DIR, 'modelseed_M12_model_BR.pkl')
        decompose_r1_path = decompose_r1_path or os.path.join(DATA_DIR, 'modelseed_decompose_r1.json')
        decompose_r2_path = decompose_r2_path or os.path.join(DATA_DIR, 'modelseed_decompose_r2.json')
        group_names_r1_path = group_names_r1_path or os.path.join(DATA_DIR, 'modelseed_group_names_r1.txt')
        group_names_r2_path = group_names_r2_path or os.path.join(DATA_DIR, 'modelseed_group_names_r2.txt')
        compound_cache_path = compound_cache_path or os.path.join(CC_DATA_DIR, 'modelseed_compounds.json.gz')
        rxn_stoich_path = rxn_stoich_path or os.path.join(DATA_DIR, 'modelseed_reaction_stoich.json')
        smiles_path = smiles_path or os.path.join(DATA_DIR, 'modelseed_compounds.csv')

        logger.info("Loading model...")
        self.model = joblib.load(open(model_path, 'rb'))

        logger.info("Loading decomposition vectors...")
        with open(decompose_r1_path) as fh:
            self.decompose_r1 = json.load(fh)
        with open(decompose_r2_path) as fh:
            self.decompose_r2 = json.load(fh)

        logger.info("Loading group names...")
        with open(group_names_r1_path) as fh:
            self.group_names_r1 = fh.read().splitlines()
        with open(group_names_r2_path) as fh:
            self.group_names_r2 = fh.read().splitlines()

        logger.info("Loading compound cache...")
        self.compound_dict = {}
        with gzip.open(compound_cache_path, 'rt', encoding='utf-8') as fh:
            for d in json.load(fh):
                self.compound_dict[d['compound_id']] = Compound.from_json_dict(d)

        logger.info("Loading reaction stoichiometry...")
        with open(rxn_stoich_path) as fh:
            self.rxn_stoich = json.load(fh)

        logger.info("Loading compound SMILES...")
        smiles_df = pd.read_csv(smiles_path, dtype=str)
        self.cpd_smiles = dict(zip(smiles_df['id'], smiles_df['smiles']))

        # Build group name -> index mappings for efficient vector construction
        self._r1_idx = {name: i for i, name in enumerate(self.group_names_r1)}
        self._r2_idx = {name: i for i, name in enumerate(self.group_names_r2)}
        self._all_decomposed = set(self.decompose_r1.keys()) | SKIP_CPDS

        # Cache compound vectors as we compute them
        self._r1_vec_cache = {}
        self._r2_vec_cache = {}

        logger.info(f"ModelSEEDdGPredictor ready: {len(self.compound_dict)} compounds, "
                    f"{len(self.rxn_stoich)} reactions, "
                    f"{len(self.group_names_r1)}+{len(self.group_names_r2)} groups")

    def can_predict(self, rxn_id):
        """Check whether a reaction can be predicted (all compounds are decomposable)."""
        if rxn_id not in self.rxn_stoich:
            return False
        for cpd_id in self.rxn_stoich[rxn_id]:
            if cpd_id not in self._all_decomposed:
                return False
        return True

    def get_predictable_reactions(self):
        """Return list of reaction IDs that can be predicted."""
        return [rxn_id for rxn_id in self.rxn_stoich if self.can_predict(rxn_id)]

    def _cpd_to_vec(self, cpd_id, radius):
        """Get (or compute and cache) the dense group vector for a compound."""
        if radius == 1:
            cache = self._r1_vec_cache
            decompose = self.decompose_r1
            idx_map = self._r1_idx
            n = len(self.group_names_r1)
        else:
            cache = self._r2_vec_cache
            decompose = self.decompose_r2
            idx_map = self._r2_idx
            n = len(self.group_names_r2)

        if cpd_id not in cache:
            vec = np.zeros(n)
            if cpd_id in decompose:
                for group, count in decompose[cpd_id].items():
                    i = idx_map.get(group)
                    if i is not None:
                        vec[i] = count
            cache[cpd_id] = vec
        return cache[cpd_id]

    def _build_feature_vector(self, cpd_stoic):
        """Build the combined feature vector for a reaction."""
        n_r1 = len(self.group_names_r1)
        n_r2 = len(self.group_names_r2)

        rule_r1 = np.zeros(n_r1)
        rule_r2 = np.zeros(n_r2)

        for cpd_id, stoic in cpd_stoic.items():
            if cpd_id in SKIP_CPDS:
                continue
            rule_r1 += self._cpd_to_vec(cpd_id, 1) * stoic
            rule_r2 += self._cpd_to_vec(cpd_id, 2) * stoic

        pad = np.zeros(44)
        return np.concatenate([rule_r1, pad, rule_r2, pad]).reshape(1, -1)

    def _get_ddG0(self, cpd_stoic, pH, I, T):
        """Compute the pH/ionic-strength correction for a reaction."""
        ddG0 = 0.0
        for cpd_id, coeff in cpd_stoic.items():
            if cpd_id in self.compound_dict:
                ddG0 += coeff * self.compound_dict[cpd_id].transform_pH7(pH, I, T)
        return ddG0

    def predict_reaction(self, rxn_id, pH=7.0, I=0.25, T=298.15):
        """Predict dG for a single reaction by its ModelSEED ID.

        Returns:
            (dG_mean, dG_std) in kJ/mol, or (None, None) if not predictable
        """
        if rxn_id not in self.rxn_stoich:
            logger.warning(f"Reaction {rxn_id} not found in stoichiometry")
            return None, None

        cpd_stoic = self.rxn_stoich[rxn_id]
        for cpd_id in cpd_stoic:
            if cpd_id not in self._all_decomposed:
                logger.warning(f"Reaction {rxn_id}: compound {cpd_id} not decomposable")
                return None, None

        X = self._build_feature_vector(cpd_stoic)
        ymean, ystd = self.model.predict(X, return_std=True)
        ddG0 = self._get_ddG0(cpd_stoic, pH, I, T)

        return float(ymean[0] + ddG0), float(ystd[0])

    def predict_from_equation(self, equation_str, pH=7.0, I=0.25, T=298.15,
                              arrow='<=>'):
        """Predict dG from a reaction equation string using ModelSEED compound IDs.

        Example:
            predict_from_equation("1 cpd00001 + 1 cpd00012 <=> 2 cpd00009")

        Returns:
            (dG_mean, dG_std) in kJ/mol
        """
        cpd_stoic = self._parse_equation(equation_str, arrow)

        for cpd_id in cpd_stoic:
            if cpd_id not in self._all_decomposed:
                logger.warning(f"Compound {cpd_id} not decomposable")
                return None, None

        X = self._build_feature_vector(cpd_stoic)
        ymean, ystd = self.model.predict(X, return_std=True)
        ddG0 = self._get_ddG0(cpd_stoic, pH, I, T)

        return float(ymean[0] + ddG0), float(ystd[0])

    def predict_from_smiles(self, equation_str, pH=7.0, I=0.25, T=298.15,
                            arrow='<=>'):
        """Predict dG from a reaction equation using SMILES strings.

        Example:
            predict_from_smiles("1 O + 1 O=P([O-])(=O)OP(=O)([O-])[O-] <=> 2 [O-]P(=O)([O-])O")

        Returns:
            (dG_mean, dG_std) in kJ/mol
        """
        cpd_stoic = self._parse_equation(equation_str, arrow)

        # Decompose SMILES on the fly
        novel_r1 = {}
        novel_r2 = {}
        for smiles in cpd_stoic:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Cannot parse SMILES: {smiles}")
                return None, None
            mol = Chem.RemoveHs(mol)
            novel_r1[smiles] = _count_substructures(1, mol)
            novel_r2[smiles] = _count_substructures(2, mol)

        # Merge with existing decompositions temporarily
        r1_merged = dict(self.decompose_r1)
        r1_merged.update(novel_r1)
        r2_merged = dict(self.decompose_r2)
        r2_merged.update(novel_r2)

        r1_df = pd.DataFrame.from_dict(r1_merged, orient='columns').fillna(0)
        r1_df = r1_df.reindex(self.group_names_r1).fillna(0)
        r2_df = pd.DataFrame.from_dict(r2_merged, orient='columns').fillna(0)
        r2_df = r2_df.reindex(self.group_names_r2).fillna(0)

        n_r1 = len(self.group_names_r1)
        n_r2 = len(self.group_names_r2)
        rule_r1 = np.zeros(n_r1)
        rule_r2 = np.zeros(n_r2)

        for smiles, stoic in cpd_stoic.items():
            if smiles in r1_df.columns:
                rule_r1 += r1_df[smiles].values * stoic
            if smiles in r2_df.columns:
                rule_r2 += r2_df[smiles].values * stoic

        pad = np.zeros(44)
        X = np.concatenate([rule_r1, pad, rule_r2, pad]).reshape(1, -1)
        ymean, ystd = self.model.predict(X, return_std=True)

        # No pH correction for SMILES-only reactions (no compound cache data)
        return float(ymean[0]), float(ystd[0])

    def predict_all(self, pH=7.0, I=0.25, T=298.15):
        """Predict dG for all predictable ModelSEED reactions.

        Returns:
            dict of {rxn_id: {'dG_mean': float, 'dG_std': float}}
        """
        results = {}
        predictable = self.get_predictable_reactions()
        total = len(predictable)
        logger.info(f"Predicting dG for {total} reactions...")

        for idx, rxn_id in enumerate(predictable):
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"  Progress: {idx}/{total}")
            dG, std = self.predict_reaction(rxn_id, pH, I, T)
            if dG is not None:
                results[rxn_id] = {'dG_mean': dG, 'dG_std': std}

        logger.info(f"Predicted dG for {len(results)} reactions")
        return results

    @staticmethod
    def _parse_equation(equation_str, arrow='<=>'):
        """Parse '2 cpd00001 + cpd00012 <=> cpd00009' into {id: stoic}."""
        tokens = equation_str.split(arrow)
        if len(tokens) != 2:
            raise ValueError(f"Equation must contain exactly one '{arrow}': {equation_str}")

        sparse = {}
        for side, sign in [(tokens[0], -1), (tokens[1], +1)]:
            for member in re.split(r'\s+\+\s+', side.strip()):
                parts = member.split(None, 1)
                if len(parts) == 0:
                    continue
                if len(parts) == 1:
                    amount = 1.0
                    key = parts[0]
                else:
                    try:
                        amount = float(parts[0])
                        key = parts[1]
                    except ValueError:
                        amount = 1.0
                        key = member
                sparse[key] = sparse.get(key, 0) + sign * amount

        return sparse


def _count_substructures(radius, molecule):
    """Extract molecular signature at the given radius."""
    m = molecule
    smi_count = {}
    for i in range(m.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
        atoms = set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())
        if len(atoms) == 0:
            atoms = {i}
        smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms),
                                        bondsToUse=env, canonical=True)
        smi_count[smi] = smi_count.get(smi, 0) + 1
    return smi_count


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Predict dG using ModelSEED IDs')
    parser.add_argument('--rxn', type=str, help='Predict a single reaction by ID (e.g., rxn00001)')
    parser.add_argument('--equation', type=str,
                        help='Predict from equation (e.g., "1 cpd00001 + 1 cpd00012 <=> 2 cpd00009")')
    parser.add_argument('--all', action='store_true', help='Predict all feasible reactions')
    parser.add_argument('--output', type=str, help='Output JSON file for --all predictions')
    parser.add_argument('--pH', type=float, default=7.0)
    parser.add_argument('--ionic_strength', type=float, default=0.25)
    args = parser.parse_args()

    predictor = ModelSEEDdGPredictor()

    if args.rxn:
        dG, std = predictor.predict_reaction(args.rxn, pH=args.pH, I=args.ionic_strength)
        if dG is not None:
            print(f"{args.rxn}: dG = {dG:.2f} +/- {std:.2f} kJ/mol")
        else:
            print(f"{args.rxn}: cannot predict (missing compound decompositions)")

    elif args.equation:
        dG, std = predictor.predict_from_equation(args.equation, pH=args.pH, I=args.ionic_strength)
        if dG is not None:
            print(f"dG = {dG:.2f} +/- {std:.2f} kJ/mol")
        else:
            print("Cannot predict (missing compound decompositions)")

    elif args.all:
        results = predictor.predict_all(pH=args.pH, I=args.ionic_strength)
        out_path = args.output or os.path.join(DATA_DIR, 'modelseed_reaction_dG.json')
        with open(out_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved {len(results)} predictions to {out_path}")

    else:
        # Print summary
        total_rxns = len(predictor.rxn_stoich)
        predictable = len(predictor.get_predictable_reactions())
        print(f"ModelSEED dG Predictor ready:")
        print(f"  Compounds:            {len(predictor.compound_dict)}")
        print(f"  Decomposed compounds: {len(predictor.decompose_r1)}")
        print(f"  Total reactions:      {total_rxns}")
        print(f"  Predictable reactions: {predictable} ({100*predictable/total_rxns:.1f}%)")
