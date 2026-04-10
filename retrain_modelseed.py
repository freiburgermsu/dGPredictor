#!/usr/bin/env python3
"""
Retrain dGPredictor using all ModelSEED biochemistry compounds.

This script:
1. Loads all compounds and reactions from the ModelSEED dev branch (sharded TSVs)
2. Builds KEGG <-> ModelSEED ID mappings from alias files
3. Constructs a compound cache with pKa data (bypassing ChemAxon)
4. Decomposes all compounds with SMILES into molecular fingerprints (radius 1 & 2)
5. Maps training reactions from KEGG IDs to ModelSEED IDs
6. Rebuilds the feature matrix with the expanded group vocabulary
7. Retrains BayesianRidge model
8. Builds stoichiometry dict for all ModelSEED reactions
9. Predicts dG for all feasible ModelSEED reactions

Usage:
    python retrain_modelseed.py --modelseed_dir ../ModelSEEDDatabase
"""

import os
import sys
import json
import glob
import gzip
import logging
import argparse
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# Add CC directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CC'))
from compound import Compound
from thermodynamic_constants import R, default_T

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
CC_DATA_DIR = os.path.join(SCRIPT_DIR, 'CC', 'data_cc')

# Output file names (ModelSEED-keyed versions)
OUT_SMILES_CSV = os.path.join(DATA_DIR, 'modelseed_compounds.csv')
OUT_DECOMPOSE_R1 = os.path.join(DATA_DIR, 'modelseed_decompose_r1.json')
OUT_DECOMPOSE_R2 = os.path.join(DATA_DIR, 'modelseed_decompose_r2.json')
OUT_GROUP_NAMES_R1 = os.path.join(DATA_DIR, 'modelseed_group_names_r1.txt')
OUT_GROUP_NAMES_R2 = os.path.join(DATA_DIR, 'modelseed_group_names_r2.txt')
OUT_RXN_STOICH = os.path.join(DATA_DIR, 'modelseed_reaction_stoich.json')
OUT_TRAINING_MAT = os.path.join(DATA_DIR, 'modelseed_training.mat')
OUT_MODEL = os.path.join(MODEL_DIR, 'modelseed_M12_model_BR.pkl')
OUT_COMPOUND_CACHE = os.path.join(CC_DATA_DIR, 'modelseed_compounds.json.gz')
OUT_KEGG_MAP = os.path.join(DATA_DIR, 'kegg_to_modelseed_compound_map.json')
OUT_RXN_DG = os.path.join(DATA_DIR, 'modelseed_reaction_dG.json')

# Special compounds that are skipped during fingerprinting / have manual params
# Mapped to their ModelSEED IDs
SPECIAL_COMPOUNDS_KEGG = {
    'C00080': 'cpd00067',  # H+
    'C00282': 'cpd11640',  # H2
    'C00087': 'cpd00074',  # S (sulfur)
    'C00237': 'cpd00204',  # CO
    'C01353': 'cpd00242',  # carbonic acid / bicarbonate
    'C00076': 'cpd00063',  # Ca2+
    'C00238': 'cpd00205',  # K+
    'C00305': 'cpd00254',  # Mg2+
    'C14818': 'cpd10515',  # Fe2+
    'C14819': 'cpd10516',  # Fe3+
    'C00138': 'cpd11620',  # reduced ferredoxin
    'C00139': 'cpd11621',  # oxidized ferredoxin
}

# These are excluded from reaction fingerprint calculation (their group contribution is 0)
SKIP_IN_FINGERPRINT = set()  # will be set to {cpd00067, cpd11640} in main


# ============================================================================
# Step 1: Load ModelSEED data
# ============================================================================

def load_modelseed_compounds(biochem_dir):
    """Load all compound shards from ModelSEED dev branch."""
    compound_files = sorted(glob.glob(os.path.join(biochem_dir, 'compound_*.tsv')))
    if not compound_files:
        raise FileNotFoundError(f"No compound_*.tsv files found in {biochem_dir}")

    frames = []
    for f in compound_files:
        df = pd.read_csv(f, sep='\t', dtype=str, na_values=['null'])
        frames.append(df)
    compounds_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(compounds_df)} compounds from {len(compound_files)} shard files")
    return compounds_df


def load_modelseed_reactions(biochem_dir):
    """Load all reaction shards from ModelSEED dev branch."""
    reaction_files = sorted(glob.glob(os.path.join(biochem_dir, 'reaction_*.tsv')))
    if not reaction_files:
        raise FileNotFoundError(f"No reaction_*.tsv files found in {biochem_dir}")

    frames = []
    for f in reaction_files:
        df = pd.read_csv(f, sep='\t', dtype=str, na_values=['null'])
        frames.append(df)
    reactions_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(reactions_df)} reactions from {len(reaction_files)} shard files")
    return reactions_df


def load_modelseed_structures(biochem_dir):
    """Load InChI strings from Unique_ModelSEED_Structures.txt.

    Dev branch format (6 columns): ID  Type  Aliases  Formula  Charge  Structure
    Master branch format (8 columns): ID  Type  Charged/Original  ExtID  Source  Formula  Charge  Structure
    """
    struct_file = os.path.join(biochem_dir, 'Structures', 'Unique_ModelSEED_Structures.txt')
    cpd_inchi = {}
    with open(struct_file) as fh:
        header = fh.readline()  # skip header
        n_cols = len(header.strip().split('\t'))
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            cpd_id, struct_type = parts[0], parts[1]
            if struct_type != 'InChI':
                continue

            if n_cols <= 6:
                # Dev branch: ID  Type  Aliases  Formula  Charge  Structure
                if len(parts) >= 6:
                    inchi_str = parts[5]
                else:
                    continue
            else:
                # Master branch: ID  Type  Charged/Orig  ExtID  Source  Formula  Charge  Structure
                if len(parts) >= 8 and parts[2] == 'Charged':
                    inchi_str = parts[7]
                else:
                    continue

            if cpd_id not in cpd_inchi:
                cpd_inchi[cpd_id] = inchi_str

    logger.info(f"Loaded InChI structures for {len(cpd_inchi)} compounds")
    return cpd_inchi


def build_kegg_to_modelseed_map(biochem_dir):
    """Build KEGG compound ID -> ModelSEED compound ID mapping from aliases."""
    alias_file = os.path.join(biochem_dir, 'Aliases', 'Unique_ModelSEED_Compound_Aliases.txt')
    kegg_to_ms = {}
    ms_to_kegg = defaultdict(list)

    with open(alias_file) as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            ms_id, ext_id, source = parts[0], parts[1], parts[2]
            if source == 'KEGG' and ext_id.startswith('C'):
                kegg_to_ms[ext_id] = ms_id
                ms_to_kegg[ms_id].append(ext_id)

    logger.info(f"Built KEGG->ModelSEED map: {len(kegg_to_ms)} KEGG IDs -> ModelSEED")
    return kegg_to_ms, dict(ms_to_kegg)


def build_kegg_to_modelseed_rxn_map(biochem_dir):
    """Build KEGG reaction ID -> ModelSEED reaction ID mapping from aliases."""
    alias_file = os.path.join(biochem_dir, 'Aliases', 'Unique_ModelSEED_Reaction_Aliases.txt')
    kegg_to_ms = {}
    with open(alias_file) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            ms_id, ext_id, source = parts[0], parts[1], parts[2]
            if source == 'KEGG' and ext_id.startswith('R'):
                kegg_to_ms[ext_id] = ms_id
    logger.info(f"Built KEGG->ModelSEED reaction map: {len(kegg_to_ms)} entries")
    return kegg_to_ms


# ============================================================================
# Step 2: Build compound cache (bypass ChemAxon)
# ============================================================================

def parse_modelseed_pka(pka_str, pkb_str):
    """Parse ModelSEED pKa/pKb format 'fragment:atom:value;...' into a list of floats.

    Both pKa (acid) and pKb (base) dissociation constants are returned as a
    single sorted (descending) list, filtered to the range (0, 14).
    """
    all_pkas = []
    for s in [pka_str, pkb_str]:
        if pd.isna(s) or s == '' or s == 'null':
            continue
        for entry in s.split(';'):
            parts = entry.strip().split(':')
            if len(parts) == 3:
                try:
                    val = float(parts[2])
                    if 0.0 < val < 14.0:
                        all_pkas.append(val)
                except ValueError:
                    continue
    return sorted(all_pkas, reverse=True)


def parse_formula_to_atom_bag(formula_str, charge_val):
    """Parse a chemical formula string (e.g., 'C10H13N5O13P3') into an atom bag dict."""
    from rdkit.Chem import rdchem
    periodic_table = rdchem.GetPeriodicTable()

    atom_bag = {}
    if pd.isna(formula_str) or formula_str == '' or formula_str == 'null':
        return atom_bag

    for atom, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula_str):
        if atom == '':
            continue
        count = int(count) if count else 1
        atom_bag[atom] = atom_bag.get(atom, 0) + count

    # Compute electron count from proton count minus charge
    try:
        n_protons = 0
        for elem, c in atom_bag.items():
            try:
                n_protons += c * periodic_table.GetAtomicNumber(str(elem))
            except RuntimeError:
                # Skip non-standard elements (R-groups, X, etc.)
                continue
        atom_bag['e-'] = n_protons - int(charge_val if not pd.isna(charge_val) else 0)
    except (ValueError, TypeError):
        pass

    return atom_bag


def build_compound_cache(compounds_df, cpd_inchi_map):
    """Build Compound objects for all ModelSEED compounds with structure data.

    Uses pKa/pKb values from ModelSEED (originally from ChemAxon) to avoid
    needing ChemAxon installed.
    """
    compound_dict = {}

    # First, handle the special compounds manually (same logic as compound.py)
    special_defs = {
        'cpd00067': {  # H+ (C00080)
            'atom_bag': {'H': 1}, 'pKas': [], 'smiles_pH7': None,
            'majorMSpH7': 0, 'nHs': [0], 'zs': [0]
        },
        'cpd11640': {  # H2 (C00282)
            'atom_bag': {'H': 2, 'e-': 2}, 'pKas': [], 'smiles_pH7': None,
            'majorMSpH7': 0, 'nHs': [2], 'zs': [0]
        },
        'cpd00074': {  # S (C00087)
            'atom_bag': {'S': 1, 'e-': 16}, 'pKas': [], 'smiles_pH7': 'S',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [0]
        },
        'cpd00204': {  # CO (C00237)
            'atom_bag': {'C': 1, 'O': 1, 'e-': 14}, 'pKas': [], 'smiles_pH7': '[C-]#[O+]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [0]
        },
        'cpd00242': {  # HCO3- / bicarbonate (C01353)
            'atom_bag': {'C': 1, 'H': 1, 'O': 3, 'e-': 32},
            'pKas': [10.33, 3.43], 'smiles_pH7': 'OC(=O)[O-]',
            'majorMSpH7': 1, 'nHs': [0, 1, 2], 'zs': [-2, -1, 0]
        },
        'cpd00063': {  # Ca2+ (C00076)
            'atom_bag': {'Ca': 1, 'e-': 18}, 'pKas': [], 'smiles_pH7': '[Ca++]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [2]
        },
        'cpd00205': {  # K+ (C00238)
            'atom_bag': {'K': 1, 'e-': 18}, 'pKas': [], 'smiles_pH7': '[K+]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [1]
        },
        'cpd00254': {  # Mg2+ (C00305)
            'atom_bag': {'Mg': 1, 'e-': 10}, 'pKas': [], 'smiles_pH7': '[Mg++]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [2]
        },
        'cpd10515': {  # Fe2+ (C14818)
            'atom_bag': {'Fe': 1, 'e-': 24}, 'pKas': [], 'smiles_pH7': '[Fe++]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [2]
        },
        'cpd10516': {  # Fe3+ (C14819)
            'atom_bag': {'Fe': 1, 'e-': 23}, 'pKas': [], 'smiles_pH7': '[Fe+++]',
            'majorMSpH7': 0, 'nHs': [0], 'zs': [3]
        },
        'cpd11620': {  # reduced ferredoxin (C00138)
            'atom_bag': {'Fe': 1, 'e-': 26}, 'pKas': [], 'smiles_pH7': None,
            'majorMSpH7': 0, 'nHs': [0], 'zs': [0]
        },
        'cpd11621': {  # oxidized ferredoxin (C00139)
            'atom_bag': {'Fe': 1, 'e-': 25}, 'pKas': [], 'smiles_pH7': None,
            'majorMSpH7': 0, 'nHs': [0], 'zs': [1]
        },
    }

    for cpd_id, params in special_defs.items():
        inchi = cpd_inchi_map.get(cpd_id, None)
        comp = Compound('ModelSEED', cpd_id, inchi,
                        params['atom_bag'], params['pKas'], params['smiles_pH7'],
                        params['majorMSpH7'], params['nHs'], params['zs'])
        compound_dict[cpd_id] = comp

    # Now process all other compounds
    n_built = 0
    n_skipped = 0
    for _, row in compounds_df.iterrows():
        cpd_id = row['id']
        if cpd_id in compound_dict:
            continue  # already handled as special

        smiles = row.get('smiles')
        if pd.isna(smiles) or smiles == '' or smiles == 'null':
            n_skipped += 1
            continue

        inchi = cpd_inchi_map.get(cpd_id, None)
        formula = row.get('formula', '')
        charge = row.get('charge', 0)
        try:
            charge = int(float(charge)) if not pd.isna(charge) else 0
        except (ValueError, TypeError):
            charge = 0

        pka_str = row.get('pka', '')
        pkb_str = row.get('pkb', '')
        pKas = parse_modelseed_pka(pka_str, pkb_str)

        # Build atom bag from formula and charge
        atom_bag = parse_formula_to_atom_bag(formula, charge)

        # Determine major microspecies index and build nHs/zs arrays
        # The SMILES in ModelSEED compounds.tsv is the charged (pH 7) form
        major_ms_nH = atom_bag.get('H', 0)
        major_ms_charge = charge

        n_species = len(pKas) + 1
        if not pKas:
            majorMSpH7 = 0
        else:
            majorMSpH7 = len([1 for pka in pKas if pka > 7])

        nHs = []
        zs = []
        for i in range(n_species):
            zs.append((i - majorMSpH7) + major_ms_charge)
            nHs.append((i - majorMSpH7) + major_ms_nH)

        comp = Compound('ModelSEED', cpd_id, inchi, atom_bag, pKas,
                        smiles, majorMSpH7, nHs, zs)
        compound_dict[cpd_id] = comp
        n_built += 1

    logger.info(f"Built compound cache: {n_built} from data, "
                f"{len(special_defs)} special, {n_skipped} skipped (no SMILES)")
    return compound_dict


def save_compound_cache(compound_dict, output_path):
    """Save compound cache as compressed JSON."""
    data = sorted(compound_dict.values(), key=lambda c: c.compound_id)
    dict_data = [c.to_json_dict() for c in data]
    with gzip.open(output_path, 'wt', encoding='utf-8') as fp:
        json.dump(dict_data, fp, sort_keys=True, indent=2)
    logger.info(f"Saved compound cache ({len(dict_data)} compounds) to {output_path}")


# ============================================================================
# Step 3: Decompose compounds into molecular fingerprints
# ============================================================================

def count_substructures(radius, molecule):
    """Extract molecular signature (substructure counts) at the given radius."""
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


def decompose_all_compounds(compounds_df, radius):
    """Decompose all compounds with SMILES at the given radius.

    Returns:
        decompose_dict: {cpd_id: {smiles_fragment: count, ...}}
        non_decomposable: list of cpd_ids that failed
    """
    decompose_dict = {}
    non_decomposable = []

    smiles_series = compounds_df.set_index('id')['smiles'].dropna()
    smiles_series = smiles_series[smiles_series != '']
    smiles_series = smiles_series[smiles_series != 'null']

    # Filter out incomplete structures (containing * attachment points or R-groups)
    incomplete_mask = smiles_series.str.contains(r'\*', na=False)
    n_incomplete = incomplete_mask.sum()
    if n_incomplete > 0:
        logger.info(f"  Skipping {n_incomplete} compounds with incomplete structures (* in SMILES)")
    smiles_series = smiles_series[~incomplete_mask]

    total = len(smiles_series)
    for idx, (cpd_id, smiles) in enumerate(smiles_series.items()):
        if idx % 5000 == 0:
            logger.info(f"  Decomposing (r={radius}): {idx}/{total}")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                non_decomposable.append(cpd_id)
                continue
            mol = Chem.RemoveHs(mol)
            smi_count = count_substructures(radius, mol)
            decompose_dict[cpd_id] = smi_count
        except Exception:
            non_decomposable.append(cpd_id)

    logger.info(f"  Decomposed {len(decompose_dict)} compounds at radius {radius}, "
                f"{len(non_decomposable)} failed")
    return decompose_dict, non_decomposable


def extract_group_names(decompose_dict):
    """Collect all unique substructure names from decomposition vectors."""
    groups = set()
    for smi_count in decompose_dict.values():
        groups.update(smi_count.keys())
    return sorted(groups)


# ============================================================================
# Step 4: Map training data to ModelSEED IDs
# ============================================================================

def load_training_reactions_from_mat(kegg_to_ms_cpd):
    """Load the 4001 training reactions from component_contribution_python.mat.

    The mat file contains:
        - train_S: (673 compounds x 4001 reactions) stoichiometric matrix
        - train_cids / cids: 673 KEGG compound IDs
        - b: (4001,) observed dG values (same as y in Test_KEGG_all_grp.mat)

    Returns:
        training_reactions: list of dicts, each {ms_cpd_id: stoic_coeff}
        y: array of dG values
        n_unmapped: count of reactions with unmappable compounds
    """
    cc_mat = loadmat(os.path.join(DATA_DIR, 'component_contribution_python.mat'))
    train_S = cc_mat['train_S']  # (673, 4001)
    train_cids = [str(c).strip() for c in cc_mat['train_cids'].flatten()]
    y = cc_mat['b'].flatten()  # (4001,)

    n_rxns = train_S.shape[1]
    training_reactions = []
    y_out = []
    n_unmapped = 0

    for rxn_idx in range(n_rxns):
        col = train_S[:, rxn_idx]
        nonzero_mask = col != 0
        if not np.any(nonzero_mask):
            continue

        cpd_stoic = {}
        mappable = True
        for cpd_idx in np.where(nonzero_mask)[0]:
            kegg_id = train_cids[cpd_idx]
            stoic = float(col[cpd_idx])

            if kegg_id in kegg_to_ms_cpd:
                ms_id = kegg_to_ms_cpd[kegg_id]
            else:
                mappable = False
                break
            cpd_stoic[ms_id] = cpd_stoic.get(ms_id, 0) + stoic

        if mappable and cpd_stoic:
            training_reactions.append(cpd_stoic)
            y_out.append(y[rxn_idx])
        else:
            n_unmapped += 1

    logger.info(f"Loaded {len(training_reactions)} training reactions from mat file "
                f"({n_unmapped} unmapped)")
    return training_reactions, np.array(y_out), n_unmapped


def _compound_to_vector(cpd_decompose, group_name_to_idx, n_groups):
    """Convert a compound's decomposition dict to a dense vector."""
    vec = np.zeros(n_groups)
    for group, count in cpd_decompose.items():
        idx = group_name_to_idx.get(group)
        if idx is not None:
            vec[idx] = count
    return vec


def build_feature_matrix(training_reactions, y, decompose_r1, decompose_r2,
                         group_names_r1, group_names_r2, skip_cpds):
    """Build the combined feature matrix X for training reactions.

    For each reaction, the feature vector is:
        [r1_group_changes | padding(44) | r2_group_changes | padding(44)]

    This matches the original dGPredictor feature layout (see get_rule in
    dg_prediction.py), keeping 44 zero-columns between r1 and r2 blocks
    and after r2 for compatibility.

    Uses a memory-efficient approach: instead of building a full
    (n_groups x n_compounds) DataFrame, we convert each compound vector
    on demand and cache only those used in training reactions.

    Returns:
        X: numpy array (n_reactions, n_features)
        y_valid: numpy array of dG values for successfully featurized reactions
        n_skipped: count of reactions skipped due to missing decompositions
    """
    n_r1 = len(group_names_r1)
    n_r2 = len(group_names_r2)

    # Build group name -> index mappings for fast lookup
    r1_idx = {name: i for i, name in enumerate(group_names_r1)}
    r2_idx = {name: i for i, name in enumerate(group_names_r2)}

    all_decomposed = set(decompose_r1.keys()) | skip_cpds

    # Cache compound vectors as we compute them
    r1_cache = {}
    r2_cache = {}

    X_rows = []
    y_valid = []
    n_skipped = 0

    for rxn_idx, cpd_stoic in enumerate(training_reactions):
        if rxn_idx % 500 == 0 and rxn_idx > 0:
            logger.info(f"  Building features: {rxn_idx}/{len(training_reactions)}")

        # Check all non-skip compounds are decomposable
        can_compute = True
        for cpd_id in cpd_stoic:
            if cpd_id in skip_cpds:
                continue
            if cpd_id not in all_decomposed:
                can_compute = False
                break
        if not can_compute:
            n_skipped += 1
            continue

        # Compute group change vectors
        rule_r1 = np.zeros(n_r1)
        rule_r2 = np.zeros(n_r2)

        for cpd_id, stoic in cpd_stoic.items():
            if cpd_id in skip_cpds:
                continue

            # Get or compute r1 vector
            if cpd_id not in r1_cache:
                if cpd_id in decompose_r1:
                    r1_cache[cpd_id] = _compound_to_vector(
                        decompose_r1[cpd_id], r1_idx, n_r1)
                else:
                    r1_cache[cpd_id] = np.zeros(n_r1)
            rule_r1 += r1_cache[cpd_id] * stoic

            # Get or compute r2 vector
            if cpd_id not in r2_cache:
                if cpd_id in decompose_r2:
                    r2_cache[cpd_id] = _compound_to_vector(
                        decompose_r2[cpd_id], r2_idx, n_r2)
                else:
                    r2_cache[cpd_id] = np.zeros(n_r2)
            rule_r2 += r2_cache[cpd_id] * stoic

        # Concatenate with padding
        pad = np.zeros(44)
        row = np.concatenate([rule_r1, pad, rule_r2, pad])
        X_rows.append(row)
        y_valid.append(y[rxn_idx])

    X = np.array(X_rows)
    y_valid = np.array(y_valid)

    logger.info(f"Built feature matrix: {X.shape} for {len(X_rows)} reactions "
                f"({n_skipped} skipped)")
    return X, y_valid, n_skipped


# ============================================================================
# Step 5: Retrain model
# ============================================================================

def retrain_model(X, y):
    """Train BayesianRidge regression on the new feature matrix."""
    model = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    logger.info(f"Model trained: MSE = {mse:.2f}, R² = {r2:.4f}")
    return model


# ============================================================================
# Step 6: Build ModelSEED reaction stoichiometry
# ============================================================================

def parse_modelseed_stoichiometry(reactions_df):
    """Parse the stoichiometry column from ModelSEED reactions into a dict.

    ModelSEED stoichiometry format:
        -1:cpd00001:0:"H2O";-1:cpd00012:0:"PPi";2:cpd00009:0:"Phosphate"

    Returns:
        {rxn_id: {cpd_id: coefficient, ...}}
    """
    rxn_stoich = {}
    n_parsed = 0
    n_failed = 0

    for _, row in reactions_df.iterrows():
        rxn_id = row['id']
        stoich_str = row.get('stoichiometry', '')
        if pd.isna(stoich_str) or stoich_str == '' or stoich_str == 'null':
            n_failed += 1
            continue

        cpd_stoic = {}
        try:
            for entry in stoich_str.split(';'):
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split(':')
                if len(parts) < 2:
                    continue
                coeff = float(parts[0])
                cpd_id = parts[1]
                cpd_stoic[cpd_id] = cpd_stoic.get(cpd_id, 0) + coeff

            if cpd_stoic:
                rxn_stoich[rxn_id] = cpd_stoic
                n_parsed += 1
        except (ValueError, IndexError):
            n_failed += 1

    logger.info(f"Parsed stoichiometry for {n_parsed} reactions ({n_failed} failed)")
    return rxn_stoich


# ============================================================================
# Step 7: Predict dG for all ModelSEED reactions
# ============================================================================

def predict_all_reactions(model, rxn_stoich, decompose_r1, decompose_r2,
                          group_names_r1, group_names_r2, compound_dict,
                          skip_cpds, pH=7.0, I=0.25, T=298.15):
    """Predict dG for all ModelSEED reactions where all compounds are decomposable."""

    n_r1 = len(group_names_r1)
    n_r2 = len(group_names_r2)

    r1_idx = {name: i for i, name in enumerate(group_names_r1)}
    r2_idx = {name: i for i, name in enumerate(group_names_r2)}

    all_decomposed = set(decompose_r1.keys()) | skip_cpds

    # Cache compound vectors
    r1_cache = {}
    r2_cache = {}

    results = {}
    n_predicted = 0
    n_skipped = 0

    total = len(rxn_stoich)
    for idx, (rxn_id, cpd_stoic) in enumerate(rxn_stoich.items()):
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Predicting: {idx}/{total}")

        # Check all compounds are decomposable
        can_predict = True
        for cpd_id in cpd_stoic:
            if cpd_id not in all_decomposed:
                can_predict = False
                break
        if not can_predict:
            n_skipped += 1
            continue

        # Build feature vector
        rule_r1 = np.zeros(n_r1)
        rule_r2 = np.zeros(n_r2)
        for cpd_id, stoic in cpd_stoic.items():
            if cpd_id in skip_cpds:
                continue

            if cpd_id not in r1_cache:
                if cpd_id in decompose_r1:
                    r1_cache[cpd_id] = _compound_to_vector(
                        decompose_r1[cpd_id], r1_idx, n_r1)
                else:
                    r1_cache[cpd_id] = np.zeros(n_r1)
            rule_r1 += r1_cache[cpd_id] * stoic

            if cpd_id not in r2_cache:
                if cpd_id in decompose_r2:
                    r2_cache[cpd_id] = _compound_to_vector(
                        decompose_r2[cpd_id], r2_idx, n_r2)
                else:
                    r2_cache[cpd_id] = np.zeros(n_r2)
            rule_r2 += r2_cache[cpd_id] * stoic

        pad = np.zeros(44)
        X = np.concatenate([rule_r1, pad, rule_r2, pad]).reshape(1, -1)

        ymean, ystd = model.predict(X, return_std=True)

        # pH / ionic strength correction
        ddG0 = 0.0
        for cpd_id, coeff in cpd_stoic.items():
            if cpd_id in compound_dict:
                comp = compound_dict[cpd_id]
                ddG0 += coeff * comp.transform_pH7(pH, I, T)

        dG = ymean[0] + ddG0
        results[rxn_id] = {
            'dG_mean': float(dG),
            'dG_std': float(ystd[0]),
            'dG_model_only': float(ymean[0]),
            'ddG0_pH_correction': float(ddG0),
        }
        n_predicted += 1

    logger.info(f"Predicted dG for {n_predicted} reactions ({n_skipped} skipped, "
                f"missing compound decompositions)")
    return results


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Retrain dGPredictor with ModelSEED compounds')
    parser.add_argument('--modelseed_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, '..', 'ModelSEEDDatabase'),
                        help='Path to ModelSEEDDatabase root')
    parser.add_argument('--skip_predict', action='store_true',
                        help='Skip the final prediction step')
    parser.add_argument('--pH', type=float, default=7.0)
    parser.add_argument('--ionic_strength', type=float, default=0.25)
    args = parser.parse_args()

    biochem_dir = os.path.join(args.modelseed_dir, 'Biochemistry')
    if not os.path.isdir(biochem_dir):
        raise FileNotFoundError(f"Biochemistry directory not found: {biochem_dir}")

    global SKIP_IN_FINGERPRINT
    SKIP_IN_FINGERPRINT = {'cpd00067', 'cpd11640'}  # H+ and H2

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ModelSEED data")
    logger.info("=" * 60)
    compounds_df = load_modelseed_compounds(biochem_dir)
    reactions_df = load_modelseed_reactions(biochem_dir)
    cpd_inchi_map = load_modelseed_structures(biochem_dir)
    kegg_to_ms_cpd, ms_to_kegg_cpd = build_kegg_to_modelseed_map(biochem_dir)
    kegg_to_ms_rxn = build_kegg_to_modelseed_rxn_map(biochem_dir)

    # Save the mapping for reference
    with open(OUT_KEGG_MAP, 'w') as fh:
        json.dump(kegg_to_ms_cpd, fh, indent=2)
    logger.info(f"Saved KEGG->ModelSEED compound map to {OUT_KEGG_MAP}")

    # Save compound SMILES CSV for reference
    cpd_smiles = compounds_df[['id', 'smiles', 'formula', 'charge']].copy()
    cpd_smiles = cpd_smiles[cpd_smiles['smiles'].notna() & (cpd_smiles['smiles'] != 'null')]
    cpd_smiles.to_csv(OUT_SMILES_CSV, index=False)
    logger.info(f"Saved {len(cpd_smiles)} compound SMILES to {OUT_SMILES_CSV}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Building compound cache")
    logger.info("=" * 60)
    compound_dict = build_compound_cache(compounds_df, cpd_inchi_map)
    save_compound_cache(compound_dict, OUT_COMPOUND_CACHE)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Decomposing compounds into molecular fingerprints")
    logger.info("=" * 60)
    logger.info("Decomposing at radius 1...")
    decompose_r1, failed_r1 = decompose_all_compounds(compounds_df, radius=1)
    logger.info("Decomposing at radius 2...")
    decompose_r2, failed_r2 = decompose_all_compounds(compounds_df, radius=2)

    # Extract and save group names
    group_names_r1 = extract_group_names(decompose_r1)
    group_names_r2 = extract_group_names(decompose_r2)
    logger.info(f"Group vocabulary: {len(group_names_r1)} (r1), {len(group_names_r2)} (r2)")

    with open(OUT_GROUP_NAMES_R1, 'w') as fh:
        fh.write('\n'.join(group_names_r1))
    with open(OUT_GROUP_NAMES_R2, 'w') as fh:
        fh.write('\n'.join(group_names_r2))

    # Save decomposition vectors
    with open(OUT_DECOMPOSE_R1, 'w') as fh:
        json.dump(decompose_r1, fh)
    with open(OUT_DECOMPOSE_R2, 'w') as fh:
        json.dump(decompose_r2, fh)
    logger.info(f"Saved decomposition vectors to {OUT_DECOMPOSE_R1} and {OUT_DECOMPOSE_R2}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Loading training data and mapping to ModelSEED IDs")
    logger.info("=" * 60)
    training_reactions, y_train, n_unmapped = load_training_reactions_from_mat(kegg_to_ms_cpd)

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Building feature matrix and retraining model")
    logger.info("=" * 60)
    X, y, n_skipped = build_feature_matrix(
        training_reactions, y_train, decompose_r1, decompose_r2,
        group_names_r1, group_names_r2, SKIP_IN_FINGERPRINT)

    # Save training data
    savemat(OUT_TRAINING_MAT, {
        'X_comb_all': X,
        'y': y.reshape(-1, 1),
    })
    logger.info(f"Saved training matrix ({X.shape}) to {OUT_TRAINING_MAT}")

    model = retrain_model(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, OUT_MODEL, compress=3)
    logger.info(f"Saved retrained model to {OUT_MODEL}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Building ModelSEED reaction stoichiometry")
    logger.info("=" * 60)
    rxn_stoich = parse_modelseed_stoichiometry(reactions_df)
    with open(OUT_RXN_STOICH, 'w') as fh:
        json.dump(rxn_stoich, fh)
    logger.info(f"Saved stoichiometry for {len(rxn_stoich)} reactions to {OUT_RXN_STOICH}")

    # ------------------------------------------------------------------
    if not args.skip_predict:
        logger.info("=" * 60)
        logger.info("STEP 7: Predicting dG for all ModelSEED reactions")
        logger.info("=" * 60)
        results = predict_all_reactions(
            model, rxn_stoich, decompose_r1, decompose_r2,
            group_names_r1, group_names_r2, compound_dict,
            SKIP_IN_FINGERPRINT, pH=args.pH, I=args.ionic_strength)

        with open(OUT_RXN_DG, 'w') as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Saved dG predictions for {len(results)} reactions to {OUT_RXN_DG}")

    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  ModelSEED compounds loaded:    {len(compounds_df)}")
    logger.info(f"  Compounds with SMILES:         {len(cpd_smiles)}")
    logger.info(f"  Compounds decomposed (r1):     {len(decompose_r1)}")
    logger.info(f"  Compounds decomposed (r2):     {len(decompose_r2)}")
    logger.info(f"  Group vocabulary (r1):          {len(group_names_r1)}")
    logger.info(f"  Group vocabulary (r2):          {len(group_names_r2)}")
    logger.info(f"  Training reactions loaded:       {len(training_reactions)}")
    logger.info(f"  Training reactions featurized:  {X.shape[0]}")
    logger.info(f"  Feature dimensions:             {X.shape[1]}")
    logger.info(f"  ModelSEED reactions parsed:     {len(rxn_stoich)}")
    if not args.skip_predict:
        logger.info(f"  Reactions with dG predictions:  {len(results)}")


if __name__ == '__main__':
    main()
