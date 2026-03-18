import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import re
import sys
import joblib

sys.path.append('./CC/')

from compound import Compound
from compound_cacher import CompoundCacher


def load_molsig_rad1():
    return json.load(open('./data/decompose_vector_ac.json'))

def load_molsig_rad2():
    return json.load(open('./data/decompose_vector_ac_r2_py3_indent_modified_manual.json'))

def load_model():
    return joblib.load(open('./model/M12_model_BR.pkl', 'rb'))


def parse_reaction_formula_side(s):
    if s.strip() == "null":
        return {}
    compound_bag = {}
    for member in re.split(r'\s+\+\s+', s):
        tokens = member.split(None, 1)
        if len(tokens) == 0:
            continue
        if len(tokens) == 1:
            amount = 1
            key = member
        else:
            amount = float(tokens[0])
            key = tokens[1]
        compound_bag[key] = compound_bag.get(key, 0) + amount
    return compound_bag


def parse_formula(formula, arrow='<=>'):
    tokens = formula.split(arrow)
    if len(tokens) < 2:
        return None
    left = tokens[0].strip()
    right = tokens[1].strip()
    sparse_reaction = {}
    for cid, count in parse_reaction_formula_side(left).items():
        sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count
    for cid, count in parse_reaction_formula_side(right).items():
        sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count
    return sparse_reaction


def precompute_signature_matrices(molsig_r1, molsig_r2):
    """Pre-build numpy arrays and lookup dicts from molecular signatures."""
    moie_r1 = open('./data/group_names_r1.txt').read().splitlines()
    moie_r2 = open('./data/group_names_r2_py3_modified_manual.txt').read().splitlines()

    molsigna_df1 = pd.DataFrame.from_dict(molsig_r1).fillna(0).reindex(moie_r1).fillna(0)
    molsigna_df2 = pd.DataFrame.from_dict(molsig_r2).fillna(0).reindex(moie_r2).fillna(0)

    compounds1 = molsigna_df1.columns.tolist()
    compounds2 = molsigna_df2.columns.tolist()
    cid_to_idx1 = {cid: i for i, cid in enumerate(compounds1)}
    cid_to_idx2 = {cid: i for i, cid in enumerate(compounds2)}

    mat1 = molsigna_df1.values  # (n_groups_r1, n_compounds)
    mat2 = molsigna_df2.values  # (n_groups_r2, n_compounds)

    return mat1, mat2, cid_to_idx1, cid_to_idx2


def build_rule_vector(rxn_dict, mat1, mat2, cid_to_idx1, cid_to_idx2, n_features):
    """Build a single rule vector from a parsed reaction dict."""
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]

    rule_vec1 = np.zeros(n1)
    rule_vec2 = np.zeros(n2)

    for met, stoic in rxn_dict.items():
        if met == "C00080" or met == "C00282":
            continue
        if met not in cid_to_idx1 or met not in cid_to_idx2:
            return None
        rule_vec1 += mat1[:, cid_to_idx1[met]] * stoic
        rule_vec2 += mat2[:, cid_to_idx2[met]] * stoic

    # Pad each with 44 zeros, then concatenate
    X1 = np.concatenate([rule_vec1, np.zeros(44)])
    X2 = np.concatenate([rule_vec2, np.zeros(44)])
    result = np.concatenate([X1, X2])

    # Pad to match model feature count if needed
    if len(result) < n_features:
        result = np.concatenate([result, np.zeros(n_features - len(result))])

    return result


def compute_ddG0(rxn_dict, pH, I, T, ccache):
    """Compute the thermodynamic correction for a reaction."""
    ddG0_forward = 0.0
    for compound_id, coeff in rxn_dict.items():
        comp = ccache.get_compound(compound_id)
        ddG0_forward += coeff * comp.transform_pH7(pH, I, T)
    return ddG0_forward


# ===== Main =====

print('Loading data...')
molsig_r1 = load_molsig_rad1()
molsig_r2 = load_molsig_rad2()
loaded_model = load_model()
ccache = CompoundCacher()

print('Pre-computing signature matrices...')
mat1, mat2, cid_to_idx1, cid_to_idx2 = precompute_signature_matrices(molsig_r1, molsig_r2)
del molsig_r1, molsig_r2

n_features = loaded_model.n_features_in_
model_coef = loaded_model.coef_
model_sigma = loaded_model.sigma_
model_alpha = loaded_model.alpha_

# Load ModelSEED reactions
Rxn_f0 = './../ModelSEEDDatabase/Biochemistry/reaction_00.json'
json_read = json.load(open(Rxn_f0))

KEGG_id_ls = []
mseed_rxn_id_ls = []

print('Extracting KEGG IDs...')
for rxn in json_read:
    try:
        kegg_id_str = None
        for ki in rxn.get('aliases', []):
            if 'KEGG' in ki:
                kegg_id_str = ki
        if kegg_id_str is not None:
            KEGG_id = kegg_id_str.replace(' ', '').split(':')[1]
        else:
            KEGG_id = 'No KEGG id'
        KEGG_id_ls.append(KEGG_id)
        mseed_rxn_id_ls.append(rxn['id'])
    except:
        KEGG_id_ls.append('No KEGG id')
        mseed_rxn_id_ls.append(rxn['id'])

kegg_rxn_eqn = json.load(open('./data/KEGG_rxn_eqn_master_branch.json'))
kegg_rxn_eqn_keys = set(kegg_rxn_eqn.keys())

# Phase 1: Parse all reactions and build rule vectors
print('Phase 1: Building rule vectors...')
# For each (mseed_idx, kegg_rxn_id), store the rule vector and parsed rxn_dict
jobs = []  # list of (mseed_idx, krxn, rule_vector, rxn_dict) or (mseed_idx, krxn, None, None)

for ix, mseed in enumerate(mseed_rxn_id_ls):
    kid = KEGG_id_ls[ix]
    for krxn in kid.split(';'):
        if krxn not in kegg_rxn_eqn_keys:
            jobs.append((ix, krxn, None, None))
            continue
        try:
            rxn_dict = parse_formula(kegg_rxn_eqn[krxn])
            if rxn_dict is None:
                jobs.append((ix, krxn, None, None))
                continue
            rule_vec = build_rule_vector(rxn_dict, mat1, mat2, cid_to_idx1, cid_to_idx2, n_features)
            if rule_vec is None:
                jobs.append((ix, krxn, None, None))
                continue
            jobs.append((ix, krxn, rule_vec, rxn_dict))
        except:
            jobs.append((ix, krxn, None, None))

# Separate valid jobs for batch prediction
valid_indices = []
X_list = []
for i, (mseed_idx, krxn, rule_vec, rxn_dict) in enumerate(jobs):
    if rule_vec is not None:
        valid_indices.append(i)
        X_list.append(rule_vec)

print(f'Total jobs: {len(jobs)}, valid for prediction: {len(valid_indices)}')

# Phase 2: Batch predict (manual computation to avoid slow return_std)
print('Phase 2: Batch predicting...')
if X_list:
    X_batch = np.array(X_list)  # (n_valid, n_features)

    # ymean = X @ coef
    ymeans = X_batch @ model_coef

    # ystd = sqrt(X @ sigma @ X.T diagonal + 1/alpha)
    # Compute (X @ sigma) * X row-wise, then sum => diagonal of X @ sigma @ X.T
    CHUNK = 500  # process in chunks to manage memory
    ystds = np.empty(len(X_list))
    for start in tqdm(range(0, len(X_list), CHUNK), desc='Computing std'):
        end = min(start + CHUNK, len(X_list))
        Xc = X_batch[start:end]
        sigmas_sq = (Xc @ model_sigma * Xc).sum(axis=1)
        ystds[start:end] = np.sqrt(sigmas_sq + 1.0 / model_alpha)

    CIs = (ystds * 1.96) / np.sqrt(4001)

# Phase 3: Compute ddG0 corrections and assemble results
print('Phase 3: Computing thermodynamic corrections...')
pH = 7.0
I = 0.25
T = 298.15

# Map valid predictions back
pred_map = {}  # valid_index_in_jobs -> (ymean, CI)
for vi_pos, job_idx in enumerate(valid_indices):
    pred_map[job_idx] = (ymeans[vi_pos], CIs[vi_pos])

dG_dict = {}
failed_ddG = 0
for i, (mseed_idx, krxn, rule_vec, rxn_dict) in tqdm(enumerate(jobs), total=len(jobs), desc='Assembling'):
    mseed = mseed_rxn_id_ls[mseed_idx]
    if mseed not in dG_dict:
        dG_dict[mseed] = {}

    if i not in pred_map:
        dG_dict[mseed][krxn] = {'dG': np.nan, 'dG_ConfidenceInterval': np.nan}
        continue

    ymean_val, CI_val = pred_map[i]
    try:
        ddG0 = compute_ddG0(rxn_dict, pH, I, T, ccache)
        dG_dict[mseed][krxn] = {'dG': float(ymean_val + ddG0), 'dG_ConfidenceInterval': float(CI_val)}
    except:
        failed_ddG += 1
        dG_dict[mseed][krxn] = {'dG': np.nan, 'dG_ConfidenceInterval': np.nan}

print(f'Failed ddG0 corrections: {failed_ddG}')

print('Dumping results...')
fdump_name = './Modelseed_dG/dG_rxn_file_1.json'
with open(fdump_name, 'w') as f:
    json.dump(dG_dict, f, indent=4)

print(f'Results saved to {fdump_name}')
