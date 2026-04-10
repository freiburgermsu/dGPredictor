# dGPredictor: Changes Report

## 1. Environment Setup

### uv Virtual Environment (`.venv/`)
- Installed all dependencies into the existing uv-managed `.venv` (Python 3.13.2) using `uv pip install`:
  - `pandas`, `numpy`, `tqdm`, `pillow`, `scikit-learn`, `joblib`, `rdkit`, `openbabel-wheel`
- No `requirements.txt` or `pyproject.toml` existed in the repo; packages were identified by reading import statements across the codebase.

## 2. Model Regeneration

### `model/M12_model_BR.pkl`
- The original pickle file was created with Python 3.8 and was incompatible with Python 3.13 (`KeyError: 118` during unpickling).
- Regenerated the BayesianRidge model by running the training pipeline in `model_gen.py` against `data/Test_KEGG_all_grp.mat`.
- The retrained model achieved identical metrics: **MSE = 9.60, R² = 0.9998**.
- Training took ~22 minutes due to the 26,404-feature BayesianRidge fit with `tol=1e-6` and `compute_score=True`.

## 3. Data Compatibility Fix

### ModelSEED Reaction File Symlink
- The script expected `./../ModelSEEDDatabase/Biochemistry/reaction_00.json` (a split file from an older database layout), but the current ModelSEEDDatabase only contains `reactions.json` (a single consolidated file with the same JSON array structure).
- Created a symlink: `reaction_00.json` → `reactions.json`.

### Output Directory
- Created `Modelseed_dG/` directory, which the script expected but did not exist.

## 4. Bug Fixes in `dG_prediction_modelseed_dev_branch_file_run.py`

### 4a. Missing Reaction Parsing
- **Bug**: The script passed raw equation strings (e.g., `"2 C00001 + C00002 <=> C00003"`) directly to `get_dG0_only()`, which expected a parsed dictionary `{compound_id: stoichiometric_coefficient}`.
- **Fix**: Added a `parse_formula()` call to convert the string to a sparse reaction dict before prediction.

### 4b. NumPy 2.0 Incompatibility
- **Bug**: `np.NaN` was removed in NumPy 2.0 (installed version: 2.4.3), causing `AttributeError` in the exception handler.
- **Fix**: Replaced `np.NaN` with `np.nan`.

### 4c. Regex Escape Sequence Warning
- **Fix**: Changed bare string `'\s+\+\s+'` to raw string `r'\s+\+\s+'` to eliminate `SyntaxWarning: invalid escape sequence`.

## 5. Performance Optimization of `dG_prediction_modelseed_dev_branch_file_run.py`

The original script took ~30 seconds per reaction (estimated **~15 days** for 43,965 reactions). The optimized version completes in **~2 minutes** — a **~10,000x speedup**.

### 5a. Pre-computed Signature Matrices
- **Problem**: `get_rule()` rebuilt two full pandas DataFrames from ~4,000-compound dictionaries (`pd.DataFrame.from_dict(molsig)`) and re-read `data/group_names_r1.txt` and `data/group_names_r2_py3_modified_manual.txt` from disk on every single call.
- **Fix**: Added `precompute_signature_matrices()` that builds the DataFrames and converts them to numpy arrays with compound-ID-to-column-index lookup dicts once at startup. The hot-path function `build_rule_vector()` now uses direct numpy array indexing instead of pandas operations.

### 5b. Batched Model Prediction
- **Problem**: `sklearn.BayesianRidge.predict(X, return_std=True)` took **2.2 seconds per call** due to computing the full posterior variance over the 26,404×26,404 covariance matrix (`sigma_`).
- **Profiling Results**:
  - `predict(return_std=True)`: 2,188 ms/call
  - `predict(return_std=False)`: 0.3 ms/call
  - Manual batched (1,000 samples): 7.5 ms/sample
- **Fix**: Replaced per-reaction sklearn `predict()` calls with a three-phase batch pipeline:
  1. **Phase 1**: Pre-parse all 44,504 reaction jobs and build rule vectors (seconds).
  2. **Phase 2**: Batch-predict all 11,397 valid reactions by manually computing `ymean = X @ coef_` and `ystd = sqrt((X @ sigma_ * X).sum(axis=1) + 1/alpha_)` in chunks of 500 (~1 min 45s total).
  3. **Phase 3**: Apply thermodynamic corrections (`transform_pH7`) and assemble results (~2 seconds).

### 5c. Removed Unused Imports
- Removed imports not needed for the batch script: `PIL`, `pickle`, `pdb`, `rdkit.Chem.Draw`, `rdkit.Chem.rdChemReactions`, `chemaxon`.

## 6. Repository Cleanup

### `.gitignore` (new file)
- Added rules to exclude:
  - `model/M12_model_BR.pkl` (143 MB, too large for GitHub)
  - `.DS_Store` (macOS artifact)
  - `__pycache__/` (Python bytecode cache)

### Removed from Git Tracking
- `model/M12_model_BR.pkl` — removed via `git rm --cached` (file kept locally for runtime use).
- `CC/__pycache__/*.pyc` — removed stale Python 3.8 and 3.9 bytecode files that were previously committed.

## 7. Generated Output

### `Modelseed_dG/dG_rxn_file_1.json`
- Contains dG predictions and confidence intervals for 43,965 ModelSEED reactions mapped to KEGG IDs.
- 11,397 reactions successfully predicted; remainder marked as `NaN` (no KEGG equation available or compound not in signature database).
- Predictions computed at pH 7.0, ionic strength I = 0.25 M, T = 298.15 K.
