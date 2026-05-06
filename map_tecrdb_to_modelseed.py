"""Map all TECRDB metabolites to ModelSEED compound IDs.

TECRDB compounds come as KEGG IDs (C#####) in CC/data_cc/TECRDB.tsv. Most map
directly via ModelSEED's alias file. A handful are "fake" KEGG IDs (C80###)
added manually in CC/data_cc/kegg_additions.tsv — for these we fall back to
matching by InChIKey, then by name.

Output:
    data/tecrdb_to_modelseed_map.json   — per-compound mapping with provenance
"""

import os
import re
import glob
import json
import logging
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODELSEED_DIR = '../ModelSEEDDatabase/Biochemistry'
TECRDB_FILE = 'CC/data_cc/TECRDB.tsv'
KEGG_ADDITIONS = 'CC/data_cc/kegg_additions.tsv'
OUT_FILE = 'data/tecrdb_to_modelseed_map.json'


def load_tecrdb_compounds():
    """Extract all unique KEGG compound IDs used in TECRDB reaction equations."""
    tecrdb = pd.read_csv(TECRDB_FILE, sep='\t', header=None)
    all_cpds = set()
    for eq in tecrdb[6].dropna():
        all_cpds.update(re.findall(r'C\d{5}', str(eq)))
    logger.info(f"Found {len(all_cpds)} unique TECRDB compound IDs")
    return sorted(all_cpds)


def load_kegg_additions():
    """Load the 'fake' KEGG IDs (C80###) from kegg_additions.tsv.

    Returns: {kegg_id: {'name': ..., 'inchi': ...}}
    """
    additions = {}
    with open(KEGG_ADDITIONS) as fh:
        header = fh.readline().strip().split('\t')  # name, cid, inchi
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            name = parts[0]
            cid_num = parts[1]
            inchi = parts[2].strip('"')
            try:
                cid = f'C{int(cid_num):05d}'
            except ValueError:
                continue
            additions[cid] = {'name': name, 'inchi': inchi}
    logger.info(f"Loaded {len(additions)} KEGG additions")
    return additions


def load_kegg_to_modelseed_aliases():
    """Build KEGG C##### -> ModelSEED cpd##### map from the alias file."""
    alias_file = os.path.join(MODELSEED_DIR, 'Aliases', 'Unique_ModelSEED_Compound_Aliases.txt')
    kegg_to_ms = {}
    with open(alias_file) as fh:
        fh.readline()
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            ms_id, ext_id, source = parts[0], parts[1], parts[2]
            if source == 'KEGG' and ext_id.startswith('C'):
                kegg_to_ms.setdefault(ext_id, []).append(ms_id)
    logger.info(f"Built KEGG->ModelSEED alias: {len(kegg_to_ms)} KEGG IDs")
    return kegg_to_ms


def load_modelseed_compounds():
    """Load all ModelSEED compounds with name, inchikey, formula, smiles."""
    frames = []
    for f in sorted(glob.glob(os.path.join(MODELSEED_DIR, 'compound_*.tsv'))):
        frames.append(pd.read_csv(f, sep='\t', dtype=str, na_values=['null']))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df)} ModelSEED compounds")
    return df


def inchi_to_inchikey(inchi):
    """Convert an InChI string to an InChIKey via RDKit."""
    if not inchi:
        return None
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def build_name_lookup(ms_df):
    """Build a lowercase-name -> cpd_id lookup from ModelSEED name and aliases."""
    name_to_cpd = {}
    for _, row in ms_df.iterrows():
        cpd_id = row['id']
        for field in ['name', 'abbreviation']:
            v = row.get(field)
            if pd.isna(v) or not v:
                continue
            name_to_cpd.setdefault(str(v).lower().strip(), cpd_id)

        # Also parse the aliases column for "Name: a; b; c|...." tokens
        aliases_str = row.get('aliases')
        if pd.isna(aliases_str) or not aliases_str:
            continue
        for block in str(aliases_str).split('|'):
            if block.startswith('Name:'):
                names = block[5:].split(';')
                for n in names:
                    n = n.strip().lower()
                    if n:
                        name_to_cpd.setdefault(n, cpd_id)
    logger.info(f"Built name lookup with {len(name_to_cpd)} entries")
    return name_to_cpd


def main():
    tecrdb_cpds = load_tecrdb_compounds()
    kegg_additions = load_kegg_additions()
    kegg_to_ms = load_kegg_to_modelseed_aliases()
    ms_df = load_modelseed_compounds()

    # Build InChIKey lookup from ModelSEED
    inchikey_to_cpd = {}
    for _, row in ms_df.iterrows():
        ik = row.get('inchikey')
        if not pd.isna(ik) and ik and ik != 'null':
            inchikey_to_cpd.setdefault(str(ik).strip(), row['id'])
    logger.info(f"Built InChIKey lookup: {len(inchikey_to_cpd)} entries")

    name_to_cpd = build_name_lookup(ms_df)

    # Map each TECRDB compound
    mapping = {}
    stats = {'via_alias': 0, 'via_inchikey': 0, 'via_name': 0, 'unmapped': 0}
    unmapped = []

    for kegg_id in tecrdb_cpds:
        # 1. Try direct KEGG→ModelSEED alias
        if kegg_id in kegg_to_ms:
            ms_ids = kegg_to_ms[kegg_id]
            mapping[kegg_id] = {
                'modelseed_ids': ms_ids,
                'primary': ms_ids[0],
                'method': 'kegg_alias',
            }
            stats['via_alias'] += 1
            continue

        # 2. Fake KEGG ID (C80###) — try InChIKey then name
        if kegg_id in kegg_additions:
            name = kegg_additions[kegg_id]['name']
            inchi = kegg_additions[kegg_id]['inchi']
            inchikey = inchi_to_inchikey(inchi)

            # Try InChIKey match
            if inchikey and inchikey in inchikey_to_cpd:
                mapping[kegg_id] = {
                    'modelseed_ids': [inchikey_to_cpd[inchikey]],
                    'primary': inchikey_to_cpd[inchikey],
                    'method': 'inchikey',
                    'name': name,
                    'inchikey': inchikey,
                }
                stats['via_inchikey'] += 1
                continue

            # Try first block of InChIKey (connectivity only, ignore stereo/charge)
            if inchikey:
                ik_first = inchikey.split('-')[0]
                candidates = [cpd for ik, cpd in inchikey_to_cpd.items()
                              if ik.split('-')[0] == ik_first]
                if candidates:
                    mapping[kegg_id] = {
                        'modelseed_ids': list(set(candidates)),
                        'primary': candidates[0],
                        'method': 'inchikey_first_block',
                        'name': name,
                        'inchikey': inchikey,
                    }
                    stats['via_inchikey'] += 1
                    continue

            # Try name match
            if name.lower().strip() in name_to_cpd:
                mapping[kegg_id] = {
                    'modelseed_ids': [name_to_cpd[name.lower().strip()]],
                    'primary': name_to_cpd[name.lower().strip()],
                    'method': 'name',
                    'name': name,
                }
                stats['via_name'] += 1
                continue

            mapping[kegg_id] = {
                'modelseed_ids': [], 'primary': None,
                'method': 'unmapped', 'name': name, 'inchi': inchi,
                'inchikey': inchikey,
            }
            stats['unmapped'] += 1
            unmapped.append((kegg_id, name))
            continue

        # 3. Neither in alias nor in additions
        mapping[kegg_id] = {
            'modelseed_ids': [], 'primary': None, 'method': 'unmapped',
        }
        stats['unmapped'] += 1
        unmapped.append((kegg_id, None))

    # Build summary
    summary = {
        'n_tecrdb_compounds': len(tecrdb_cpds),
        'mapped_total': sum(v for k, v in stats.items() if k != 'unmapped'),
        'unmapped_total': stats['unmapped'],
        'by_method': stats,
        'unmapped_list': unmapped,
    }

    output = {'_summary': summary, 'mappings': mapping}
    with open(OUT_FILE, 'w') as fh:
        json.dump(output, fh, indent=2)

    logger.info(f"Saved mapping to {OUT_FILE}")
    print()
    print("=" * 60)
    print("TECRDB -> ModelSEED mapping summary")
    print("=" * 60)
    print(f"  Total TECRDB compounds: {len(tecrdb_cpds)}")
    print(f"  Mapped via KEGG alias:       {stats['via_alias']:4d}")
    print(f"  Mapped via InChIKey:         {stats['via_inchikey']:4d}")
    print(f"  Mapped via name:             {stats['via_name']:4d}")
    print(f"  Unmapped:                    {stats['unmapped']:4d}")
    if unmapped:
        print(f"\n  Unmapped compounds:")
        for cid, name in unmapped:
            print(f"    {cid}: {name}")


if __name__ == '__main__':
    main()
