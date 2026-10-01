#!/usr/bin/env python3
"""Verify file integrity, Python syntax and the supplied scientific datasets."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import warnings
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def table(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def physical_id(row):
    sites = [row[f'M{i}'] for i in range(1, 5)]
    assert len(set(sites)) == 4, sites
    return '-'.join(sites[:2] + sorted(sites[2:]))


def main():
    verified = set()
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        p = ROOT / relative
        assert p.resolve().is_relative_to(ROOT), relative
        assert p.is_file() and not p.is_symlink(), relative
        assert digest(p) == expected, relative
        verified.add(relative)
    for p in ROOT.rglob('*'):
        assert not p.is_symlink(), p
        if p.is_file():
            assert not p.name.upper().startswith('POTCAR') or p.suffix in {'.spec', '.json'}, p
            assert p.suffix.lower() not in {'.ttf', '.otf', '.pdf', '.png', '.bmp', '.svg', '.jpg', '.jpeg', '.tif', '.tiff'}, p
            if p.suffix.lower() not in {'.py','.md'}:
                assert not re.search(r'INCAR|KPOINTS|POTCAR|POSCAR|CONTCAR|OUTCAR|OSZICAR|DOSCAR|PROCAR|EIGENVAL|CHGCAR|WAVECAR|AECCAR|LOCPOT|ELFCAR|XDATCAR|IBZKPT|vasprun|vaspout|ACF\.dat|COHPCAR|ICOHPLIST', p.name, re.I), p
                assert p.suffix.lower() not in {'.vasp','.cif','.xyz','.xml','.zip','.gz','.tgz','.lobster'}, p
    for excluded in ['data/raw','data/protocols','data/dft_templates','figures','scripts/figures']:
        assert not (ROOT/excluded).exists(), excluded
    copied = table('provenance/file_manifest.csv')
    for row in copied:
        assert row['public_path'] in verified, row['public_path']
        p = ROOT / row['public_path']
        assert p.stat().st_size == int(row['bytes']), p
        assert digest(p) == row['public_sha256'], p
        if row['change'] == 'byte_identical':
            assert row['previous_sha256'] == row['public_sha256'], p
    for old in ['history_research','next_plan','SISSO','paper','results']:
        assert not (ROOT/old).exists(), old
    syntax_warnings = []
    python_files = list(ROOT.rglob('*.py'))
    for p in python_files:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p.relative_to(ROOT)))
        for warning in recorded:
            syntax_warnings.append({'path':str(p.relative_to(ROOT)), 'message':str(warning.message)})
    training = table('data/processed/training/consolidated_ground_state.csv')
    heldout = table('data/processed/architecture/blindtest_v2_predictions_v3.csv')
    validation = table('data/processed/dft_validation/validation_membership.csv')
    screening = table('data/processed/screening/activity_map_28560_v3.csv')
    cheap = table('data/processed/screening/SI_table_cheap_candidates.csv')
    paired = table('data/analysis/electronic_structure/paired_responses/paired_electronic_changes.csv')
    assert [len(x) for x in [training, heldout, validation, screening, cheap, paired]] == [995,110,60,28560,137,129]
    train_ids = {physical_id(row) for row in training}
    heldout_ids = {physical_id(row) for row in heldout}
    validation_ids = {physical_id(row) for row in validation}
    assert len(train_ids) == 995 and len(heldout_ids) == 110 and len(validation_ids) == 60
    assert not train_ids & heldout_ids
    assert len(train_ids & validation_ids) == 2
    assert len(heldout_ids & validation_ids) == 1
    assert len({row['physical_id'] for row in paired}) == 107
    feat = table('data/processed/features/training_87d.csv')
    assert len(feat) == 995 and len(feat[0]) == 93  # six metadata fields plus 87 features
    assert {row['physical_id'] for row in feat} == train_ids
    metadata = json.loads((ROOT / 'models/ensemble/symmetry_ensemble_v2_production_v3_meta.json').read_text())
    assert metadata['n_train'] == 995 and metadata['n_members'] == 10
    assert metadata['input_dims'] == {'M1':16, 'M2':13, 'M34_sym':6, 'global':52, 'total':87}
    counts = {'training':995, 'held_out':110, 'validation':60, 'full_space':28560,
              'all_3d_candidates':137, 'r7_record_pairs':129, 'r7_compositions':107}
    print(json.dumps({'status':'pass', 'hashed_files':len(verified), 'retained_files':len(copied),
                      'retained_bytes':sum(int(r['bytes']) for r in copied),
                      'adapted_files':sum(r['change']!='byte_identical' for r in copied),
                      'python_files_parsed':len(python_files), 'syntax_warnings':syntax_warnings,
                      'data_counts':counts,
                      'raw_calculation_files_present':False,
                      'git_directory_present':(ROOT / '.git').exists(),
                      'scope':'File integrity, Python syntax, dataset identities and stored model metadata.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
