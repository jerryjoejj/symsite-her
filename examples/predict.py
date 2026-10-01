#!/usr/bin/env python3
"""Load the retained production checkpoint on CPU; never train or overwrite it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    import torch
    from dataclasses import replace
    sys.path.insert(0, str(ROOT / 'src'))
    from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareEnsembleV2, SymmetryAwareMLP_V2
    path = ROOT / 'models/ensemble/symmetry_ensemble_v2_production_v3.pt'
    with (ROOT / 'provenance/file_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        expected = next(r['public_sha256'] for r in csv.DictReader(f)
                        if r['public_path'] == str(path.relative_to(ROOT)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Checkpoint checksum mismatch'
    # This exact project checkpoint contains its dataclass configuration.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = replace(checkpoint['config'], device='cpu')
    model = SymmetryAwareEnsembleV2(config)
    model.stats_ = checkpoint['stats']
    model.y_mean_ = checkpoint['y_mean']
    model.y_std_ = checkpoint['y_std']
    for state in checkpoint['member_states']:
        member = SymmetryAwareMLP_V2(config).to('cpu')
        member.load_state_dict(state, strict=True)
        member.eval()
        model.members.append(member)
    assert len(model.members) == 10
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/compositions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/example_predictions.csv')
    parser.add_argument('--check-model', action='store_true', help='Only deserialize weights on CPU and check tensor shapes; no predictions.')
    args = parser.parse_args()
    model = load_model()
    if args.check_model:
        print(json.dumps({'status':'pass', 'members':len(model.members), 'device':'cpu',
                          'parameters_per_member':sum(p.numel() for p in model.members[0].parameters()),
                          'prediction_performed':False}, indent=2))
        return
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Choose an output inside this bundle\'s outputs/ directory.')
    if output.exists():
        raise FileExistsError(output)
    import numpy as np
    import pandas as pd
    from herald.features.calculator import FeatureCalculator, load_element_properties
    props = load_element_properties(ROOT / 'data/elements/element_properties.json')
    rows = pd.read_csv(args.input)
    for name in ['M1','M2','M3','M4']:
        if name not in rows:
            raise ValueError('Input must contain M1, M2, M3, M4 columns')
    for values in rows[['M1','M2','M3','M4']].itertuples(index=False, name=None):
        if len(set(values)) != 4 or not set(values).issubset(props):
            raise ValueError('Each row must use four distinct supported metals')
    if rows.empty:
        raise ValueError('Input has no compositions')
    calculator = FeatureCalculator(props)
    features = calculator.compute(rows)
    inputs = calculator.compute_symmetric_input_from_features(features)
    prediction = model.predict(inputs)
    result = rows[['M1','M2','M3','M4']].copy()
    result['predicted_delta_G_H_eV'] = prediction.mean
    result['epistemic_sd_eV'] = np.sqrt(np.maximum(prediction.epistemic, 0))
    result['aleatoric_sd_eV'] = np.sqrt(np.maximum(prediction.aleatoric, 0))
    result['total_sd_eV'] = prediction.total_std
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as handle:
        result.to_csv(handle, index=False)
    print(f'Wrote {len(result)} predictions to {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
