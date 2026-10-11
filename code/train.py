#!/usr/bin/env python3
"""Train the production ensemble or repeat the published ML comparisons."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from reproduce import CODE,ROOT,LIB,Reproduction,BLOCKS,identities
import numpy as np
import pandas as pd
import torch
from herald.features.calculator import FeatureCalculator,load_element_properties
from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareEnsembleV2,SymmetryEnsembleConfigV2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['production','architecture','acquisition'],default='production')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; supplied models are never replaced.')
    parser.add_argument('--smoke',action='store_true',help='Test production training with one member and two epochs.')
    args=parser.parse_args()
    if args.smoke and args.task!='production':parser.error('--smoke applies to production training only.')
    output=args.output.resolve()
    for protected in [ROOT/'data',ROOT/'models',CODE,LIB]:
        if output==protected or output.is_relative_to(protected):parser.error('Output must be outside input and source directories.')
    models=ROOT/'models' if (ROOT/'models').is_dir() else ROOT/'data/models'
    Reproduction(ROOT/'data',models,output).verify_inputs()
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(4)
    feature_path=ROOT/'data/processed/training/consolidated_ground_state_features_v2.csv'
    df=pd.read_csv(feature_path)
    props=load_element_properties(ROOT/'data/elements/element_properties.json')
    data=FeatureCalculator(props).compute_symmetric_input_from_features(df)
    y=df.delta_G_H.to_numpy()
    manifest=dict(task=args.task,smoke=args.smoke,device='cpu',threads=4,
                  training_compositions=len(df),base_seed=42)
    if args.task=='production':
        cfg=SymmetryEnsembleConfigV2(device='cpu',n_members=1 if args.smoke else 10,epochs=2 if args.smoke else 300)
        ensemble=SymmetryAwareEnsembleV2(cfg).fit(data,y,verbose=True)
        ensemble.save(output/'trained_ensemble.pt')
        prediction=ensemble.predict(data)
        assert np.isfinite(prediction.mean).all()
        pd.DataFrame(dict(physical_id=identities(df),dft=y,prediction=prediction.mean,
                          epistemic_std=np.sqrt(prediction.epistemic))).to_csv(output/'training_predictions.csv',index=False)
        manifest.update(config=asdict(cfg),training_MAE_eV=float(np.mean(abs(prediction.mean-y))),
                        member_seeds=list(range(cfg.seed,cfg.seed+cfg.n_members)))
    elif args.task=='architecture':
        import architecture_training as protocol
        protocol.TRAIN_FEATURES=feature_path
        # Sequential cells keep the capsule's CPU and memory requirements fixed.
        results=[protocol.run_cell(cell) for cell in protocol.CELLS]
        pd.DataFrame([dict(cell=r['cell'],arch=r['arch'],loss=r['loss'],fold=i,mae=mae)
                      for r in results for i,mae in enumerate(r['maes'])]).to_csv(output/'ablation_folds.csv',index=False)
        manifest.update(folds=5,members_per_fold=10,maximum_epochs=200)
    else:
        import acquisition_training as protocol
        protocol.TRAIN_FEATURES=feature_path
        protocol.PROPS_PATH=ROOT/'data/elements/element_properties.json'
        protocol.OUT=output
        records=[]
        for rep in range(protocol.N_REPLICATES):
            records.extend(protocol._run_replicate((rep,len(df))))
            # Persist finished repeats during the longer training job.
            pd.DataFrame(records).to_csv(output/'acquisition_curves.csv',index=False)
        protocol._paired_stats(pd.DataFrame(records)).to_csv(output/'paired_statistics.csv',index=False)
        manifest.update(repeats=8,rounds=5,initial_size=200,batch_size=100,members=5,epochs=200,
                        split_seeds=list(range(42,50)))
    manifest.update(elapsed_seconds=time.monotonic()-started,status='completed')
    (output/'training_run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
