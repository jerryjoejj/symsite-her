#!/usr/bin/env python3
"""Recompute the NC manuscript's numerical results from curated tables and models."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
LIB = ROOT / 'src' if (ROOT / 'src').is_dir() else CODE / 'src'
sys.path.insert(0, str(LIB))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.acquisition.physics_guided import PhysicsPrior

SITES = ['M1', 'M2', 'M3', 'M4']
BLOCKS = ['x_M1', 'x_M2', 'x_M34_sym', 'g_global']


def identities(df):
    return ['-'.join(list(x[:2]) + sorted(x[2:])) for x in df[SITES].itertuples(index=False, name=None)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reproduction:
    def __init__(self, data, models, output):
        self.data, self.models, self.output = data, models, output
        self.checks, self.inputs, self.metrics = [], {}, {}
        self.section = ''

    def input(self, relative):
        path = self.models / relative[7:] if relative.startswith('models/') else self.data / relative[5:]
        if not path.is_file():
            raise FileNotFoundError(f'Required reproduction input is missing: {relative}')
        self.inputs[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return path

    def table(self, relative):
        return pd.read_csv(self.input('data/' + relative))

    def js(self, relative):
        return json.loads(self.input(relative).read_text())

    def save(self, name, df):
        df.to_csv(self.output / (name + '.csv'), index=False, float_format='%.12g')

    def check(self, name, actual, expected, atol=0., reference=''):
        actual, expected = float(actual), float(expected)
        ok = math.isfinite(actual) and abs(actual - expected) <= atol
        self.checks.append(dict(section=self.section, result=name, actual=actual,
                                expected=expected, absolute_tolerance=atol, passed=ok,
                                manuscript_reference=reference))
        if not ok:
            raise AssertionError(f'{self.section}/{name}: {actual} differs from {expected} (tolerance {atol})')

    def equal(self, name, a, b, atol=1e-10):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise AssertionError(f'{name}: shape mismatch {a.shape} != {b.shape}')
        error = np.max(np.abs(a - b)) if a.size else 0.
        self.check(name, error, 0., atol)

    def training(self):
        self.section = 'R1: datasets and symmetric representation'
        records = self.table('reproduction/training_records_1206.csv')
        records['physical_id'] = identities(records)
        counts = records.groupby('cohort').size()
        for cohort, n in [('initial', 500), ('adaptive', 98), ('reference', 608)]:
            self.check(f'{cohort}_records', counts[cohort], n, reference='Results R1; Methods: DFT Dataset Construction')
        self.check('all_training_source_records', len(records), 1206)
        train = records.groupby('physical_id', sort=True).agg(
            E_clean=('E_clean', 'min'), E_H=('E_H', 'min'), n_replicas=('record_id', 'size')).reset_index()
        train[SITES] = train.physical_id.str.split('-', expand=True)
        train['delta_G_H'] = train.E_H - train.E_clean + 3.481
        for name in ['E_clean', 'E_H']:
            winners = records.loc[records.groupby('physical_id')[name].idxmin()].set_index('physical_id').record_id
            train[name + '_record_id'] = train.physical_id.map(winners)
        train['training_row_0based'] = np.arange(len(train))
        self.check('training_compositions', len(train), 995)
        reference = self.table('processed/training/consolidated_ground_state.csv')
        reference['physical_id'] = identities(reference)
        ref = reference.set_index('physical_id').loc[train.physical_id]
        self.equal('deduplicated_energies_and_labels', train[['E_clean','E_H','delta_G_H']], ref[['E_clean','E_H','delta_G_H']])
        self.save('training_995_rebuilt', train)
        self.save('training_cohort_counts', counts.rename('n_records').reset_index())
        al = self.table('reproduction/adaptive_selection_100.csv')
        self.check('adaptive_selected', len(al), 100)
        self.check('adaptive_valid_results', al.delta_G_H.notna().sum(), 98)
        self.equal('adaptive_labels', al.dropna(subset=['delta_G_H']).delta_G_H,
                   al.dropna(subset=['delta_G_H']).E_H - al.dropna(subset=['delta_G_H']).E_clean + 3.481)
        self.save('adaptive_selection_inventory', al)
        initial = self.table('processed/cohorts/round_0/selection/selected_500_structures.csv')
        initial['physical_id'] = identities(initial)
        self.save('initial_selection_inventory', initial)
        initial_results=initial.merge(records[records.cohort=='initial'][SITES+['E_clean','E_H','delta_G_H']],on=SITES,validate='one_to_one')
        initial_results['hit_0p2']=abs(initial_results.delta_G_H)<=0.2
        groups=initial_results.groupby('strategy').hit_0p2.agg(['count','sum'])
        for strategy,n,hits in [('cluster',250,55),('physics',150,8),('random',100,21)]:
            self.check('initial_'+strategy+'_size',groups.loc[strategy,'count'],n)
            self.check('initial_'+strategy+'_hits',groups.loc[strategy,'sum'],hits)
        self.save('initial_selection_results',initial_results)
        # Rebuild the actual seed memberships used in the eight retrospective repeats.
        memberships=[]
        for rep in range(8):
            order=np.random.default_rng(42+rep).permutation(len(train))
            for position, i in enumerate(order):
                memberships.append(dict(repeat=rep,seed=42+rep,permutation_position=position,
                                        training_row_0based=int(i),physical_id=train.physical_id.iloc[i],
                                        membership='initial_200' if position<200 else 'candidate_pool_795'))
        self.save('retrospective_initial_memberships',pd.DataFrame(memberships))
        folds=[]
        for fold, (_,test) in enumerate(KFold(5,shuffle=True,random_state=42).split(train)):
            for i in test: folds.append(dict(training_row_0based=int(i),physical_id=train.physical_id.iloc[i],fold=fold))
        self.save('five_fold_membership',pd.DataFrame(folds).sort_values('training_row_0based'))
        props_path=self.input('data/elements/element_properties.json')
        self.props=load_element_properties(props_path)
        self.calculator=FeatureCalculator(self.props)
        self.features=self.calculator.compute(train[SITES])
        self.blocks=self.calculator.compute_symmetric_input_from_features(self.features)
        self.train=train
        reference87=self.table('processed/features/training_87d.csv').set_index('physical_id').loc[train.physical_id]
        self.equal('87D_features_from_elements',np.concatenate([self.blocks[k] for k in BLOCKS],axis=1), reference87.drop(columns=SITES+['delta_G_H']).to_numpy(),1e-9)
        swapped=train[SITES].rename(columns={'M3':'M4','M4':'M3'})[SITES]
        sw=self.calculator.compute_symmetric_input_from_features(self.calculator.compute(swapped))
        for block in BLOCKS:self.equal('swap_invariance_'+block,self.blocks[block],sw[block],1e-10)
        self.check('feature_count',sum(x.shape[1] for x in self.blocks.values()),87)
        self.check('ordered_space_size',math.perm(len(self.props),4),57120)
        self.check('physical_space_size',math.perm(len(self.props),4)//2,28560)
        pairs=self.table('processed/symmetry/M3M4_symmetry_pairs.csv')
        deviation=(pairs.delta_G_H_original-pairs.delta_G_H_swapped).abs()*1000
        near=np.minimum(abs(pairs.delta_G_H_original),abs(pairs.delta_G_H_swapped))<=0.1
        self.check('remote_pairs',len(pairs),124)
        self.check('remote_pairs_below_10_meV',(deviation<10).sum(),68)
        self.check('near_thermoneutral_pairs',near.sum(),15)
        self.check('near_median_meV',deviation[near].median(),3.4,0.05)
        self.check('other_median_meV',deviation[~near].median(),6.7,0.05)
        self.check('symmetry_Mann_Whitney_p',stats.mannwhitneyu(deviation[near],deviation[~near],alternative='two-sided').pvalue,0.99,0.005)
        early=self.table('processed/symmetry/early_tm_results.csv')
        self.check('independent_remote_pairs',len(early),40)
        self.check('independent_pairs_below_10_meV',((early.dG_original-early.dG_swapped).abs()<0.01).sum(),29)
        self.save('remote_exchange_deviations',pairs[SITES[:2]].assign(deviation_meV=deviation,near_thermoneutral=near))
        reassessed=self.table('processed/symmetry/verification_results.csv')
        delta=1000*abs(reassessed.dG_original_new-reassessed.dG_swapped_new)
        self.check('reassessed_exchange_pairs',len(reassessed),56)
        self.check('reassessed_pairs_above_10_meV',(delta>10).sum(),47)
        self.save('reassessed_exchange_deviations',reassessed.assign(rebuilt_deviation_meV=delta))

    def architecture(self):
        self.section='R2: model comparisons'
        from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareMLP_V2,SymmetryEnsembleConfigV2
        from herald.models.flat_mlp_ensemble import FlatMLPMember,FlatMLPConfig
        self.check('symmetric_member_parameters',sum(p.numel() for p in SymmetryAwareMLP_V2(SymmetryEnsembleConfigV2(device='cpu')).parameters()),32418)
        self.check('flat_member_parameters',sum(p.numel() for p in FlatMLPMember(FlatMLPConfig(device='cpu')).parameters()),64706)
        folds=self.table('processed/architecture/ablation_strict_2x2_folds.csv')
        summary=folds.groupby(['cell','arch','loss']).mae.agg(['mean','std','count']).reset_index()
        for cell,value in [('Flat_MSE',0.1172),('Flat_NLL',0.1193),('Sym_MSE',0.1085),('Sym_NLL',0.1067)]:
            self.check(cell+'_CV_MAE',summary.loc[summary.cell==cell,'mean'].iloc[0],value,0.00005,'Figure 3b; Table S3')
        self.save('architecture_fold_summary',summary)
        tests=[]
        pivot=folds.pivot(index='fold',columns='cell',values='mae')
        for loss,p in [('NLL',0.003),('MSE',0.020)]:
            result=stats.ttest_rel(pivot['Sym_'+loss],pivot['Flat_'+loss])
            self.check(loss+'_architecture_p',result.pvalue,p,0.0005)
            tests.append(dict(loss=loss,t=result.statistic,p_value=result.pvalue))
        self.save('architecture_paired_tests',pd.DataFrame(tests))
        held=self.table('processed/architecture/arch_advantage_blindtest.csv')
        held['physical_id']=identities(held)
        self.check('heldout_training_overlap',len(set(held.physical_id)&set(self.train.physical_id)),0)
        sym=abs(held.pred_sym-held.dft); flat=abs(held.pred_flat-held.dft)
        self.check('heldout_sym_MAE',sym.mean(),0.1095,0.00005,'Figure 3c')
        self.check('heldout_flat_MAE',flat.mean(),0.1302,0.00005)
        self.check('heldout_improved_compositions',(sym<flat).sum(),72)
        self.check('heldout_paired_p',stats.ttest_rel(sym,flat).pvalue,0.0012,0.00005)
        self.save('heldout_architecture_errors',held[SITES+['physical_id','dft','pred_sym','pred_flat']].assign(ae_sym=sym,ae_flat=flat,error_difference=sym-flat))
        comparison=self.table('processed/architecture/model_comparison_v2.csv').sort_values('mae_mean')
        for i,value in enumerate([0.1058,0.1082,0.1095]):
            self.check(f'model_comparison_rank{i+1}_MAE',comparison.iloc[i].mae_mean,value,0.00005)
        self.save('model_comparison_ranking',comparison.assign(rank=np.arange(1,len(comparison)+1)))

    def acquisition(self):
        self.section='R3: chemical prior and retrospective acquisition'
        prior=PhysicsPrior.calibrate(self.train,self.props)
        score=prior.evaluate(self.train.M1.to_numpy(),self.props)
        hit=abs(self.train.delta_G_H)<=0.2
        top=score>=np.quantile(score,0.8)
        self.check('prior_top_quintile_including_ties',top.sum(),209)
        self.check('prior_top_hits',(top&hit).sum(),146)
        self.check('training_hits_0p2',hit.sum(),253)
        self.check('prior_chi_center',prior.chi_star,1.818,0.0005)
        self.check('prior_chi_width',prior.tau_chi,0.306,0.0005)
        self.save('chemical_prior_scores',self.train[['physical_id','M1','delta_G_H']].assign(chemical_prior=score,top_quintile=top,hit_0p2=hit))
        self.save('chemical_prior_groups',pd.DataFrame([dict(d_electrons=k,hit_fraction=v,normalized_prior=v/max(prior.h_by_nd.values())) for k,v in sorted(prior.h_by_nd.items())]))
        curves=self.table('processed/acquisition/ablation_acquisition_curves.csv')
        self.check('retrospective_round_records',len(curves),80)
        self.check('retrospective_repeats',curves.rep.nunique(),8)
        self.check('retrospective_terminal_size',curves.loc[curves['round']==5,'n_labeled'].min(),700)
        terminal=curves[curves['round']==5]
        result=[]
        for threshold,ph,ucb,p in [(0.2,238.,228.,0.0003),(0.1,119.25,115.50,0.0042)]:
            metric=f'cum_hits_{threshold}'; paired=terminal.pivot(index='rep',columns='arm',values=metric)
            test=stats.ttest_rel(paired.physics,paired.ucb)
            self.check(f'physics_hits_{threshold}',paired.physics.mean(),ph)
            self.check(f'ucb_hits_{threshold}',paired.ucb.mean(),ucb)
            self.check(f'acquisition_p_{threshold}',test.pvalue,p,0.00005,'Figure 4; Table S5')
            if threshold==0.2:self.check('physics_wins_all_repeats',(paired.physics>paired.ucb).sum(),8)
            result.append(dict(threshold_eV=threshold,physics_mean=paired.physics.mean(),physics_sd=paired.physics.std(),ucb_mean=paired.ucb.mean(),ucb_sd=paired.ucb.std(),paired_p=test.pvalue))
        self.save('retrospective_paired_summary',pd.DataFrame(result))
        self.save('retrospective_round_summary',curves.groupby(['arm','round']).agg(n_labeled=('n_labeled','first'),hits_0p2_mean=('cum_hits_0.2','mean'),hits_0p2_sd=('cum_hits_0.2','std'),hits_0p1_mean=('cum_hits_0.1','mean'),hits_0p1_sd=('cum_hits_0.1','std'),best_abs_mean=('cum_best_abs','mean')).reset_index())

    def load_production(self):
        import torch
        from herald.models.symmetry_aware_ensemble_v2 import SymmetryAwareEnsembleV2,SymmetryAwareMLP_V2
        torch.set_num_threads(2)
        # The checkpoint hash is verified by verify_inputs before deserialization.
        checkpoint=torch.load(self.input('models/ensemble/symmetry_ensemble_v2_production_v3.pt'),map_location='cpu',weights_only=False)
        config=replace(checkpoint['config'],device='cpu')
        model=SymmetryAwareEnsembleV2(config)
        model.stats_=checkpoint['stats'];model.y_mean_=checkpoint['y_mean'];model.y_std_=checkpoint['y_std']
        for state in checkpoint['member_states']:
            member=SymmetryAwareMLP_V2(config);member.load_state_dict(state);member.eval();model.members.append(member)
        self.check('production_members',len(model.members),10)
        self.check('symmetry_parameters_per_member',sum(p.numel() for p in model.members[0].parameters()),32418)
        self.model=model

    def screening(self):
        self.section='R4: checkpoint inference and full-space screening'
        self.load_production()
        metals=sorted(self.props)
        sites=[(a,b,c,d) for a,b in itertools.permutations(metals,2) for c,d in itertools.combinations([m for m in metals if m not in {a,b}],2)]
        space=pd.DataFrame(sites,columns=SITES)
        # Recompute descriptors from element properties; no saved predictions are used as inputs.
        features=self.calculator.compute(space)
        blocks=self.calculator.compute_symmetric_input_from_features(features)
        prediction=self.model.predict(blocks)
        space['physical_id']=identities(space)
        space['pred_dG']=prediction.mean;space['epistemic']=np.sqrt(np.maximum(prediction.epistemic,0.))
        space['abs_dG']=abs(space.pred_dG)
        ref=self.table('processed/screening/activity_map_28560_v3.csv')
        ref['physical_id']=identities(ref);ref=ref.set_index('physical_id').loc[space.physical_id]
        self.equal('full_space_checkpoint_prediction_error_eV',space.pred_dG,ref.pred_dG,3e-6)
        self.equal('full_space_checkpoint_uncertainty_error_eV',space.epistemic,ref.epistemic,3e-6)
        threshold=space.epistemic.median()
        hits=space[space.abs_dG<=0.1]
        robust=hits[hits.epistemic<=threshold]
        all3d=set('Sc Ti V Cr Mn Fe Co Ni Cu Zn'.split())
        cheap=robust[robust[SITES].isin(all3d).all(axis=1)].sort_values('abs_dG').copy()
        cheap.insert(0,'Rank',range(1,len(cheap)+1))
        self.check('predicted_hits',len(hits),2428,reference='Figures 1 and 5')
        self.check('low_uncertainty_hits',len(robust),1566)
        self.check('all_3d_candidates',len(cheap),137)
        expected_cheap=self.table('processed/screening/SI_table_cheap_candidates.csv')
        assert list(cheap.physical_id)==identities(expected_cheap), 'All-3d ranking changed'
        self.check('nine_valence_hits',hits.M1.isin(['Co','Rh','Ir']).sum(),1621)
        enrichment=hits.M1.isin(['Co','Rh','Ir']).mean()/space.M1.isin(['Co','Rh','Ir']).mean()
        self.check('nine_valence_enrichment',enrichment,3.78,0.005)
        self.save('full_space_predictions',space.sort_values('abs_dG'))
        self.save('all_3d_candidates',cheap)
        self.save('screening_by_center',space.assign(hit=space.abs_dG<=0.1).groupby('M1').agg(n=('M1','size'),hits=('hit','sum'),hit_fraction=('hit','mean')).reset_index())
        self.space,self.cheap=space,cheap
        held=self.table('processed/architecture/blindtest_v2_predictions_v3.csv')
        pr=self.model.predict(self.calculator.compute_symmetric_input_from_features(self.calculator.compute(held[SITES])))
        self.equal('heldout_production_prediction_error_eV',pr.mean,held.pred,3e-6)
        epi=np.sqrt(np.maximum(pr.epistemic,0.));error=abs(pr.mean-held.dft.to_numpy())
        rho,p=stats.spearmanr(epi,error)
        self.check('uncertainty_error_spearman',rho,0.381,0.0005,'Figure S2')
        self.check('uncertainty_error_p',p,4.1e-5,0.05e-5)
        order=np.argsort(epi,kind='stable');buckets=[]
        # Match pandas qcut: counts 37, 36, 37 for this dataset.
        bucket=pd.qcut(epi,3,labels=['low','medium','high'])
        for name,target in [('low',0.065),('medium',0.088),('high',0.176)]:
            mask=bucket==name;value=error[mask].mean();self.check(name+'_uncertainty_MAE',value,target,0.0005)
            buckets.append(dict(bucket=name,n=int(mask.sum()),mae_eV=value,mean_epistemic_eV=epi[mask].mean()))
        self.save('uncertainty_buckets',pd.DataFrame(buckets))
        self.save('heldout_production_predictions',held[SITES+['dft']].assign(pred=pr.mean,epistemic=epi,absolute_error=error))

    def validation(self):
        self.section='R5: DFT validation'
        val=self.table('processed/dft_validation/verification_results_all60.csv')
        val['physical_id']=identities(val)
        val['seen_training']=val.physical_id.isin(self.train.physical_id)
        val=val.merge(self.space[['physical_id','pred_dG','epistemic']].rename(columns={'pred_dG':'production_prediction'}),on='physical_id',validate='one_to_one')
        original=self.table('processed/dft_validation/verification_results.csv')
        original_ids=set(identities(original));val['prespecified']=val.physical_id.isin(original_ids)
        val['all3d_rank']=val.physical_id.map(self.cheap.set_index('physical_id').Rank)
        val['dft_hit_0p1']=abs(val.delta_G_H)<=0.1
        self.check('validated_compositions',len(val),60)
        unseen=val[val.prespecified & ~val.seen_training]
        self.check('prespecified_unseen',len(unseen),38)
        rho,p=stats.spearmanr(unseen.production_prediction,unseen.delta_G_H)
        self.check('validation_spearman',rho,0.545,0.0005,'Figure 6a')
        self.check('validation_spearman_p',p,3.98e-4,0.005e-4)
        residual=unseen.production_prediction-unseen.delta_G_H
        self.check('validation_RMSE',np.sqrt(np.mean(residual**2)),0.101,0.0005)
        self.check('validation_bias',residual.mean(),-0.0055,0.00005)
        a=val[val.prespecified & val.group.str.startswith('A') & ~val.seen_training]
        b=val[val.prespecified & val.group.str.startswith('B')]
        self.check('unseen_A_size',len(a),18);self.check('unseen_A_hits',a.dft_hit_0p1.sum(),12)
        self.check('A_mean_abs_eV',abs(a.delta_G_H).mean(),0.0775,0.00005)
        self.check('B_hits',b.dft_hit_0p1.sum(),8);self.check('B_mean_abs_eV',abs(b.delta_G_H).mean(),0.0691,0.00005)
        self.check('A_B_Welch_p',stats.ttest_ind(abs(a.delta_G_H),abs(b.delta_G_H),equal_var=False).pvalue,0.670,0.0005)
        top=val[val.all3d_rank<=30]
        self.check('top30_validated',len(top),30);self.check('top30_hits',top.dft_hit_0p1.sum(),20)
        self.check('unseen_top30',(~top.seen_training).sum(),29)
        self.check('unseen_top30_hits',top.loc[~top.seen_training,'dft_hit_0p1'].sum(),19)
        champion=val.loc[abs(val.delta_G_H).idxmin()]
        assert champion.physical_id=='Co-Cu-Fe-Zn'
        self.check('champion_free_energy_eV',champion.delta_G_H,-0.0015,0.00005)
        self.check('champion_rank',champion.all3d_rank,27)
        self.save('validation_membership_and_predictions',val)
        self.save('top30_validation',top.sort_values('all3d_rank'))

    def center_analysis(self):
        self.section='R6: adsorption-center properties and electronic results'
        y=self.train.delta_G_H.to_numpy()
        means=self.train.groupby('M1').delta_G_H.mean()
        between=np.sum((self.train.M1.map(means).to_numpy()-y.mean())**2)
        eta=between/np.sum((y-y.mean())**2)
        self.check('M1_explained_variance_fraction',eta,0.901,0.0005)
        columns=['M1_valence_electrons','M1_atomic_radius','M1_d_band_center','M1_lattice_constant','M1_electronegativity']
        model=lambda:make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=0.1))
        cv=KFold(5,shuffle=True,random_state=42)
        scores=cross_val_score(model(),self.features[columns].to_numpy(),y,cv=cv,scoring='r2')
        self.check('five_attribute_CV_R2',scores.mean(),0.8938,0.00005)
        four=[c for c in columns if c!='M1_d_band_center']
        self.check('four_attribute_CV_R2',cross_val_score(model(),self.features[four].to_numpy(),y,cv=cv,scoring='r2').mean(),0.8585,0.00005)
        held=self.table('processed/design_principle/descriptor_blindtest.csv')
        # The published held-out model uses unscaled polynomial attributes.
        # Its evaluation attributes were stored as float32 in the feature bank.
        fit=make_pipeline(PolynomialFeatures(2,include_bias=False),Ridge(alpha=0.1)).fit(self.features[columns].to_numpy(),y)
        pred=fit.predict(self.calculator.compute(held[SITES])[columns].to_numpy().astype(np.float32).astype(np.float64))
        self.equal('five_attribute_stored_prediction_error',pred,held.pred_5desc,1e-7)
        self.check('five_attribute_heldout_R2',r2_score(held.dft,pred),0.8927,0.00005)
        self.check('five_attribute_heldout_MAE',mean_absolute_error(held.dft,pred),0.1554,0.00005)
        self.save('five_attribute_predictions',held[SITES+['dft']].assign(prediction=pred))
        self.save('five_attribute_CV_scores',pd.DataFrame({'fold':range(5),'R2':scores}))
        obs=self.table('processed/electronic_structure/sc_co_mechanism_observations.csv')
        obs=obs[obs.valid_pair.astype(str).str.lower().eq('true')]
        summary=obs.groupby('M1').agg(n=('M1','size'),H_population=('q_H_ads','mean'),metal_H_A=('M1_H','mean'),protrusion_A=('protrusion','mean'),d_population=('d_occupancy','mean')).reset_index()
        for metal,n,h,bond,z,d in [('Sc',7,1.636,1.866,0.918,1.04),('Co',3,1.058,1.447,0.107,7.41)]:
            row=summary.set_index('M1').loc[metal]
            for name,value,tol in [('n',n,0.),('H_population',h,0.0005),('metal_H_A',bond,0.0005),('protrusion_A',z,0.0005),('d_population',d,0.005)]:
                self.check(metal+'_'+name,row[name],value,tol,'Figure 7d')
        self.save('Sc_Co_electronic_summary',summary)
        # Aggregate stored, composition-resolved SHAP arrays; they are result data.
        shap_rows=[]
        for model_name,m1,global_share in [('Symmetry',40.0,53.6),('FlatMLP',31.5,60.1)]:
            with np.load(self.input(f'data/processed/design_principle/shap_{model_name}_audit_values.npz'),allow_pickle=False) as archive:
                values=archive['values']
            means_abs=np.mean(np.abs(values),axis=0);total=means_abs.sum()
            for block,start,end in [('M1',0,16),('M2',16,29),('M34',29,35),('global',35,87)]:
                share=100*means_abs[start:end].sum()/total
                shap_rows.append(dict(model=model_name,block=block,share_percent=share))
                if block=='M1':self.check(model_name+'_M1_SHAP_percent',share,m1,0.05)
                if block=='global':self.check(model_name+'_global_SHAP_percent',share,global_share,0.05)
        self.save('neural_SHAP_site_shares',pd.DataFrame(shap_rows))
        xgb=self.table('processed/design_principle/shap_xgb_87d_by_site.csv')
        self.check('XGBoost_M1_SHAP_percent',100*xgb.set_index('site').loc['M1','mean_abs_shap']/xgb.mean_abs_shap.sum(),79.0,0.05)

    def descriptors(self):
        self.section='R7: site-resolved empirical expression'
        from empirical import fit_predict
        lock=self.js('models/descriptors/frozen_expressions.json')
        props=lock['elements']
        def axes(frame,unordered=False):
            rows=[]
            for metals in frame[SITES].itertuples(index=False,name=None):
                r=[props[m]['radius_pm'] for m in metals];n=[props[m]['d_electrons'] for m in metals]
                pairs=[]
                for near in ([1,2,3] if unordered else [1]):
                    far=[i for i in [1,2,3] if i!=near]
                    pairs.append([(r[near]+(r[far[0]]+r[far[1]])/2+abs(r[far[0]]-r[far[1]])-2*r[0])/100,
                                  (n[near]-n[0])*((n[far[0]]+n[far[1]])/2-n[near])])
                rows.append(np.mean(pairs,axis=0))
            return np.asarray(rows)
        def predict(frame,x,formula):
            index=[formula['metals'].index(m) for m in frame.M1]
            coef=np.asarray(formula.get('slopes',formula.get('coefficients')))
            return np.asarray(formula['intercepts'])[index]+np.sum(x*coef[index],axis=1)
        reference=self.table('analysis/descriptors/site_role_controls/predictions.csv')
        folds=self.js('data/analysis/descriptors/site_role_controls/outer_models.json')
        splits=self.js('data/analysis/descriptors/validation_protocol/splits.json')
        axis_by_model={'De1_original':axes(self.train),'De1_unordered':axes(self.train,True)}
        outputs=[]
        fitted_models=[]
        for record in folds:
            case,fold,kind=record['case'],record['fold'],record['model']
            test=splits[case][fold]['test'];train=splits[case][fold]['train']
            train_groups={tuple(sorted(x)) for x in self.train.iloc[train][SITES].itertuples(index=False,name=None)}
            test_groups={tuple(sorted(x)) for x in self.train.iloc[test][SITES].itertuples(index=False,name=None)}
            assert not train_groups&test_groups
            frame=self.train.iloc[test];pred=predict(frame,axis_by_model[kind][test],record['formula'])
            rebuilt,coefficients=fit_predict(axis_by_model[kind],self.train.delta_G_H.to_numpy(),
                self.train.M1.to_numpy(),np.asarray(train),np.asarray(test),'independent51',record['alpha'])
            self.equal(f'{case}_{kind}_fold{fold}_refit_error',rebuilt,pred,1e-10)
            fitted_models.append(dict(case=case,fold=fold,model=kind,formula=coefficients))
            ref=reference[(reference['case']==case)&(reference.fold==fold)].set_index('physical_id').loc[frame.physical_id,kind]
            self.equal(f'{case}_{kind}_fold{fold}_prediction_error',pred,ref,1e-10)
            for pid,m,y,p in zip(frame.physical_id,frame.M1,frame.delta_G_H,pred):
                outputs.append(dict(case=case,fold=fold,model=kind,physical_id=pid,M1=m,dft=y,prediction=p,absolute_error=abs(y-p)))
        result=pd.DataFrame(outputs)
        per_center=result.groupby(['case','model','M1']).absolute_error.mean().rename('mae_eV').reset_index()
        macro=per_center.groupby(['case','model']).mae_eV.mean()
        for key,value in [(('primary','De1_original'),0.1181),(('alternative','De1_original'),0.1184),(('primary','De1_unordered'),0.1328),(('alternative','De1_unordered'),0.1310)]:
            self.check('_'.join(key)+'_macro_MAE',macro.loc[key],value,0.00005,'Figure 8; Figure S12')
        self.save('D01_grouped_predictions',result);self.save('D01_per_center_MAE',per_center)
        self.save('D01_macro_MAE',macro.reset_index())
        (self.output/'D01_refitted_fold_models.json').write_text(json.dumps(fitted_models,indent=2)+'\n')
        held=self.table('analysis/descriptors/frozen_expression_validation/record_inventory.csv')
        formula=next(m['formula'] for m in lock['models'] if m['id']=='All_De1_original')
        all_rows=np.arange(len(self.train))
        _,full_fit=fit_predict(axis_by_model['De1_original'],self.train.delta_G_H.to_numpy(),
            self.train.M1.to_numpy(),all_rows,all_rows,'independent51',formula['alpha'])
        self.equal('D01_full_fit_coefficients',full_fit['slopes'],formula['coefficients'],1e-10)
        self.equal('D01_full_fit_intercepts',full_fit['intercepts'],formula['intercepts'],1e-10)
        (self.output/'D01_refitted_full_model.json').write_text(json.dumps(full_fit,indent=2)+'\n')
        pred=predict(held,axes(held),formula)
        median=held.M1.map(self.train.groupby('M1').delta_G_H.median()).to_numpy()
        hits=abs(held.dft.to_numpy())<=0.1
        def expected_hits(scores,k):
            cut=np.sort(abs(scores))[k-1];inside=abs(scores)<cut;ties=abs(scores)==cut
            return hits[inside].sum()+(k-inside.sum())*hits[ties].mean()
        self.check('D01_heldout_MAE',mean_absolute_error(held.dft,pred),0.125739,0.0000005)
        self.check('D01_top22_hits',expected_hits(pred,22),10.)
        self.check('median_top22_expected_hits',expected_hits(median,22),10.2,1e-10)
        self.save('D01_heldout_predictions',held[SITES+['dft']].assign(prediction=pred,center_median_prediction=median))

    def bonding(self):
        self.section='R7: charge, orbital and Co-N bonding responses'
        paired=self.table('reproduction/paired_electronic_endpoints.csv')
        self.check('electronic_pairs',len(paired),129)
        self.check('electronic_compositions',paired.physical_id.nunique(),107)
        paired['delta_dz2_e']=paired.M1_dz2_e_H-paired.M1_dz2_e_clean
        paired['delta_d_e']=paired.M1_d_e_H-paired.M1_d_e_clean
        paired['delta_abs_moment_muB']=paired.M1_moment_abs_muB_H-paired.M1_moment_abs_muB_clean
        self.check('dz2_decreases',(paired.delta_dz2_e<0).sum(),128)
        self.check('d_population_increases',(paired.delta_d_e>0).sum(),127)
        self.check('absolute_moment_decreases',(paired.delta_abs_moment_muB<0).sum(),127)
        self.save('paired_orbital_and_moment_responses',paired)
        atoms=self.table('reproduction/CoN4_atomic_charges.csv')
        charge=[]
        for (pid,source),part in atoms.groupby(['physical_id','source']):
            assert len(part)==5
            dq=part.q_H_e-part.q_clean_e
            charge.append(dict(physical_id=pid,source=source,net_charge_e=dq.sum(),absolute_sum_e=abs(dq).sum(),positive_atoms=(dq>0).sum(),negative_atoms=(dq<0).sum()))
        charges=pd.DataFrame(charge)
        champion=charges[charges.physical_id=='Co-Cu-Fe-Zn'].iloc[0]
        self.check('CoCuFeZn_net_CoN4_charge_e',champion.net_charge_e,-0.0178,0.00005)
        self.check('CoCuFeZn_absolute_CoN4_charge_e',champion.absolute_sum_e,0.2192,0.00005)
        self.check('CoCuFeZn_electron_loss_atoms',champion.positive_atoms,4)
        self.save('CoN4_charge_redistribution',charges)
        bonds=self.table('reproduction/CoN_orbital_endpoints.csv')
        bonds=bonds[bonds.atom1.str.startswith('Co')&bonds.atom2.str.startswith('N')].copy()
        totals=bonds[(bonds.orb1=='total')&(bonds.orb2=='total')]
        summed=totals.groupby(['physical_id','variant','state']).ICOHP_eV.sum().unstack('state')
        summed['B_eV']=summed['H']-summed['clean']
        standard=summed.xs('standard',level='variant')
        for pid,value in [('Co-Cu-Cr-Zn',0.4418),('Co-Cu-Fe-Zn',0.3527),('Co-Mn-Cr-Zn',0.5077),('Co-Mn-Fe-Zn',0.5355)]:
            self.check(pid+'_B_eV',standard.loc[pid,'B_eV'],value,0.00005,'Results R7; Figure S13')
        self.save('CoN_bonding_response',summed.reset_index())
        orbitals=bonds[(bonds.orb1!='total')&(bonds.orb2!='total')].copy()
        orbitals['shell']=orbitals.orb1.str.extract(r'^(\d[spd])')[0]
        response=orbitals.groupby(['physical_id','variant','shell','state']).ICOHP_eV.sum().unstack('state')
        response['response_eV']=response['H']-response['clean']
        contrasts=[]
        for (variant,shell),group in response.reset_index().groupby(['variant','shell']):
            for neighbor in ['Cu','Mn']:
                vals=group.set_index('physical_id').response_eV
                cr,fe=f'Co-{neighbor}-Cr-Zn',f'Co-{neighbor}-Fe-Zn'
                if {cr,fe}<=set(vals.index):contrasts.append(dict(variant=variant,neighbor=neighbor,shell=shell,Fe_minus_Cr_eV=vals[fe]-vals[cr]))
        contrasts=pd.DataFrame(contrasts)
        for neighbor,shell,value in [('Cu','3d',-0.04301),('Mn','3d',-0.04475),('Mn','4s',0.04441),('Mn','4p',0.02794),('Cu','4p',-0.05402)]:
            row=contrasts[(contrasts.variant=='standard')&(contrasts.neighbor==neighbor)&(contrasts.shell==shell)]
            self.check(neighbor+'_'+shell+'_contrast_eV',row.Fe_minus_Cr_eV.iloc[0],value,0.0000051)
        self.save('CoN_orbital_responses',response.reset_index());self.save('CoN_shell_contrasts',contrasts)
        alternative=contrasts[(contrasts.variant=='pv4p')&(contrasts.neighbor=='Mn')].set_index('shell').Fe_minus_Cr_eV
        self.check('second_basis_Mn_3d_negative',alternative['3d']<0,True)
        self.check('second_basis_Mn_4s_positive',alternative['4s']>0,True)
        self.check('second_basis_Mn_4p_positive',alternative['4p']>0,True)
        geometry=self.table('reproduction/CoN_bond_endpoints.csv')
        geometry['elongation_A']=geometry.r_H_A-geometry.r_clean_A
        geometry['B_eV']=geometry.ICOHP_H_eV-geometry.ICOHP_clean_eV
        standard_bonds=geometry[geometry.variant=='standard']
        self.check('negative_clean_CoN_bonds',(standard_bonds.ICOHP_clean_eV<0).sum(),16)
        self.check('weakened_CoN_bonds',(standard_bonds.B_eV>0).sum(),16)
        mean=standard_bonds.groupby('physical_id').elongation_A.mean()
        for neighbor in ['Cu','Mn']:
            self.check(neighbor+'_elongation_decreases',mean[f'Co-{neighbor}-Fe-Zn']<mean[f'Co-{neighbor}-Cr-Zn'],True)
        self.save('CoN_bond_lengths_and_responses',geometry)
        energies=self.table('reproduction/fixed_geometry_energies.csv')
        ads=energies.pivot(index=['geometry_donor','target'],columns='state',values='without_entropy_eV')
        ads['adsorption_electronic_energy_eV']=ads.H-ads.clean
        js=[]
        for geometry,value in [('Co-Cu-Cr-Zn',78.70876),('Co-Cu-Fe-Zn',54.55077)]:
            pair=ads.loc[geometry].adsorption_electronic_energy_eV
            J=1000*(pair['Co-Cu-Fe-Zn']-pair['Co-Cu-Cr-Zn'])
            self.check(geometry+'_fixed_geometry_J_meV',J,value,0.00002)
            js.append(dict(geometry=geometry,J_meV=J))
        self.save('fixed_geometry_adsorption_energies',ads.reset_index());self.save('fixed_geometry_J',pd.DataFrame(js))
        populations=self.table('reproduction/fixed_geometry_Co_populations.csv').merge(energies,on='key',validate='one_to_one')
        populations['other_d_e']=populations.d_XML_actual_occ-populations.dz2_XML_actual_occ
        for geometry,expected in [('Co-Cu-Cr-Zn',0.09469),('Co-Cu-Fe-Zn',0.08443)]:
            group=populations[(populations.geometry_donor==geometry)&(populations.state=='clean')].set_index('target')
            change=group.loc['Co-Cu-Fe-Zn','dz2_XML_actual_occ']-group.loc['Co-Cu-Cr-Zn','dz2_XML_actual_occ']
            self.check(geometry+'_clean_dz2_contrast_e',change,expected,0.000005)
            self.check(geometry+'_other_d_decrease',group.loc['Co-Cu-Fe-Zn','other_d_e']<group.loc['Co-Cu-Cr-Zn','other_d_e'],True)
        self.save('fixed_geometry_Co_populations',populations)
        sensitivity=self.table('reproduction/state_comparison_energies.csv')
        endpoints=sensitivity[(sensitivity.geometry_donor=='Co-Cu-Fe-Zn')&(sensitivity.evaluation=='original')]
        values=endpoints.pivot(index='target',columns='state',values='without_entropy_eV')
        baseline=values.H-values.clean
        I=(baseline['Co-Mn-Fe-Zn']-baseline['Co-Mn-Cr-Zn'])-(baseline['Co-Cu-Fe-Zn']-baseline['Co-Cu-Cr-Zn'])
        restart=sensitivity[sensitivity.evaluation=='restart'].without_entropy_eV.iloc[0]
        ncl=sensitivity[sensitivity.evaluation=='noncollinear'].without_entropy_eV.iloc[0]
        delta=restart-values.loc['Co-Mn-Fe-Zn','clean']
        self.check('restart_clean_energy_change_meV',1000*delta,-168.80494,0.00002,'Figure S14; SI S1.10')
        self.check('background_contrast_original_meV',1000*I,-180.96773,0.00002)
        self.check('background_contrast_one_endpoint_replaced_meV',1000*(I-delta),-12.16279,0.00002)
        self.check('noncollinear_energy_change_meV',1000*(ncl-restart),-1.37447,0.00002)
        self.save('electronic_state_energy_contrasts',pd.DataFrame([dict(restart_delta_meV=1000*delta,
            original_contrast_meV=1000*I,replaced_endpoint_contrast_meV=1000*(I-delta),noncollinear_delta_meV=1000*(ncl-restart))]))

    def verify_inputs(self):
        manifest=CODE/'input_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError('code/input_manifest.json is required; the input inventory is part of the release.')
        for relative,expected in json.loads(manifest.read_text())['files'].items():
            p=self.input(relative)
            if sha(p)!=expected['sha256']:raise ValueError(f'Input checksum mismatch: {relative}')

    def run(self, verify=True):
        if verify:self.verify_inputs()
        self.output.mkdir(parents=True,exist_ok=False)
        started=time.monotonic()
        try:
            for step in [self.training,self.architecture,self.acquisition,self.screening,self.validation,self.center_analysis,self.descriptors,self.bonding]:
                print(f'Running {step.__name__} ...',flush=True);step()
            status='pass'
        except Exception:
            self.save('manuscript_checks',pd.DataFrame(self.checks))
            raise
        self.save('manuscript_checks',pd.DataFrame(self.checks))
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','scikit-learn','h5py']}
        report=dict(status=status,checks=len(self.checks),all_checks_passed=all(x['passed'] for x in self.checks),
                    elapsed_seconds=time.monotonic()-started,python=sys.version.split()[0],packages=packages,
                    output_tables=len(list(self.output.glob('*.csv'))),input_files=len(self.inputs),device='cpu')
        (self.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (self.output/'input_files.json').write_text(json.dumps({'files':self.inputs},indent=2)+'\n')
        print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--models',type=Path,default=ROOT/'models' if (ROOT/'models').is_dir() else ROOT/'data/models')
    p.add_argument('--output',type=Path,default=ROOT/'results/nc_reproduction' if (CODE/'src').is_dir() else ROOT/'outputs/nc_reproduction')
    args=p.parse_args()
    output=args.output.resolve()
    for protected in [args.data.resolve(),args.models.resolve(),CODE.resolve(),LIB.resolve()]:
        if output==protected or output.is_relative_to(protected):p.error('Output must be outside the input and code directories.')
    Reproduction(args.data.resolve(),args.models.resolve(),output).run()


if __name__=='__main__':main()
