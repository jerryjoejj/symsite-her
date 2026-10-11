#!/usr/bin/env python3
"""Export a standalone Code Ocean capsule from the explicit input inventory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/codeocean')
    args=parser.parse_args()
    output=args.output.resolve()
    if output==ROOT or any(output.is_relative_to(ROOT/x) for x in ['code','data','models','src','docs','environment','tools','provenance']):
        parser.error('Choose a new export directory outside the source and input folders.')
    if output.exists():parser.error('Export directory already exists; choose a new --output directory.')
    manifest=json.loads((ROOT/'code/input_manifest.json').read_text())
    for relative,record in manifest['files'].items():
        source=ROOT/relative
        if not source.resolve().is_relative_to(ROOT) or source.is_symlink():raise ValueError(relative)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Input checksum mismatch: '+relative)
    capsule=output/'capsule'
    capsule.mkdir(parents=True)
    for relative in sorted(manifest['files']):
        target=capsule/('data/'+relative if relative.startswith('models/') else relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    shutil.copytree(ROOT/'code',capsule/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'src',capsule/'code/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'docs',capsule/'code/docs')
    shutil.copytree(ROOT/'environment',capsule/'environment')
    (capsule/'metadata').mkdir()
    metadata=dict(title='SymSite-HER: symmetry-guided discovery and cooperative site effects',
        description='Reproduction of final training datasets, ensemble screening, empirical expressions and Co-N bonding responses for multimetallic single-atom HER catalysts.',
        authors=['Tongtong Li','Jianning Tong','Sen Han'],language='Python',
        run_script='code/run',results_directory='results/nc_reproduction',
        manuscript=json.loads((ROOT/'provenance/manuscript_snapshot.json').read_text()))
    (capsule/'metadata/capsule.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (capsule/'code/README.md').write_text('# SymSite-HER capsule\n\nRun `bash run` from the code directory. Results are written to `../results/nc_reproduction/`.\n\nSee [reproduction procedures](docs/REPRODUCIBILITY.md), [training](docs/TRAINING.md) and [data definitions](docs/DATA_DICTIONARY.md).\n')
    for path in [capsule/'code/run',capsule/'environment/postInstall']:path.chmod(0o755)
    inventory={p.relative_to(capsule).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(capsule.rglob('*')) if p.is_file()}
    (capsule/'metadata/file_checksums.json').write_text(json.dumps(inventory,indent=2)+'\n')
    archive=output/'symsite-her-codeocean.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(capsule.rglob('*')):
            if path.is_file():
                info=zipfile.ZipInfo(path.relative_to(capsule).as_posix(),date_time=(2026,10,10,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=path.stat().st_mode<<16
                z.writestr(info,path.read_bytes())
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (output/'symsite-her-codeocean.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(capsule=str(capsule),archive=str(archive),scientific_inputs=len(manifest['files']),
                         archive_bytes=archive.stat().st_size,archive_sha256=digest),indent=2))


if __name__=='__main__':main()
