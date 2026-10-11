#!/usr/bin/env python3
"""Check the complete release inventory, scientific inputs and Python syntax."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from update_manifest import ROOT,release_files


def main():
    actual={p.relative_to(ROOT).as_posix():p for p in release_files() if p.name!='SHA256SUMS'}
    expected={name:digest for digest,name in (line.split('  ',1) for line in (ROOT/'SHA256SUMS').read_text().splitlines())}
    assert set(actual)==set(expected), {'missing':sorted(set(expected)-set(actual)),'unlisted':sorted(set(actual)-set(expected))}
    forbidden=re.compile(r'^(INCAR|KPOINTS|POSCAR|CONTCAR|POTCAR|OUTCAR|OSZICAR|CHGCAR|WAVECAR|PROCAR|DOSCAR|EIGENVAL|AECCAR|LOCPOT|ELFCAR|XDATCAR|vasprun|vaspout)(\.|$)',re.I)
    syntax=0
    for name,p in actual.items():
        assert not p.is_symlink(),name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[name],name
        assert not forbidden.search(p.name),name
        assert p.suffix.lower() not in {'.pdf','.png','.jpg','.jpeg','.svg','.xml','.vasp','.cif','.zip','.gz','.lobster'},name
        if p.suffix in {'.py','.md','.json','.csv','.toml','.txt','.cff'} or p.name in {'run','postInstall','Dockerfile'}:
            text=p.read_text(encoding='utf-8-sig')
            assert not re.search('[\u3400-\u4dbf\u4e00-\u9fff]',text),name
            if p.resolve()!=Path(__file__).resolve():
                assert not re.search(r'history_research|next_plan|/home/lma_',text),name
            if p.suffix=='.py':ast.parse(text,filename=name);syntax+=1
    inputs=json.loads((ROOT/'code/input_manifest.json').read_text())['files']
    for name,record in inputs.items():
        assert name in expected and record['sha256']==expected[name],name
        assert actual[name].stat().st_size==record['bytes'],name
    for path in ['data/raw','data/dft_templates','figures','scripts/figures']:
        assert not (ROOT/path).exists(),path
    with (ROOT/'provenance/file_manifest.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            assert expected[row['path']]==row['sha256'],row['path']
            assert actual[row['path']].stat().st_size==int(row['bytes']),row['path']
    print(json.dumps(dict(status='pass',hashed_files=len(expected),scientific_inputs=len(inputs),
                         python_files_parsed=syntax,raw_calculation_files=0,chinese_text_files=0),indent=2))


if __name__=='__main__':main()
