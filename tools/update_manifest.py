#!/usr/bin/env python3
"""Refresh the release file inventory and checksums after reviewed edits."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','outputs','results','.venv','__pycache__','.pytest_cache'}


def release_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
                  and p.suffix!='.pyc')


def main():
    scientific=json.loads((ROOT/'code/input_manifest.json').read_text())['files']
    for name,expected in scientific.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Scientific input changed; review its provenance and input manifest: '+name)
    tables=[]
    for name in scientific:
        if name.endswith('.csv'):
            with (ROOT/name).open(newline='',encoding='utf-8-sig') as f:
                reader=csv.reader(f);columns=next(reader);count=sum(1 for _ in reader)
            tables.append(dict(path=name,rows=count,columns='|'.join(columns)))
    with (ROOT/'provenance/table_index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','rows','columns'],lineterminator='\n');writer.writeheader();writer.writerows(tables)
    skip={'SHA256SUMS','provenance/file_manifest.csv'}
    rows=[]
    for p in release_files():
        name=p.relative_to(ROOT).as_posix()
        if name in skip:continue
        rows.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         role=scientific.get(name,{}).get('role','code_or_documentation')))
    with (ROOT/'provenance/file_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256','role'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()
           for p in release_files() if p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(hashed_files=len(lines),scientific_inputs=len(scientific),csv_tables=len(tables))))


if __name__=='__main__':main()
