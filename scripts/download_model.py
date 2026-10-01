"""Resumable mirror downloader; validates ranges and full LFS SHA256 before publication."""
import argparse
import shutil
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time
import requests

parser=argparse.ArgumentParser()
parser.add_argument('--repo', default='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice')
parser.add_argument('--out', type=Path, default=Path('.local/models/qwen3-tts'))
parser.add_argument('--reuse', type=Path, help='Reuse matching verified weights from another local model')
parser.add_argument('--workers', type=int, default=12)
args=parser.parse_args()
repo=args.repo
session=requests.Session(); session.trust_env=False
meta=session.get(f'https://hf-mirror.com/api/models/{repo}/revision/main?blobs=true',timeout=30)
meta.raise_for_status(); meta=meta.json(); revision=meta['sha']
root=args.out; root.mkdir(parents=True,exist_ok=True)
(root/'source.json').write_text(json.dumps(dict(repo=repo,revision=revision),indent=2))
for f in meta['siblings']:
    name=f['rfilename']; dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
    if 'lfs' not in f:
        r=session.get(url,timeout=60);r.raise_for_status();dest.write_bytes(r.content);continue
    digest=f['lfs']['sha256']; size=f['size']
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==digest:continue
    if args.reuse:
        existing=args.reuse/name
        if existing.exists() and existing.stat().st_size==size and hashlib.sha256(existing.read_bytes()).hexdigest()==digest:
            shutil.copyfile(existing,dest);print('reused verified',name,digest,flush=True);continue
    parts=root/(name.replace('/','_')+'.parts');parts.mkdir(exist_ok=True)
    block=4*1024*1024
    def fetch(start):
        end=min(start+block,size)-1;path=parts/str(start)
        if path.exists() and path.stat().st_size==end-start+1:return
        for attempt in range(8):
            try:
                with requests.Session() as s:
                    s.trust_env=False
                    with s.get(url+f'?range_start={start}&attempt={attempt}',
                               headers={'Range':f'bytes={start}-{end}'},
                               timeout=(20,90),stream=True) as r:
                        r.raise_for_status()
                        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                            raise ValueError('Incorrect partial response headers')
                        content=r.content
                        if len(content)!=end-start+1:
                            raise ValueError('Incorrect partial response length')
                path.write_bytes(content);return
            except Exception as exc:
                print(f'retry {name} range={start}-{end} attempt={attempt+1}: {type(exc).__name__}', flush=True)
                if attempt==7:raise
                time.sleep(min(2**attempt,10))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(fetch,start) for start in range(0,size,block)]
        for i,future in enumerate(cf.as_completed(tasks)):
            future.result()
            if i%20==0:print(name,i+1,'/',len(tasks),flush=True)
    tmp=dest.with_suffix('.partial');h=hashlib.sha256()
    with tmp.open('wb') as out:
        for start in range(0,size,block):
            b=(parts/str(start)).read_bytes(); h.update(b);out.write(b)
    if h.hexdigest()!=digest:raise ValueError('Checksum mismatch')
    os.replace(tmp,dest)
    for part in parts.iterdir():part.unlink()
    parts.rmdir(); print('verified',name,digest,flush=True)
print('model ready',root,revision,flush=True)
