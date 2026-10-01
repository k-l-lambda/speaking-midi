"""Auxiliary, unprompted ASR check; not a substitute for blind listening."""
import argparse
import json
import re
from pathlib import Path
from faster_whisper import WhisperModel

parser=argparse.ArgumentParser(); parser.add_argument('--model-path'); args=parser.parse_args()
path=args.model_path
if not path:
    cache=Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base'
    revision=(cache/'refs/main').read_text().strip()
    path=str(cache/'snapshots'/revision)
model=WhisperModel(path,device='cpu',compute_type='int8',cpu_threads=4)
manifest=json.loads(Path('outputs/tts/manifest.json').read_text())

def units(text, language):
    if language=='Chinese': return [c for c in text if c.isalnum()]
    return re.findall(r"[a-z0-9]+",text.lower())

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        new=[i+1]
        for j,y in enumerate(b):new.append(min(new[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=new
    return row[-1]

results=[]
for sample in manifest['samples']:
    name=sample['name']; language=sample['language']; ref=units(sample['text'],language)
    for variant in ['target','baseline','dictionary','refined']:
        audio=Path('outputs')/name/f'{variant}.wav'
        segments,info=model.transcribe(str(audio),language='zh' if language=='Chinese' else 'en',
                        beam_size=5,condition_on_previous_text=False,vad_filter=False)
        text=''.join(s.text for s in segments).strip()
        row=dict(sample=name,variant=variant,reference=sample['text'],hypothesis=text,
                 metric='CER' if language=='Chinese' else 'WER',
                 error_rate=edit_distance(ref,units(text,language))/len(ref))
        results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
Path('outputs/asr.json').write_text(json.dumps(dict(model=path,results=results,
    caveat='Single small ASR model; no prompting, no human listening. Error rate can exceed 1.'),ensure_ascii=False,indent=2))
