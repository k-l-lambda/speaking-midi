"""Measure isolated high-register piano energy, without labelling it all noise."""
import json
from pathlib import Path
import numpy as np
from speaking_midi.core import Piano,read_midi,SR
from experiments.chime_objective import OUT


def main():
    rows=json.loads((OUT/'results.json').read_text())
    for row in rows:
        ns=read_midi(OUT/f"{row['song']}-{row['start']}-{row['variant']}.mid")
        row['register_rms']={}
        for register,selected in [('body',[n for n in ns if n.pitch<84]),('high',[n for n in ns if n.pitch>=84])]:
            p=Piano()
            try:
                y=p.render(selected,12.5)[:12*SR]
                row['register_rms'][register]=float(np.sqrt(np.mean(y*y)))
            finally:p.close()
    for row in rows:
        base=next(r for r in rows if r['song']==row['song'] and r['start']==row['start'] and r['variant']=='baseline')
        row['register_change_db']={k:float(20*np.log10(max(row['register_rms'][k],1e-12)/max(base['register_rms'][k],1e-12))) for k in ['body','high']}
        print(row['song'],row['start'],row['variant'],row['register_change_db'],flush=True)
    (OUT/'audited-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
