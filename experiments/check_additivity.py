import numpy as np,time
from speaking_midi.core import *
p=Piano();notes=read_midi('outputs/poetry-backtest/quiet-night/quick.mid');seconds=9.14
actual=p.render(notes,seconds);summed=np.zeros_like(actual);start=time.monotonic()
for n in notes:
 sample=round(n.start*SR);offset=sample%64;begin=sample-offset
 dur=round(n.duration*SR)/SR
 wave=p.render([Note(n.pitch,n.velocity,offset/SR,dur)],offset/SR+dur+.8)
 end=min(len(summed),begin+len(wave));summed[begin:end]+=wave[:end-begin]
print('additive metrics',metrics(actual,summed),'seconds',time.monotonic()-start)
print('actual target',metrics(load_audio('outputs/poetry-backtest/quiet-night/source.wav'),actual))
print('summed target',metrics(load_audio('outputs/poetry-backtest/quiet-night/source.wav'),summed))
p.close()
