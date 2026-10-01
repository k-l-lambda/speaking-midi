"""Renderer-constrained speech approximation, with discrete temporal matching pursuit."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import ctypes
import ctypes.util
import numpy as np
import scipy.signal as signal
import soundfile as sf
import mido

# pyFluidSynth's discovery uses ldconfig; support an unprivileged local deb extraction.
_local = Path(__file__).resolve().parents[1]/'.local/usr/lib/x86_64-linux-gnu/libfluidsynth.so.3'
if _local.exists():
    _find = ctypes.util.find_library
    ctypes.util.find_library = lambda name: str(_local) if name == 'fluidsynth' else _find(name)
import fluidsynth

SR = 24000
HOP = 240
NFFT = 2048
SFONT = '/home/camus/work/LilyScript/web/soundfont/gm.sf3'

@dataclass
class Note:
    pitch: int
    velocity: int
    start: float
    duration: float

class Piano:
    def __init__(self, soundfont=SFONT):
        self.path = str(soundfont)
        self.synth = fluidsynth.Synth(samplerate=SR, gain=0.8)
        for key, value in [('synth.reverb.active', 0), ('synth.chorus.active', 0),
                           ('synth.polyphony', 512), ('synth.cpu-cores', 1)]:
            self.synth.setting(key, value)
        self.sfid = self.synth.sfload(self.path)
        if self.sfid < 0: raise RuntimeError('SoundFont load failed')
        if self.synth.program_select(0, self.sfid, 0, 0) != 0:
            raise RuntimeError('Cannot select bank 0, program 0')
        self.synth.noteon(0,60,1)
        self.synth.get_samples(256)
        self.synth.all_sounds_off(0)
        self.synth.get_samples(256)

    def render(self, notes, seconds):
        self.synth.all_sounds_off(0)
        self.synth.get_samples(256)  # flush voices killed at the next synthesis block
        events=[]
        for n in notes:
            events.extend([(round(n.start*SR), 1, n.pitch, n.velocity),
                           (round((n.start+n.duration)*SR), 0, n.pitch, 0)])
        # Note-off before note-on at a shared boundary.
        events.sort()
        total = round(seconds*SR); chunks=[]; cursor=0
        for sample, on, pitch, velocity in events:
            sample=min(sample,total)
            if sample>cursor:
                chunks.append(self.synth.get_samples(sample-cursor).reshape(-1,2).mean(axis=1)/32768.)
                cursor=sample
            if on: self.synth.noteon(0,pitch,velocity)
            else: self.synth.noteoff(0,pitch)
        if total>cursor:
            chunks.append(self.synth.get_samples(total-cursor).reshape(-1,2).mean(axis=1)/32768.)
        # Drain the remaining FluidSynth 64-frame block before the next reset.
        # Otherwise buffered samples from this render may leak into the next trial.
        if total % 64: self.synth.get_samples(64-total%64)
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(total,np.float32)

    def close(self): self.synth.delete()

def write_midi(notes, path):
    midi=mido.MidiFile(ticks_per_beat=1000); track=mido.MidiTrack(); midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    track.append(mido.Message('program_change', channel=0, program=0))
    events=[]
    for n in notes:
        events.extend([(round(n.start*2000),1,n.pitch,n.velocity),
                       (round((n.start+n.duration)*2000),0,n.pitch,0)])
    last=0
    for tick,on,pitch,vel in sorted(events):
        track.append(mido.Message('note_on' if on else 'note_off', channel=0, note=pitch, velocity=vel, time=tick-last)); last=tick
    midi.save(path)

def read_midi(path):
    now=0.; active={}; notes=[]
    for msg in mido.MidiFile(path):
        now+=msg.time
        if msg.type=='note_on' and msg.velocity:
            if msg.note in active: raise ValueError('Overlapping same-pitch MIDI notes')
            active[msg.note]=(now,msg.velocity)
        elif msg.type=='note_off' or (msg.type=='note_on' and not msg.velocity):
            start,vel=active.pop(msg.note)
            notes.append(Note(msg.note,vel,start,now-start))
    if active: raise ValueError('Unterminated MIDI notes')
    return notes

def load_audio(path):
    y,sr=sf.read(path,always_2d=True); y=y.mean(axis=1)
    if sr!=SR:
        import math
        g=math.gcd(sr,SR); y=signal.resample_poly(y,SR//g,sr//g)
    if not np.isfinite(y).all() or np.max(np.abs(y))<1e-8: raise ValueError('Invalid/silent audio')
    # Fixed target RMS: all methods share this exact signal; no per-result normalization.
    return (y*(0.035/np.sqrt(np.mean(y*y)))).astype(np.float32)

def magnitude(y):
    import torch
    x=torch.as_tensor(y,dtype=torch.float32)
    return torch.stft(x,NFFT,HOP,window=torch.hann_window(NFFT),center=True,
                      pad_mode='constant',return_complex=True).abs().numpy()

def filterbank():
    # 96 triangular bands with more resolution in the speech range.
    hz=np.linspace(0,SR/2,NFFT//2+1)
    edges=700*(np.exp(np.linspace(np.log(1+60/700),np.log(1+10000/700),98))-1)
    bank=np.maximum(0,np.minimum((hz[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
                                (edges[2:,None]-hz[None])/(edges[2:,None]-edges[1:-1,None])))
    return bank.astype(np.float32)

def features(y): return filterbank()@magnitude(y)

def make_dictionary(piano, cache):
    config=dict(soundfont_sha256=hashlib.sha256(Path(piano.path).read_bytes()).hexdigest(),
                sr=SR,hop=HOP,nfft=NFFT,preset_name=piano.synth.sfpreset_name(piano.sfid,0,0),
                renderer_library_sha256=hashlib.sha256(_local.read_bytes()).hexdigest() if _local.exists() else 'system',pitches=list(range(36,109)),velocities=[45,75,105],
                durations=[.06,.12,.24],tail=.30,gain=.8,bank=0,program=0,effects=False,renderer_revision=3)
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path=Path(cache)/f'dictionary-{key}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        a=np.load(path); return a['atoms'], a['params'],config
    atoms=[]; params=[]
    for pitch in config['pitches']:
        for vel in config['velocities']:
            for dur in config['durations']:
                y=piano.render([Note(pitch,vel,0,dur)],.54)
                atoms.append(features(y)); params.append((pitch,vel,dur))
    atoms=np.stack(atoms); params=np.asarray(params)
    np.savez_compressed(path,atoms=atoms,params=params)
    return atoms,params,config

def fit_temporal(y, atoms, params, max_notes=240):
    import torch
    import torch.nn.functional as F
    device='cuda' if torch.cuda.is_available() else 'cpu'
    target=features(y)
    # Inverse RMS band weights cap dominance by the strongest formant.
    weight=1/np.maximum(np.sqrt(np.mean(target**2,axis=1)),.15)**.5
    a=torch.tensor(atoms*weight[None,:,None],device=device)
    residual=torch.tensor(target*weight[:,None],device=device)
    length=a.shape[-1]; frames=residual.shape[-1]
    residual=F.pad(residual,(0,length))
    energy=a.square().sum((1,2))
    allowed=torch.ones((len(params),frames),device=device,dtype=torch.bool)
    notes=[]; history=[]
    with torch.no_grad():
        for iteration in range(max_notes):
            dots=F.conv1d(residual[None],a)[0,:,:frames]
            score=2*dots-energy[:,None]
            score.masked_fill_(~allowed,-float('inf'))
            best,idx=score.flatten().max(0)
            if best.item()<=0: break
            k,t=divmod(idx.item(),frames)
            pitch,vel,dur=params[k]
            if t*HOP/SR+dur>len(y)/SR:
                allowed[k,t]=False; continue
            notes.append(Note(int(pitch),int(vel),t*HOP/SR,float(dur)))
            residual[:,t:t+length]-=a[k]
            # Reject note-on overlaps at same pitch, including preceding long candidates.
            same=np.where(params[:,0]==pitch)[0]
            for j in same:
                left=max(0,t-int(np.ceil(params[j,2]*SR/HOP))+1)
                right=min(frames,t+int(np.ceil(dur*SR/HOP)))
                allowed[j,left:right]=False
            history.append(float(best))
            if len(notes)%40==0: print('pursuit',len(notes),'improvement',float(best),flush=True)
    return notes,dict(accepted=len(notes),surrogate_improvements=history,device=device)

def baseline(y, max_polyphony=8):
    spec=magnitude(y); freq=np.arange(spec.shape[0])*SR/NFFT
    notes=[]; step=6
    for t in range(0,spec.shape[1]-step,step):
        s=spec[:,t:t+step].mean(axis=1)
        peaks,_=signal.find_peaks(s)
        peaks=peaks[(freq[peaks]>=65)&(freq[peaks]<=4200)]
        selected=peaks[np.argsort(s[peaks])[-max_polyphony:]]
        used=set()
        for k in selected:
            if s[k]<max(.15,spec.max()*.015):continue
            p=int(round(69+12*np.log2(freq[k]/440)))
            if p in used or not 36<=p<=108:continue
            used.add(p)
            vel=int(np.clip(90+25*np.log10(max(s[k]/spec.max(),1e-6)),30,110))
            notes.append(Note(p,vel,t*HOP/SR,step*HOP/SR))
    return notes

def make_log_band_loss(y):
    """Cache fixed target analysis for repeated renderer trials."""
    target_log = np.log1p(features(y))
    samples = len(y)

    def loss(estimate):
        estimate_log = np.log1p(features(estimate[:samples]))
        return float(np.mean(np.abs(target_log - estimate_log)))

    return loss

def metrics(y, estimate):
    estimate=estimate[:len(y)]
    a=magnitude(y); b=magnitude(estimate)
    fa=filterbank()@a; fb=filterbank()@b
    return dict(spectral_convergence=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                log_band_mae=float(np.mean(np.abs(np.log1p(fa)-np.log1p(fb)))),
                waveform_rmse=float(np.sqrt(np.mean((y-estimate)**2))),
                peak=float(np.max(np.abs(estimate))))

def render_saved(piano,notes,path,seconds):
    write_midi(notes,path.with_suffix('.mid'))
    # Metrics use the exported MIDI read back from disk, not the surrogate.
    decoded=read_midi(path.with_suffix('.mid'))
    audio=piano.render(decoded,seconds)
    sf.write(path.with_suffix('.wav'),audio,SR,subtype='FLOAT')
    path.with_suffix('.json').write_text(json.dumps([asdict(n) for n in decoded],indent=2))
    return audio
