import numpy as np
from pyedflib import highlevel
import os
from datetime import datetime

# ── 1) Scale back to ~16K windows ────────────────────────────────────────────
N_SUBJECTS      = 16     # 16 subjects
WINDOWS_PER_SUB = 1000   # 16 × 1 000 = 16 000 total windows
SEIZURE_FRAC    = 0.07   # ~7% seizure windows

SAMPLE_RATE     = 200    # Hz
WINDOW_SEC      = 4      # seconds
SAMPLES_PER_WIN = SAMPLE_RATE * WINDOW_SEC

BASE_DIR = './synthetic_dataset'

# ── 2) Signal generators ───────────────────────────────────────────────────
def make_background(channels, samples):
    t = np.arange(samples) / SAMPLE_RATE
    sig = np.zeros((channels, samples))
    for fmin, fmax in [(1,4),(4,8),(8,12),(12,30)]:
        freq = np.random.uniform(fmin, fmax)
        amp  = np.random.uniform(5, 20)
        sig += amp * np.sin(2*np.pi*freq * t)[None,:]
    # pink noise
    fft = np.fft.rfft(np.random.randn(samples))
    ps = fft / np.maximum(np.fft.rfftfreq(samples), 1e-6)
    pink = np.fft.irfft(ps)
    sig += pink[None,:] * np.random.uniform(0.5, 2.0)
    sig += np.random.randn(channels, samples) * 2.0
    return sig

def make_seizure(channels, samples):
    sig = make_background(channels, samples)
    t = np.arange(samples) / SAMPLE_RATE
    burst_len = np.random.randint(int(0.5*SAMPLE_RATE), int(1.5*SAMPLE_RATE))
    start = np.random.randint(0, samples - burst_len)
    for ch in range(channels):
        freq = np.random.uniform(20, 50)
        amp  = np.random.uniform(50, 100)
        sig[ch, start:start+burst_len] += amp * np.sin(2*np.pi*freq * t[:burst_len])
    return sig

# ── 3) EDF header template ─────────────────────────────────────────────────
from pyedflib import highlevel
channel_labels = [
    'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
    'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
    'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
]
n_ch = len(channel_labels)

edf_header = {
    'technician': 'Synthetic',
    'recording_additional': 'Synth EEG',
    'patientname': 'TestPatient',
    'patientcode': '00001',
    'equipment': 'SyntheticEEG',
    'gender': 'M',
    'startdate': datetime.now()
}

# ── 4) Main loop ────────────────────────────────────────────────────────────
for subj in range(1, N_SUBJECTS+1):
    for win in range(WINDOWS_PER_SUB):
        is_seiz = (np.random.rand() < SEIZURE_FRAC)
        sig     = make_seizure(n_ch, SAMPLES_PER_WIN) if is_seiz else make_background(n_ch, SAMPLES_PER_WIN)

        # compute integer physical ranges to avoid 8‑char warnings
        phys_min = int(np.floor(sig.min()))
        phys_max = int(np.ceil (sig.max()))

        channel_info = []
        for ch in channel_labels:
            channel_info.append({
                'label'           : ch,
                'dimension'       : 'uV',
                'sample_frequency': SAMPLE_RATE,
                'physical_min'    : phys_min,
                'physical_max'    : phys_max,
                'digital_min'     : -32768,
                'digital_max'     :  32767,
                'transducer'      : '',
                'prefilter'       : ''
            })

        # directory & filenames
        d = os.path.join(BASE_DIR,
                         'train',
                         f'{subj:03d}_session',
                         f'{subj:03d}_win{win:04d}')
        os.makedirs(d, exist_ok=True)

        edf_fname = os.path.join(d, f'{subj:03d}_win{win:04d}.edf')
        tse_fname = edf_fname.replace('.edf', '.tse_bi')

        # write EDF + labels
        highlevel.write_edf(edf_fname, sig, channel_info, edf_header)

        with open(tse_fname, 'w') as f:
            f.write("version = tse_v1.0.0\nstart_time = 0.0000\n")
            lbl = 'seiz' if is_seiz else 'bckg'
            f.write(f"0.0000 {WINDOW_SEC:.4f} {lbl} 1.0000\n")