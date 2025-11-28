import mne
import numpy as np
from pathlib import Path
from scipy import signal
from tqdm import tqdm

class EEGDataLoader:
    def __init__(self, base_path='eegDataset', runs_per_subject=3):
        self.base_path = Path(base_path)
        self.runs_per_subject = runs_per_subject
    
    def load_subject(self, subject_id):
        subject_dir = self.base_path / f'S{subject_id:03d}'
        raw_list = []
        target_sfreq = None
        
        for run in range(1, self.runs_per_subject + 1):
            filepath = subject_dir / f'S{subject_id:03d}R{run:02d}.edf'
            if not filepath.exists():
                continue
                
            raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
            
            if target_sfreq is None:
                target_sfreq = raw.info['sfreq']
            elif raw.info['sfreq'] != target_sfreq:
                raw.resample(target_sfreq, verbose=False)
            
            raw_list.append(raw)
        
        if raw_list:
            return mne.concatenate_raws(raw_list, verbose=False)
        return None

class EEGPreprocessor:
    def __init__(self, lowcut=1.0, highcut=40.0):
        self.lowcut = lowcut
        self.highcut = highcut
    
    def filter_signal(self, raw):
        raw.filter(l_freq=self.lowcut, h_freq=self.highcut, verbose=False)
        return raw
    
    def extract_eeg_channels(self, raw):
        eeg_channels = [ch for ch in raw.ch_names 
                       if 'EEG' in ch or ch.startswith(('Fp', 'F', 'C', 'P', 'O', 'T'))]
        if eeg_channels:
            raw.pick_channels(eeg_channels, verbose=False)
        return raw
    
    def set_reference(self, raw):
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)
        return raw
    
    def preprocess(self, raw):
        raw = self.extract_eeg_channels(raw)
        raw = self.filter_signal(raw)
        raw = self.set_reference(raw)
        return raw

class FeatureExtractor:
    def __init__(self, segment_length=4.0, overlap=2.0, 
                 window_size=256, window_overlap=128):
        self.segment_length = segment_length
        self.overlap = overlap
        self.window_size = window_size
        self.window_overlap = window_overlap
    
    def segment_data(self, raw, max_segments=5):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        segment_samples = int(self.segment_length * sfreq)
        overlap_samples = int(self.overlap * sfreq)
        step = segment_samples - overlap_samples
        
        segments = []
        for start in range(0, data.shape[1] - segment_samples + 1, step):
            segments.append(data[:, start:start + segment_samples])
            if len(segments) >= max_segments:
                break
        
        return np.array(segments)
    
    def compute_spectrograms(self, segments, sfreq):
        spectrograms = []
        
        for segment in segments:
            segment_spectrograms = []
            for channel_data in segment:
                f, t, Sxx = signal.spectrogram(
                    channel_data,
                    fs=sfreq,
                    nperseg=self.window_size,
                    noverlap=self.window_overlap
                )
                segment_spectrograms.append(Sxx)
            spectrograms.append(np.array(segment_spectrograms))
        
        return np.array(spectrograms)
    
    def normalize(self, spectrograms):
        mean = spectrograms.mean()
        std = spectrograms.std()
        return (spectrograms - mean) / (std + 1e-8)
    
    def extract_features(self, raw, max_segments=5):
        segments = self.segment_data(raw, max_segments)
        sfreq = raw.info['sfreq']
        spectrograms = self.compute_spectrograms(segments, sfreq)
        return self.normalize(spectrograms)

def process_dataset(num_subjects=109, base_path='eegDataset', 
                   runs_per_subject=3, max_segments=5):
    loader = EEGDataLoader(base_path, runs_per_subject)
    preprocessor = EEGPreprocessor()
    extractor = FeatureExtractor()
    
    X_list = []
    y_list = []
    
    for subject_id in tqdm(range(1, num_subjects + 1), desc='Processing subjects'):
        raw = loader.load_subject(subject_id)
        if raw is None:
            continue
        
        raw = preprocessor.preprocess(raw)
        features = extractor.extract_features(raw, max_segments)
        
        X_list.extend(features)
        y_list.extend([subject_id - 1] * len(features))
    
    return {
        'X': np.array(X_list),
        'y': np.array(y_list),
        'num_subjects': num_subjects
    }
