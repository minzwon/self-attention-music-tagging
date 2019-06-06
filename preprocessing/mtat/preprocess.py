import os
import numpy as np
import glob
from essentia.standard import MonoLoader
import librosa
import fire
import tqdm


class Processor:
    def __init__(self):
        self.fs = 16000
        self.window = 512
        self.hop = 256
        self.mel = 96

    def get_paths(self, audio_path, data_path):
        self.files = glob.glob(os.path.join(audio_path, '*/*.mp3'))
        self.spec_path = os.path.join(data_path, 'spec')
        self.raw_path = os.path.join(data_path, 'raw')
        if not os.path.exists(self.spec_path):
            os.makedirs(self.spec_path)
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)

    def get_melspectrogram(self, fn):
        loader = MonoLoader(filename=fn, sampleRate=self.fs)
        x = loader()
        return x, librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, sr=self.fs, n_fft=self.window, hop_length=self.hop, n_mels=self.mel))

    def iterate(self, audio_path, data_path):
        self.get_paths(audio_path, data_path)
        for fn in tqdm.tqdm(self.files):
            spec_fn = os.path.join(self.spec_path, fn.split('/')[-1][:-3]+'npy')
            raw_fn = os.path.join(self.raw_path, fn.split('/')[-1][:-3]+'npy')
            if not os.path.exists(raw_fn):
                try:
                    x, melspec = self.get_melspectrogram(fn)
                    np.save(open(raw_fn, 'wb'), x)
                    #np.save(open(spec_fn, 'wb'), melspec)
                except RuntimeError:
                    # some audio files are broken
                    print(fn)
                    continue

if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.iterate})
