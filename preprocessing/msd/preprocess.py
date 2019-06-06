import os
import numpy as np
import glob
from essentia.standard import MonoLoader
import librosa
import fire
import tqdm
import _pickle as pkl


class Processor:
    def __init__(self):
        self.fs = 16000
        self.window = 512
        self.hop = 256
        self.mel = 96

    def get_paths(self, audio_path, data_path, _iter=0, num_core=20):
        _iter -= 1
        self.files = self.get_filelist(_iter, num_core)
        self.spec_path = os.path.join(data_path, 'spec')
        self.raw_path = os.path.join(data_path, 'raw')
        if not os.path.exists(self.spec_path):
            os.makedirs(self.spec_path)
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)

    def get_filelist(self, _iter, num_core):
        train = pkl.load(open('filtered_list_train.cP', 'rb'))
        test = pkl.load(open('filtered_list_test.cP', 'rb'))
        fl = train + test
        hop = len(fl) // num_core
        if _iter == num_core-1:
            fl = fl[hop*_iter:]
        else:
            fl = fl[hop*_iter:hop*(_iter+1)]
        return fl

    def get_melspectrogram(self, fn):
        loader = MonoLoader(filename=fn, sampleRate=self.fs)
        x = loader()
        return x, librosa.core.amplitude_to_db(librosa.feature.melspectrogram(x, sr=self.fs, n_fft=self.window, hop_length=self.hop, n_mels=self.mel))

    def iterate(self, audio_path, data_path, _iter, num_core):
        self.get_paths(audio_path, data_path, _iter, num_core)
        for fn in tqdm.tqdm(self.files):
            audio_fn = os.path.join(audio_path, fn[2], fn[3], fn[4], fn+'.mp3')
            spec_fn = os.path.join(self.spec_path, fn[2], fn[3], fn[4], fn+'.npy')
            raw_fn = os.path.join(self.raw_path, fn[2], fn[3], fn[4], fn+'.npy')
            if not os.path.exists(raw_fn):
                try:
                    x, melspec = self.get_melspectrogram(audio_fn)
                    raw_path = os.path.dirname(raw_fn)
                    if not os.path.exists(raw_path):
                        os.makedirs(raw_path)
                    spec_path = os.path.dirname(spec_fn)
                    if not os.path.exists(spec_path):
                        os.makedirs(spec_path)
                    np.save(open(raw_fn, 'wb'), x)
                    np.save(open(spec_fn, 'wb'), melspec)
                except RuntimeError:
                    # some audio files are broken
                    print(audio_fn)
                    continue

if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.iterate})
