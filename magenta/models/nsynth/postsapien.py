import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
import os

rootdir = '/home/paperspace/data/sounds'


def start():
    for subdir, dirs, files in os.walk(rootdir):
        for filex in files:
            file_path = subdir + os.sep + filex
            encode(file_path)


def encode(fname):
    print('encoding..')
    sr = 16000
    audio = utils.load_audio(fname, sample_length=40000, sr=sr)
    sample_length = audio.shape[0]
    print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

    model_path = '/home/paperspace/data/wavenet-ckpt/model.ckpt-200000'
    encoding = fastgen.encode(audio, model_path, sample_length)
    print(encoding.shape)
    print('finished encoding..')
    # np.save(fname + '.npy', encoding)
    decode(encoding, fname, sample_length, model_path)


def decode(encoding, fname, sample_length, model_path):
    print('decoding..')
    outdir = '/home/paperspace/data/sounds_gen/'
    fastgen.synthesize(encoding, save_paths=[outdir + 'gen_' + fname], checkpoint_path=model_path,
                       samples_per_save=sample_length)
    print('finished decoding..')


if __name__ == '__main__':
    start()
