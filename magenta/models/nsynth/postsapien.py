import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

file1 = '395058__mustardplug__breakbeat-hiphop-a4-4bar-96bpm.wav'
file2 = 'kick1.wav'
fname = file2

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
print('decoding..')

fastgen.synthesize(encoding, save_paths=['gen_' + fname], samples_per_save=sample_length)
print('finished decoding..')
