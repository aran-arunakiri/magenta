import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

fname = '395058__mustardplug__breakbeat-hiphop-a4-4bar-96bpm.wav'
fname2 = '395058__mustardplug__breakbeat-hiphop-a4-4bar-96bpm.wav'
sr = 16000
audio = utils.load_audio(fname2, sample_length=40000, sr=sr)
sample_length = audio.shape[0]
print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

model_path = '/home/paperspace/data/wavenet-ckpt/model.ckpt-200000'
encoding = fastgen.encode(audio, model_path, sample_length)
print(encoding.shape)
np.save(fname2 + '.npy', encoding)
