import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen


fname = '395058__mustardplug__breakbeat-hiphop-a4-4bar-96bpm.wav'
sr = 16000
audio = utils.load_audio(fname, sample_length=40000, sr=sr)
sample_length = audio.shape[0]
print('{} samples, {} seconds'.format(sample_length, sample_length / float(sr)))

encoding = fastgen.encode(audio, 'model.ckpt-200000', sample_length)
print(encoding.shape)
np.save(fname + '.npy', encoding)
