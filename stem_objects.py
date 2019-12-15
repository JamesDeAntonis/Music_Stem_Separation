from pathlib import Path  # for file handling
import os  # for file handling
import scipy.io.wavfile as wf  # for reading wav files NO WORK FOR 24BIT
import soundfile  # for converting 24 bit to 16 bit
from scipy import signal  # for the spectrogram
from scipy import misc
import matplotlib.pyplot as plt  # for plotting spectrograms
import numpy as np
import simpleaudio as sa  # good for playing the audio directly
from playsound import playsound  # also good for playing audio


class song_obj:  # SONG OBJECT (collection of stems)
  def __init__(self, stem_list, name=None, get_fourier=False):
    self.stem_list = stem_list
    if name != None:
      self.name = name
    else:
      self.name= stem_list[0].song_name
    self.audio = self.compile_song()  # returns an audio object
    if(get_fourier==True):
      self.fourier = fourier_obj(self)
    else:
      self.fourier = None
  
  def compile_song(self):
    sr_list = [stem.audio.sr for stem in self.stem_list]
    assert(all(x == sr_list[0] for x in sr_list))
    # dim1 = 15000000  # just to ensure all files have same shape
    data_list = [stem.audio.data for stem in self.stem_list]
    # assert(max([data.shape[0] for data in data_list]) < dim1)
    # dim2 = max([data.shape[1] for data in data_list])
    # audio = np.zeros((dim1))
    length = max([data.shape[0] for data in data_list])
    audio = np.zeros(length)
    for data in data_list:
      audio[:data.shape[0]] += data  # , :data.shape[1]] += data
    return audio_obj(data=audio, sr=sr_list[0])
  
  def get_stem_splits(self):
    vocals_stem_list = []
    perc_stem_list = []
    bass_stem_list = []
    other_stem_list = []
    for stem in self.stem_list:
      stem_group = stem_dict[stem.stem_name]
      if stem_group == 'Vocals':
        vocals_stem_list.append(stem)
      elif stem_group == 'Percussion':
        perc_stem_list.append(stem)
      elif stem_group == 'Bass':
        bass_stem_list.append(stem)
      elif stem_group == 'Other':
        other_stem_list.append(stem)
      else:
        print('warning: ', stem.stem_name, 'was not added to any group')
    return [song_obj(vocals_stem_list, name='Vocals'), 
            song_obj(perc_stem_list, name='Percussion'),
            song_obj(bass_stem_list, name='Bass'),
            song_obj(other_stem_list, name='Other')]


class stem_obj:  # STEM OBJECT
  def __init__(self, song_name, stem_name, get_fourier=False):
    self.filepath = os.path.join('Raw', song_name, stem_name)
    self.song_name = song_name
    self.stem_name = stem_name
    self.audio = audio_obj(filepath=self.filepath)
    if(get_fourier==True):
      self.fourier = fourier_obj(self.audio)
    else:
      self.fourier = None

class audio_obj:  # AUDIO OBJECT
  def __init__(self, data=None, sr=None, filepath=-1):
    # one of (audio, sr) or (filepath) is required
    # this is often the cause of problems if wrong if statement entered
    if filepath != -1:
      self.data, self.sr = soundfile.read(filepath)
      if len(self.data.shape) >= 2:  # stereo
        self.data = self.data[:, 0]  # need to think about stereo handling
        # self.data = np.mean(data, axis=1)  # i think this is correct
    else:
      assert isinstance(data, np.ndarray), "specify filepath or data"
      self.data = data
      self.sr = sr
      if isinstance(data, tuple):
        self.data = self.data[0]
    assert self.data.ndim == 1, "audio data must be 1d"

  def to_fourier(self):  # convert to fourier
    self.fourier = fourier_obj(self)
    return self.fourier
  
  def to_file(self, filepath):  # save as .wav
    soundfile.write(filepath, self.data, self.sr)
  
  def boxplot(self):  # boxplot of the spectrum of values in this audio
    plt.boxplot(self.data)

class fourier_obj:  # FOURIER OBJECT
  def __init__(self, audio=-1,  # input is either this line
               fourier=-1, freqs=-1, t=-1, sr=44100,  # or this line
               abs=False):
    # freqs : frequencies : freqs corresponding to 1st axis of fourier
    # t     : times       : times corresponding to 2nd axis of fourier
    # fourier : fourier series : 2d np array of fourier
    assertion = not isinstance(audio, int) or not isinstance(fourier, int)
    assert assertion, "must either specify audio_obj or fourier array"
    if isinstance(fourier, int):
      self.m = 1024
      [self.freqs, self.t, self.fourier] = signal.stft(audio.data,
                                                       nperseg=self.m,
                                                       noverlap=3*self.m//4)
      self.sr = sr
      self.fourier_ri = self.get_two_channel_fourier()
    else:  # if just passing in a fourier to make the object
      if fourier.ndim == 2:
        self.fourier = fourier
        self.m = self.fourier.shape[0] * 2 - 2
        self.freqs = freqs
        self.t = t
        self.sr = sr
        self.fourier_ri = self.get_two_channel_fourier()
      else:  # the fourier defined is actually the fourier_ri
        assert fourier.ndim == 3, "fourier_obj must have ndim 2 or 3"
        self.fourier_ri = fourier
        self.fourier = self.ri_to_complex()
        self.m = self.fourier.shape[0] * 2 - 2
        self.freqs = freqs
        self.t = t
        self.sr = sr
    if abs == True:
      self.fourier = np.abs(self.fourier)
    assert self.fourier.ndim == 2, "fourier must be 2d"
  
  def get_two_channel_fourier(self):
    # for real in first channel, imaginary in second
    f = np.zeros(self.fourier.shape + tuple((2,)))
    f[:, :, 0] = np.real(self.fourier)
    f[:, :, 1] = np.imag(self.fourier)
    return f
  
  def ri_to_complex(self):
    # convert the 2-channel fourier to a standard 2d fourier
    real = np.array(self.fourier_ri[:, :, 0], dtype=complex)
    imag = np.array(self.fourier_ri[:, :, 1], dtype=complex) * 1j
    return real + imag
  
  def to_audio(self):  # convert to audio
    ts, data = signal.istft(self.fourier, self.sr,
                            nperseg=self.m, noverlap=3*self.m//4)
    self.audio = audio_obj(data=data, sr=self.sr)
    return self.audio
  
  def to_file(self, songname, filename):  # save as jpg
    path = os.path.join('Train', songname, 'TRANSFORM', filename + '.jpg')
    scipy.misc.imsave(path, self.fourier)
  
  def plot(self):
    # log makes the plot more readable
    fourier_like = np.log(np.abs(self.fourier).clip(min=0.0000000001))
    plt.pcolormesh(self.t, self.freqs, fourier_like)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()