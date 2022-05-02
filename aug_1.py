from scipy.fftpack import fft
import numpy as np
import scipy.io.wavfile as wav1
import glob
import matplotlib;matplotlib.use('TkAgg'),matplotlib.rc("font",family='Noto Sans CJK JP')
import matplotlib.pyplot as plt
import os
#import librosa
import wave
import random

from python_speech_features import mfcc
filename='BAC009S0002W0122.wav'
noisename='BAC009S0004W0433.wav'
noise_dir='/home/dell/文档/语音识别代码实例/语音识别代码/noise_file'

#---------------base_unit--------------
def read_audio_0(wav):
    simplerate,data=wav1.read(wav)
    return(simplerate,data)
def read_audio_1(wav):
    y,sr=librosa.load(wav)
    return(y,sr)
def read_audio_2(audio_file):
    """
    return 一维numpy数组，如（584,）,采样率"""
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data, framerate
def compute_mfcc(wav,cep=32):
    fs, audio = read_audio_0(wav)
    mfcc_feat = mfcc(audio, samplerate=fs,numcep=cep,nfilt=2*cep,winfunc=np.hamming)
    #mfcc_feat = mfcc_feat[::3]
    #mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = read_audio_0(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input
def get_feature(filename, framerate=16000, feature_dim=128):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :param feature_dim:
    :return: specgram [序列长度,特征维度]
    """
    fs,wave_data=read_audio_0(filename)
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=feature_dim)
    specgram = np.where(specgram == 0, np.finfo(float).eps, specgram)
    specgram = np.log10(specgram)
    return specgram
def tensor_to_img(spectrogram, x_range=None, y_range=None):
    plt.figure()  # arbitrary, looks good on my screen.
    #plt.imshow(spectrogram[0].T)
    plt.imshow(spectrogram)
    if x_range is not None:
        plt.xlim(0, x_range)
    if y_range is not None:
        plt.ylim(0, y_range)
    plt.show()
#---------------noise_aug--------------
def noise_aug(wav,cep=32,alpha=0):
    global noisedir
    noisename=np.random.choice(glob.glob(os.path.join(noise_dir,"*.wav")),1,replace=False)

    fs,aug_noise1=read_audio_0(noisename[0])
    fs1,wave1=read_audio_0(wav)

    if len(aug_noise1)<len(wave1):

        aug_noise=aug_noise1*alpha
        aug_ed=np.zeros(len(wave1))
        num=int(len(wave1)/len(aug_noise))
        for i in range(num):
            aug_ed[i*len(aug_noise):(i+1)*len(aug_noise)]=aug_noise[0:len(aug_noise)]
        wave=wave1+aug_ed
        mfcc_feat = mfcc(wave, samplerate=fs,numcep=cep,nfilt=2*cep,winfunc=np.hamming)
    else:

        aug_noise=aug_noise1*alpha
        #rand_pos=np.random.randint(0,len(aug_noise1)-len(wave1))
        #wave=wave1+aug_noise[rand_pos:rand_pos+len(wave1)]
        wave=wave1+aug_noise[0:len(wave1)]
        mfcc_feat=mfcc(wave, samplerate=fs,numcep=cep,nfilt=2*cep)
    return(mfcc_feat)
#----------------speed_aug--------------
def speed_aug(filename,cep=32):
    fs,samples=read_audio_0(filename)
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    speed = random.uniform(0.95,1.1)
    samples = samples.astype(np.float)
    samples = librosa.effects.time_stretch(samples, speed)
    samples = samples.astype(data_type)
    mfcc_feat = mfcc(samples, samplerate=fs,numcep=cep,nfilt=2*cep,winfunc=np.hamming)
    return(mfcc_feat)
#----------------roll_aug--------------
'''
def roll_aug(wav):
    fea=compute_fbank(wav)
    fea=np.roll(fea,np.random.randint(0,50))
    return(fea)
'''
#-------------time_shift_aug-----------
'''
def time_shift_aug(wav):
    input_data=compute_fbank(wav)
    shift_=np.random.randint(0,50)
    return np.roll(input_data,np.random.randint(0,50),axis=0)
'''
def time_shift_aug(wav,cep=32):


    fs, wavsignal = read_audio_0(wav)
    wavsignal=np.roll(wavsignal,np.random.randint(0-0.06*wavsignal.shape[0],0.06*wavsignal.shape[0]),0)
    mfcc_feat = mfcc(wavsignal, samplerate=fs,numcep=cep,nfilt=2*cep,winfunc=np.hamming)
    return(mfcc_feat)
#-------------pitch_shift_aug----------
'''
def pitch_shift_aug(wav):
    input_data=compute_fbank(wav)
    nb_cols=input_data.shape[0]
    max_shifts=nb_cols//20
    nb_shifts=np.random.randint(-max_shifts,max_shifts)
    return np.roll(input_data,nb_shifts,axis=1)
'''
#-------------time_mask_aug------------
def time_mask_aug(filename):
    data=compute_mfcc(filename)
    time_len =data.shape[0]
    for i in range(8):
        t = np.random.uniform(low=0.0, high=8)
        t = int(t)
        t0 =np.random.randint(0, time_len - t)
        data[ t0:t0 + t, :] = 0
    return data
#------------frequency_mask_aug--------
def freq_mask_aug(filename):
    fea=compute_mfcc(filename)
    freq_len =fea.shape[1]
    for i in range(2):
        f = np.random.uniform(low=1, high=3)
        f = int(f)
        f0 =np.random.randint(0, freq_len - f)
        fea[ :, f0:f0 + f] = 0
    return(fea)
#------------time_freq_mask_aug--------
def time_freq_mask_aug(filename):
    data=compute_mfcc(filename)
    time_len =data.shape[0]
    for i in range(6):
        t = np.random.uniform(low=0.0, high=6)
        t = int(t)
        t0 =np.random.randint(0, time_len - t)
        data[ t0:t0 + t, :] = 0
    freq_len =data.shape[1]
    for i in range(2):
        f = np.random.uniform(low=1, high=2)
        f = int(f)
        f0 =np.random.randint(0, freq_len - f)
        data[ :, f0:f0 + f] = 0
    return data
#------------amplitude_aug-------------
def amp_aug(wav,cep=32):
    """
    音量增益范围约为【0.316，3.16】，不均匀，指数分布，降低幂函数的底10.可以缩小范围
    :param samples: 音频数据，一维
    :param min_gain_dBFS:
    :param max_gain_dBFS:
    :return:
    """
    fs,samples=read_audio_0(wav)
    samples = samples * np.random.randint(90,110)*0.01
    mfcc_feat = mfcc(samples, samplerate=fs,numcep=cep,nfilt=2*cep,winfunc=np.hamming)
    return(mfcc_feat)


#type_=1:噪声增强
#type_=2;音速增强
#type_=3：旋转
#type_=4：时移增强
#type_=5：频移变换
#type_=6：时域遮掩
#type_=7：频域遮掩
#type_=8：音频合成




