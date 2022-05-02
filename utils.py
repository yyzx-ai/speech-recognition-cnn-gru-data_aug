import os
import difflib
import numpy as np
import tensorflow.compat.v1 as tf

import scipy.io.wavfile as wavread
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from tensorflow.keras import backend as K
from aug_1 import speed_aug,time_shift_aug,time_mask_aug,freq_mask_aug,amp_aug,time_freq_mask_aug,noise_aug
#path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/temp/'

data_base = {'data_type':'train',
    'data_path':'data/',
    'master':False,
    'speech_cmd':True,
    'thchs':False,
    'aishell':False,
    'prime':False,
    'stcmd':False,
    'batch_size':1,
    'data_length':None,
    'type_':0,
    'etc_':0,
    'alpha':0,
    'shuffle':True}

class get_data():
    def __init__(self, args):
        self.data_type = args['data_type']
        self.data_path = args['data_path']
        self.master = args['master']
        self.speech_cmd = args['speech_cmd']
        self.thchs = args['thchs']
        self.aishell = args['aishell']
        self.prime = args['prime']
        self.stcmd = args['stcmd']
        self.data_length = args['data_length']
        self.batch_size = args['batch_size']
        self.shuffle = args['shuffle']
        self.type_=args['type_']
        self.etc_=args['etc_']
        self.alpha=args['alpha']
        self.da_sp=1
        self.np_data=[]
        self.np_data_2=[]
        self.path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/temp/'
        self.source_init()

    def source_init(self):
        if self.etc_==1:
            self.path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/gen/'
        else:
            self.path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/temp/'
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.master == True:
                read_files.append('master_train.txt')
            if self.speech_cmd == True:
                read_files.append('speech_cmd_train.txt')
            if self.thchs == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd.txt')
        elif self.data_type == 'dev':
            if self.master == True:
                read_files.append('master_dev.txt')
            if self.speech_cmd == True:
                read_files.append('speech_cmd_dev.txt')
            if self.thchs == True:
                read_files.append('thchs_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
        elif self.data_type == 'test':
            if self.master == True:
                read_files.append('master_test.txt')
            if self.speech_cmd == True:
                read_files.append('speech_cmd_test.txt')
            #if self.thchs == True:
                #read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
        self.wav_lst = []
        self.pny_lst = []
        #self.han_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = 'data/' + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                #self.han_lst.append(han.strip('\n'))
        #if self.data_length:
            #self.wav_lst = self.wav_lst[:self.data_length]
            #self.pny_lst = self.pny_lst[:self.data_length]
            #self.han_lst = self.han_lst[:self.data_length]
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)
        #print(self.pny_lst)
        #print(self.am_vocab)

        print('make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
        #print(self.pny_vocab)
        #print('make lm hanzi vocab...')
        #self.han_vocab = self.mk_lm_han_vocab(self.han_lst)

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        self.np_data=os.listdir(self.path_gen)
        for r,d,i4 in os.walk(self.path_gen):
            for i5 in i4:
                self.np_data_2.append(i5)
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
                shuffle(self.np_data_2)
            if self.data_type=='test':
                num_=len(self.wav_lst) // self.batch_size
            else:
                if self.da_sp>=self.batch_size:
                    num_=len(self.wav_lst) // self.batch_size+len(self.np_data)*(self.da_sp//self.batch_size)
                else:
                    num_=len(self.wav_lst) // self.batch_size+len(self.np_data_2)//self.batch_size
            for i in range(num_):
                wav_data_lst = []
                label_data_lst = []
                if i<len(self.wav_lst) // self.batch_size:
                    begin = i * self.batch_size
                    end = begin + self.batch_size
                    sub_list = shuffle_list[begin:end]
                    #print('step_1\n')
                    #print(sub_list)
                    for index in sub_list:
    #-------------------------------------------------------------------------------aug------------------
    #0:无增强，1：时移增强，2：音速增强，3：时域遮掩，4：频域遮掩，5：音量增强，6：噪声增强,7：时频遮掩
                        if self.type_==0:
                            fbank = compute_mfcc(self.wav_lst[index])
                            #fbank = compute_fbank(self.wav_lst[index])
                        elif self.type_==1:
                            fbank=time_shift_aug(self.wav_lst[index])
                        elif self.type_==2:
                            fbank=speed_aug(self.wav_lst[index])
                        elif self.type_==3:
                            fbank=time_mask_aug(self.wav_lst[index])
                        elif self.type_==4:
                            fbank=freq_mask_aug(self.wav_lst[index])
                        elif self.type_==5:
                            fbank=amp_aug(self.wav_lst[index])
                        elif self.type_==6:
                            fbank=noise_aug(self.wav_lst[index],alpha=self.alpha)
                        elif self.type_==7:
                            fbank=time_freq_mask_aug(self.wav_lst[index])
                        pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                        pad_fbank[:fbank.shape[0], :] = fbank
                        #print(pad_fbank.shape)
                        #print('step_2')
                        #print('')
                        label = self.pny2id(self.pny_lst[index], self.am_vocab)
                        label_ctc_len = self.ctc_len(label)
                        if pad_fbank.shape[0] // 8 >= label_ctc_len:
                            wav_data_lst.append(pad_fbank)
                            label_data_lst.append(label)#
                    pad_wav_data, input_length = self.wav_padding(wav_data_lst)#
                    pad_label_data, label_length = self.label_padding(label_data_lst)#
                    #print('\n{}\n{}\n{}\n{}'.format(pad_label_data.shape, label_length,pad_wav_data.shape,input_length))
                    #print('------------------------------------------------------------\n\n')
                    inputs = {'the_inputs': pad_wav_data,
                              'the_labels': pad_label_data,
                              'input_length': input_length,
                              'label_length': label_length,
                              }
                    outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                    yield inputs, outputs
                else:
                    if self.da_sp>=self.batch_size:

                        #print('\n\nshabi\n\n')
                        for gen_1 in self.np_data:
                            gen_2=np.load(self.path_gen+gen_1)
                            for i2 in range(self.da_sp//self.batch_size):
                                #print('\n\nshabi\n\n')

                                pad_wav_data=gen_2[i2*self.batch_size:i2*self.batch_size+self.batch_size]
                                #print('\n\nshabi\n\n')

                                for i in range(self.batch_size):
                                    label_data_lst.append(self.pny_vocab.index(str(gen_1.split('.')[0])))
                                #pad_label_data, label_length = self.label_padding(label_data_lst)
                                #print('\n\nshabi\n\n')

                                label_length=[]
                                for i in range(self.batch_size):
                                    label_length.append(1)
                                label_length=np.array(label_length)
                                #print(label_length)
                                input_length=[]
                                for i in range(self.batch_size):
                                    input_length.append(104//8)
                                input_length=np.array(input_length)
                                #print('\n{}\n{}\n{}\n{}'.format(pad_label_data.shape, label_length,pad_wav_data.shape,input_length))
                                #print('------------------------------------------------------------\n\n')

                                inputs = {'the_inputs': pad_wav_data,
                                          'the_labels': pad_label_data,
                                          'input_length': input_length,
                                          'label_length': label_length,
                                          }
                                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                                yield inputs, outputs
                    else:
                        i=i-len(self.wav_lst) // self.batch_size
                        begin = i * self.batch_size
                        end = begin + self.batch_size
                        sub_list = self.np_data_2[begin:end]
                        pad_wav_data=np.zeros(((self.batch_size),104,32,1))
                        for i5 in range(self.batch_size):
                            np_data_3=np.load(self.path_gen+sub_list[i5].split('_')[0]+r'/'+sub_list[i5])
                            pad_wav_data[i5,:,:,:]=np_data_3
                        for i6 in range(self.batch_size):
                            label_data_lst.append(self.pny_vocab.index(str(sub_list[i6].split('_')[0])))
                        label_data_lst=np.array(label_data_lst)
                        label_length=[]
                        for i7 in range(self.batch_size):
                            label_length.append(1)
                        label_length=np.array(label_length)
                        input_length=[]
                        for i8 in range(self.batch_size):
                            input_length.append(104//8)
                        input_length=np.array(input_length)
                        #print('\n{}\n{}\n{}\n{}'.format(pad_label_data.shape, label_length,pad_wav_data.shape,input_length))
                        inputs = {'the_inputs': pad_wav_data,
                              'the_labels': pad_label_data,
                              'input_length': input_length,
                              'label_length': label_length,
                              }
                        outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                        yield inputs, outputs

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    #def han2id(self, line, vocab):
        #return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 32, 1))
#        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 64, 1))
#        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len


# 对音频文件提取mfcc特征
def compute_mfcc(file):
    if os.path.exists(file):
        #global mfcc_feat
        fs, audio = wavread.read(file)
#-------------------------------------------------------------------------------------------test--------------<<<<
#    mfcc_feat = mfcc(audio, samplerate=fs, numcep=64,nfilt=128)
        mfcc_feat = mfcc(audio, samplerate=fs, numcep=32,nfilt=64)
#    mfcc_feat = mfcc_feat[::3]
#    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat

'''
# 获取信号的时频图-----------------------------------------------------------------------------test-----------<<<<
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    #fs,wavsignal=speed_aug(file)
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

'''
# word error rate------------------------------------
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = False, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text
