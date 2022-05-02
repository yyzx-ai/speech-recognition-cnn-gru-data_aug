#coding=utf-8
import os
import difflib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import tqdm
import math
from utils import decode_ctc, GetEditDistance

def to_test(test_num=47,detail=0,etc_=0,type_=0,alpha=0):

    # 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
    from utils import get_data
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
    data_args=data_base
    train_data = get_data(data_args)


    # 1.声学模型-----------------------------------
    #from model_speech.cnn_ctc import Am
    #from model_speech.cnn_lstm_ctc import Am
    from model_speech.gru_ctc import Am
    am_base = {"vocab_size":None,
    'lr':0.0008,
    'gpu_nums':1,
    'is_training':True}
    am_args=am_base
    am_args['vocab_size'] = len(train_data.am_vocab)
    am = Am(am_args)
    print('loading acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

    # 2.语言模型-------------------------------------------



    # 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，

    data_args['data_type'] = 'test'
    data_args['shuffle'] = False
    data_args['batch_size'] = 1
    data_args['type_']=type_
    data_args['alpha']=alpha
    test_data = get_data(data_args)

    # 4. 进行测试-------------------------------------------
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
    am_batch = test_data.get_am_batch()
    word_num = 0
    word_error_num = 0
    print('正在测试--------------------------<<<<')

    if test_num==0:
        #print(len(test_data.wav_lst))
        t_num=len(test_data.wav_lst)
    else:
        t_num=test_num
    fin_=0
    for i in tqdm.tqdm(range(t_num)):
        # 载入训练好的模型，并进行识别
        inputs, _ = next(am_batch)
        #print('step_1')
        x = inputs['the_inputs']

        y = test_data.pny_lst[i]
        #print('step_10')
        result = am.model.predict(x, batch_size=1)#----------------------------------------------------------------------------------------------------------<<<<<
        # 将数字结果转化为文本结果
        _, text = decode_ctc(result, train_data.am_vocab)
        text = ' '.join(text)
        #text.replace('_','').replace(' ','')
        text_1=' '.join(y)
        #text_1.replace('_','').replace(' ','')
        if detail==1:
            print('\n 第 ', i, '个 example.')
            print('预测结果：', text)
            print('原文结果：', text_1)
            if data_args['speech_cmd']:
                #print('True')
                if text.replace('_','').replace(' ','')!=text_1.replace('_','').replace(' ',''):
                    fin_+=1
                fin=fin_/t_num
            else:
                word_error_num += min(len(text), len(set(text.split())&set(text_1.split())))
                #word_error_num += min(len(text),GetEditDistance(text,text_1))
                word_num += len(text.split())
                fin=word_error_num / word_num
        else:
            #print('\n 第 ', i, '个 example.')
            #print('预测结果：', text)
            #print('原文结果：', text_1)
            if data_args['speech_cmd']:
                #print('True')
                if text.replace('_','').replace(' ','')!=text_1.replace('_','').replace(' ',''):
                    fin_+=1
                fin=fin_/t_num
            else:
                word_error_num += min(len(text), len(set(text.split())&set(text_1.split())))
                #word_error_num += min(len(text),GetEditDistance(text,text_1))
                word_num += len(text.split())
                fin=word_error_num / word_num
    print(r'错误率：{}%'.format(fin*100))
    return(fin)































