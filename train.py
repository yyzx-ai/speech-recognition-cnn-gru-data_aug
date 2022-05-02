import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib
from tqdm import tqdm
from utils import get_data
from tensorflow.keras.callbacks import ModelCheckpoint
#from model_speech.cnn_ctc import Am
#from model_speech.cnn_lstm_ctc import Am
from model_speech.gru_ctc import Am
from test import to_test
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
#tensorflow_backend.set_session(tf.Session(config=config))


def to_train(learn_rate_am=0.0004,epoch_am=30,type_=0,batch_size_=6,etc_=0,da_sp=1):
    if etc_==1:
        path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/gen/'
    else:
        path_gen='/home/dell/文档/语音识别代码实例/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2-master/temp/'
    #np_data=os.listdir(path_gen)
    np_data_2=[]
    for r,d,i4 in os.walk(path_gen):
        for i5 in i4:
            np_data_2.append(i5)
# 0.准备训练所需数据------------------------------
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
    data_args['data_type'] = 'train'
    data_args['data_path ']= '../data/'
    data_args['master'] = False
    data_args['speech_cmd'] = True
    data_args['thchs']= False
    data_args['aishell'] = False
    data_args['prime'] = False
    data_args['stcmd'] = False
    data_args['batch_size'] =batch_size_
    data_args['data_length'] = None
    data_args['type_']=type_
    data_args['etc_']=etc_
    data_args['shuffle'] = True
    train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
    data_args = data_base
    data_args['data_type'] = 'dev'
    data_args['data_path'] = '../data/'
    data_args['master'] = False
    data_args['speech_cmd'] = True
    data_args['thchs'] = False
    data_args['aishell'] = False
    data_args['prime'] = False
    data_args['stcmd'] = False
    data_args['batch_size'] =batch_size_
    data_args['type_']=0
    data_args['data_length'] = None
    data_args['shuffle'] = True
    dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
    am_base = {"vocab_size":None,
    'lr':0.0008,
    'gpu_nums':1,
    'is_training':True}
    am_args=am_base
    am_args['vocab_size'] = len(train_data.am_vocab)
    am_args['gpu_nums'] = 1
    am_args['lr'] = learn_rate_am														
    am_args['is_training'] = True
    am = Am(am_args)

    if os.path.exists('logs_am/model.h5'):
        print('load acoustic model...')
        am.ctc_model.load_weights('logs_am/model.h5')

    epochs = epoch_am
    if da_sp>=batch_size_:
        batch_num = len(train_data.wav_lst) // train_data.batch_size+len(np_data)*(da_sp//batch_size_)
    else:
        batch_num = len(train_data.wav_lst) // train_data.batch_size+len(np_data_2)//batch_size_
    # checkpoint
    ckpt = "model_{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss',save_weights_only=False,verbose=1,save_best_only=True)



    # for k in range(epochs):
    #     print('this is the', k+1, 'th epochs trainning !!!')
    #     batch = train_data.get_am_batch()
    #     dev_batch = dev_data.get_am_batch()
    #     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

    batch = train_data.get_am_batch()

    dev_batch = dev_data.get_am_batch()
    '''
    boundaries=[10,20,30,40]
    learning_rates=[0.05,0.01,0.005,0.001]
    y=[]
    N=50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initalizer())
        for num_epoch in N:
            learning_rate=tf.train.piecewise_constant(num_cep,boundaries=boundaries,values=learning_rates)
            lr=sess.run([learning_rate])
            y.append[lr]
    '''

    am.ctc_model.fit(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint], workers=1, use_multiprocessing=False,  validation_data=dev_batch, validation_steps=16)
    print('训练完成')
    am.ctc_model.save_weights('logs_am/model.h5')



