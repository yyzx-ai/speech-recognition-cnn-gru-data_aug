import os
import matplotlib
import shutil

#x=1
from test import to_test
from train import to_train


x=input('请输入：1训练，2测试,3详细测试')
#print(type(x))
if int(x)==1:
    for i in range(10):
        print('\n-----------第{}次训练ing---------\n'.format(i+1))
        to_train(type_=0,epoch_am=10,learn_rate_am=abs(0.0001),batch_size_=64,etc_=1)
        #to_train(type_=0,epoch_am=50,learn_rate_am=abs(0.005-0.00054*i),batch_size_=8+2*i)#lr:0.005-0.00014
elif int(x)==2:
    to_test(test_num=0,detail=1,type_=6,alpha=0.08)
else:
    print('暂无此功能')
'''
#0:无增强0.1061，1：时移增强0.0973，2：音速增强0.1047，3：时域遮掩0.0929<，4：频域遮掩0.0693，5：音量增强0.1076---<<<s，6：噪声增强0.0929，7：时频遮掩

    #wer_i=to_test(47)
    #print('第'+str(i+1)+'次循环')
    #print('测试精度：'+str(wer_i))
    #shutil.copy('/home/c202/文档/DeepSpeechRecognition-master/logs_am/model.h5','/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/test')
    #os.rename('/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/test/model.h5','/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/test/'+str(wer_i)+'.h5')

    #if wer_i<min(wer):
        #shutil.copy('/home/c202/文档/DeepSpeechRecognition-master/logs_am/model.h5','/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/sota')
        #os.rename('/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/sota/model.h5','/home/c202/文档/DeepSpeechRecognition-master/管制语音模型/sota/'+str(wer_i)+'.h5')

    #wer.append(wer_i)
wer=[]
for i in range(10):
    print('\n-----------第{}次训练ing---------\n'.format(i+1))
    to_train(type_=0,epoch_am=int(5),learn_rate_am=abs(0.0005),batch_size_=64)
    wer_=to_test(test_num=100,detail=0)
    wer.append(wer_)
matplotlib.pyplot.plot(wer)
matplotlib.pyplot.show()
'''




'''
f1=open('record.txt','r')
f2=f1.readline()
f3=int(f2)
print(f3)
f4=open('record.txt','w')
f4.write(str(f3+1))
f4.close()
for i in range(10):
    to_train(type_=0,epoch_am=10,epoch_lm=5,learn_rate_am=0.01-0.0495*f3,learn_rate_lm=0.005-0.000245*f3,batch_size_=4+f3)
    shutil.copy('/home/c202/文档/DeepSpeechRecognition-master/logs_am/model.h5','/home/c202/文档/DeepSpeechRecognition-master/普通话模型/test')
    os.rename('/home/c202/文档/DeepSpeechRecognition-master/普通话模型/test/model.h5','/home/c202/文档/DeepSpeechRecognition-master/普通话模型/test/'+f2+'_'+str(i)+'.h5')
'''





