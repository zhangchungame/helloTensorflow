#coding:UTF-8
import os
import tensorflow as tf;
import pickle
import numpy as np

def tranfData():
    f = file('/code/python/data/zhuanli_verification_code.txt')
    str=f.read()
    trainData=[];
    testData=[]
    tmps=str.split("\r\n")
    del tmps[0]
    len=0;
    for tmp in tmps:
        tmp=tmp.split(",")
        if len<12000:
            trainData.append({"fileName":tmp[1],"code":tmp[2]})
        elif len<13000:
            testData.append({"fileName":tmp[1],"code":tmp[2]})
        else:
            pass
        len=len+1

    sess=tf.Session()
    for data in testData:
        dealFileName='/code/python/data/imgdata/'+data['fileName']+'.txt'
        if os.path.exists(dealFileName):
            continue
        image_raw_data_jpg = tf.gfile.FastGFile("/code/python/data/yanzhengma/"+data['fileName'], 'r').read()
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码
        data['img']=convert2gray(sess.run(img_data_jpg))
        f = open(dealFileName, 'wb')
        pickle.dump(data['img'], f)
        f.close()

    for data in trainData:
        dealFileName='/code/python/data/imgdata/'+data['fileName']+'.txt'
        if os.path.exists(dealFileName):
            continue
        image_raw_data_jpg = tf.gfile.FastGFile("/code/python/data/yanzhengma/"+data['fileName'], 'r').read()
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码
        try:
            data['img']=convert2gray(sess.run(img_data_jpg))
        except:
            print data
        f = open(dealFileName, 'wb')
        pickle.dump(data['img'], f)
        f.close()
    print 1
    sess.close()

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


trainData=[]
testData=[]
global currentFlag
currentFlag=0

def readData():
    f = file('/code/python/data/zhuanli_verification_code.txt')
    str=f.read()
    f.close()
    tmps=str.split("\r\n")
    del tmps[0]
    len=0;
    for tmp in tmps:
        tmp=tmp.split(",")
        if len<12000:
            trainData.append({"fileName":tmp[1],"code":tmp[2]})
        elif len<13000:
            testData.append({"fileName":tmp[1],"code":tmp[2]})
        else:
            pass
        len=len+1


def getNext():
    trainDataLen=len(trainData)
    if trainDataLen<1:
        readData()

    global currentFlag
    if currentFlag+1>trainDataLen:
        currentFlag=0
    if 'img' not in trainData[currentFlag].keys():
        dealFileName='/code/python/data/imgdata/'+trainData[currentFlag]['fileName']+'.txt'
        pkl_file = open(dealFileName, 'rb')
        trainData[currentFlag]['img'] = pickle.load(pkl_file)
        pkl_file.close()
    result= trainData[currentFlag]
    currentFlag=currentFlag+1
    return result

def getNextTest():
    testDataLen=len(testData)
    if testDataLen<1:
        readData()

    global currentFlag
    if currentFlag+1>testDataLen:
        currentFlag=0
    if 'img' not in testData[currentFlag].keys():
        dealFileName='/code/python/data/imgdata/'+testData[currentFlag]['fileName']+'.txt'
        pkl_file = open(dealFileName, 'rb')
        testData[currentFlag]['img'] = pickle.load(pkl_file)
        pkl_file.close()
    result= testData[currentFlag]
    currentFlag=currentFlag+1
    return result


def findByfileName(fileName):
    dealFileName='/code/python/data/imgdata/'+fileName+'.txt'
    pkl_file = open(dealFileName, 'rb')
    img = pickle.load(pkl_file)
    pkl_file.close()
    return img
if __name__ == "__main__":
    a=getNext()
    print 1