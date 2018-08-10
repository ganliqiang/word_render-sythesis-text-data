# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import os
import numpy as np
import pygame
import PIL
from create_corpus import create_amount_in_word
from hwcaptcha import makeDirDict,getHandwritingCaptcha
from numpy import random
from PIL import Image
from generate_chars import CharsGenerator

import json
from yinhangcard import makePic

from generate_word_training_data import byteify
from create_corpus import create_date
from testclass import Generate
import sys
import time
import math
import numpy as np


print sys.getdefaultencoding()


aa=[1,2,3,4]
cc=aa
for i in range(5):
    aa[i]=10+i
'''
        rect=bg_surf.get_rect()
        dhsk=rect
        rect.x = 100
        for i in range(5):
            rect= bg_surf.get_rect()
            rect.x = 99
            pass
'''

bb=aa[:2]
for date in create_date():
    list = []
    temp = date.split("年")
    list.append(temp[0])
    temp = temp[1].split(u"月")
    list.extend(temp)
    vhjksd=0





    fdhsk=date
    print(date)
list=tuple(range(10))
hs=map(str,list)






def demo(*para):

    para_list=np.array(para)
    mean=para_list.mean()
    return mean,(x for x in para if x>mean)

gsd=demo(1,3,4,5,6)


















count_down = 10 # 设置倒计时时间，单位：秒
interval = 1 # 设置屏幕刷新的间隔时间，单位：秒
for i in range(0, int(count_down/interval)+1):
    print("\r"+"▇"*i+" "+str(i*10)+"%"),
    time.sleep(interval)
print("\n加载完毕")






for i in range(100):
    #sys.stdout.write('\r')
    #sys.stdout.write('已扫描%d个文件' % i)
    #print('\r'),
    print('\r已扫描%d个文件' % i),
    time.sleep(1)





for progress in range(100):
    time.sleep(0.5)
    sys.stdout.write("Download progress: %d%%   \r" % (progress))
    sys.stdout.flush()





for i in range(5):
    #print("wer,%d\r" %i),
    print 'Downloading:%d\r' % i,
    #sys.stdout.flush()
    time.sleep(2)











path="/home/user/sample_generator/cortext/data/"
result1=[]
i=0
filePath="/home/user/sample_generator/cortext/data/code_list.txt"
with open(filePath) as ff:
    for line in ff:
        line.strip()
        i=i+1
        print(line,i)
        result1.append(line.decode("utf-8"))
filePath="/home/user/sample_generator/cortext/data/label_applicant.txt"
result2=[]

with open(filePath) as ff:
    for line in ff:
        line.strip()
        line=line.decode("GB18030")
        if line not in result1:
          result2.append(line)
hfjdks=0
with open(os.path.join(path,"result.txt"),'w') as ff:
    for line in result2:
        ff.write(line.encode("utf-8"))

hfjdks=0
























'''
def demo(new, old_list=[]):
    old_list.append(new)
    return old_list
print(demo('5',[1,2,3,4]))
print(demo('aaa',['a','b']))
print(demo('a'))
print(demo('b'))
print(demo('5',[1,2,3,4]))
print(demo('b1'))
print(demo('aaa',['a','b']))
print(demo('b2'))
gd=0
'''









def count(n):
    x = 0
    while x < n:
        value = yield x*x
        print 'Received : %s' % value
        if value is not None:
            print 'Received value: %s' %value
        x += 1
gen = count(5)
print gen.next()
#print gen.send('Hello')
#print gen.send('Helloll')
print gen.next()

print gen.next()






'''
gener=Generate(len=10,size=3.5)
fun="add"
funct=getattr(gener,fun)
gdcash=funct()
gks=0
'''

ad={"fa":0,"g":1}
del ad["g"]
fs=0

list1=[1,2,3,4]
list2=['a','b','c','d','f']
dhfjs=zip(list1,list2)
list3=[('a',1),(2,'b')]
fhdjsk=dict(list3)
grew=0







list_fn=[i for i in [1,2,3,4,5,6,7,8,9] if i>5]

a = (x*x for x in range(10))
#das= random.choice(a)

'''
path="/home/user/sample_generator/Synthetic_Data_Engine_For_Text_Recognition/ICDAR_2003/Word_Recognition_Train/word/word/Train/"
pathlist=[]
for file in os.listdir(path):
    pathlist.append(os.path.join(path,file))
ghks=0
'''






'''
path=u"/home/user/0/标点1.txt"
path1=u"/home/user/0/标点2.txt"
with open(path) as ff:
    with open(path1,"w") as ffw:
        for line in ff:
           try:
             str_list=line.split()
             line=str_list[1]
             #line = line.encode("utf-8")
           except Exception:
               print(line)
               continue
           #line = line.encode("utf-8")

           ffw.write(line+"\t\n")
'''



def multiple2(**args):
    fs=len(args)
    print(len(args))
  #打印不定长参数
    for key in args:
      print key + ":" + bytes(args[key])
#multiple2("a",1)
multiple2(name='Amy', age=12, single=True)








listpath=["ghsdkj","gj","hdf"]
path="/"
for i in listpath:
    path=os.path.join(path,i)
    gfj=0






root=u"/home/user/sample_generator/"
#root=os.path.split(os.path.realpath(__file__))[0]
configPath=os.path.join(root,"配置文件")
random_dir = os.path.join(root, "Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/img")
trainingchars_fn = os.path.join(root, "Synthetic_Data_Engine_For_Text_Recognition/ICDAR_2003/Word_Recognition_Train/word/word/Train/")
for dirpath, dirnames, filenames in os.walk(configPath):
    for file in filenames:
        if file.endswith(".conf"):
            path=os.path.join(dirpath,file)
            with open(path) as ff:
                jsonconfig = json.load(ff)
                config = byteify(jsonconfig)
            FontDir=os.path.join(dirpath,"font")

            labelBackgdir = os.path.join(dirpath, "background")
            output_dir=os.path.join(root, "result",config["label"].decode("utf-8" )+"_output_dir/")

            #config["randomBackgdir"]=random_dir
            config["trainingchars_fn"]=trainingchars_fn
            strr=config["char_config"]["charDir"]
            hfjds=type(strr)
            gfds=type("hnd")
            if isinstance(strr,str):
                strr=strr.split("/")
                find=False
                charDir=root
                for nu in strr:
                    if not find and nu=="cortext":
                        find=True
                    if find:
                        charDir = os.path.join(charDir, nu)
                if find:
                    config["char_config"]["charDir"]=charDir

            config["char_config"]["FontDir"]=FontDir.decode("utf-8" )
            config["labelBackgdir"] = labelBackgdir.decode("utf-8")
            try:
                del config["rotateAngle"]
            except Exception:
                pass
            '''
            config["output_config"]["keep_label"]=1
            config["char_config"]["lineChars"] = config["lineChars"]
            config["char_config"]["spaceVer"] = config["spaceVer"]
            del config["lineChars"]
            del config["spaceVer"]
            
            temp={}
            temp["output_dir"]=output_dir.decode("utf-8")
            temp["back_adjust"] = "None"
            temp["output_height"] = [64 , 72]
            temp["output_width"] = [240 , 270]
            temp["ver_random"] = 1
            temp["hor_random"] = 0
            temp["keep_label"] = 1
            temp["red"] = 1
            config["output_config"] =temp
            temp = {}
            temp["isNoise"] = 0
            temp["useOriBck"] = 1
            temp["randomBackgdir"] = config["randomBackgdir"]
            temp["rotateAngle"] = 10
            config["noise_config"] = temp

            temp = {}
            temp["FontDir"] = config["FontDir"]
            temp["FontSize"] = config["FontSize"]
            temp["charDir"] = config["charDir"]
            temp["char_spacing"] = config["char_spacing"]
            temp["leftAdjust"] = config["leftAdjust"]
            config["char_config"] = temp
            
            
            del config["randomBackgdir"]
            del config["FontDir"]
            del config["FontSize"]
            del config["charDir"]
            del config["char_spacing"]
            del config["leftAdjust"]
            del config["red"]
            del config["keep_label"]
            del config["output_dir"]
            
            
            adjust_value = {}
            adjust_value[u"《"] = 2.0
            adjust_value[u"|"] = 1.5
            adjust_value[u"“"] = 2.5
            adjust_value[u"〈"] = 2.0
            adjust_value[u"￥"] = 1
            adjust_value[u"‘"] = 2.5
            adjust_value[u"·"] = 2.0
            adjust_value[u"（"] = 1.7
            adjust_value[u"１"] = 1.7
            adjust_value[u"I"] = 1
            adjust_value[u"1"] = 0.5
            config["leftAdjust"] = adjust_value
            '''
            with open(path,'w') as ff:
                ff.write(json.dumps(config, indent=1,ensure_ascii=False))






print "end"




#for i in create_amount_in_word():
 #   print(i)
  #  fsdhj=0
charsGenerator=CharsGenerator()
path="/home/user/Downloads/digital_src/digital_src/"
for chars in charsGenerator.yinhangcard_num():
    img=makePic(chars,path)
    img.show()
    print(chars)

aa=[1,2,5,6]

aa.reverse()










gen_pic_dir="/home/user/gen_pic_src_chars/000056/"

for file in os.listdir(gen_pic_dir):
    path=os.path.join(gen_pic_dir,file)
    im = Image.open(path)
    im_np = np.array(im)
    bdf=0




gen_pic_dir="/home/user/gen_pic_src_chars/"

dir_dict = makeDirDict(dir=gen_pic_dir)
chars="456161354"
width1 = len(chars) * 15
image = getHandwritingCaptcha(chars=chars, dir_dict=dir_dict,height=  30,width= width1,
                                            space_inf=-10, space_sup=0,
                                            offset=(random.randint(-5, 10), random.randint(-15, 15)))


left = 0
up =1
right = 2
down = 3
box0 = (left, up, right, down)
down=30
print(box0)

path="/home/user/Downloads/zxd.jpg"
path1="/home/user/Downloads/zx.jpg"




vgfds=18/4.0

fkgh='#2f404a'

display_text=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "C", "K", "N", "P", "R", "U", "V", "X", "Y"]
display_text_list=[]
temp=[]

for i in range(10):
    print(i)
print(i)
for c in display_text:
    temp.append(c)
    if len(temp) > 5:
        display_text_list.append(temp)
        temp = []












aa=np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]],dtype=np.uint8)
cc=aa[...,0:2]
bb=np.array([[1,2],[4,5]])
'''
cc=aa[...,0]*bb

aa.astype(np.float)
cc=np.concatenate((aa,aa),axis=2)

aa[:,3]=1
bb=np.array([[7,8,9],[10,11,12]])
#aa=[[1,2,3],[4,5,6]]
#aa[:1]=aa[:1]/0.5
aa=list(aa)
bb=list(bb)
cc=aa*bb













num=int("7857",16)
das=unichr(65)


hksd=np.array([[[1,2],[3,4]],[[10,20],[30,40]]])
#hksd=hksd[1,:]
gfes=np.index_exp[:, :, 1]
dfaw=hksd[gfes]
vds=0


pathC="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt"
pathImage="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/yinhangcode"
ff=open(pathC,"w+")
for file in  os.listdir(pathImage):
    filePath=os.path.join(pathImage,file)
    if not os.path.isdir(filePath):
        ff.write(filePath+"\n")
ff.close()
ff=open(pathC,"a+")
lines=ff.readlines()
ff.close()
'''

path="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt2.txt"
pathC="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt"

path1="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt1.txt"
pathC1="/home/user/Downloads/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt"
with open(path1) as f:
    lines=f.readlines()
    ff=open(pathC1,"w+")
    for line in lines:
        line=line.replace("ubuntu","user/Downloads")
        ff.write(line)
        print line
    ff.close()




import pygame

