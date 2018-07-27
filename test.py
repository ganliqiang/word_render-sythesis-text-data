import os
import numpy as np
import pygame
import PIL
from create_corpus import create_amount_in_word
from hwcaptcha import makeDirDict,getHandwritingCaptcha
from numpy import random
from PIL import Image
from generate_chars import CharsGenerator
from yinhangcard import makePic

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

