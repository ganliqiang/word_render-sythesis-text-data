# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import sys
sys.path.append('..')
import numpy as np
import os
import random
from PIL import Image



def copyBck( image_background,image_chars):
    w, h = image_background.size
    image_chars = image_chars.resize((w, h), Image.ANTIALIAS)

    image_chars = image_chars.convert('L')
    print image_chars.size


    image_background_np = np.array(image_background)
    image_chars_np = np.array(image_chars)

    for i in range(h):
        for j in range(w):
            image_background_np[i, j, 0] = int(
                float(image_chars_np[i, j]) * float(image_background_np[i, j, 0]) / float(255))
            image_background_np[i, j, 1] = int(
                float(image_chars_np[i, j]) * float(image_background_np[i, j, 1]) / float(255))
            image_background_np[i, j, 2] = int(
                float(image_chars_np[i, j]) * float(image_background_np[i, j, 2]) / float(255))
    image_background = Image.fromarray(image_background_np)
    #image_background.show()
    return  image_background

def copyBck_1( image_background,image_chars):
    w, h = image_background.size
    image_chars = image_chars.resize((w, h), Image.ANTIALIAS)
    
       
    # image_chars = image_chars.convert('L')
    print image_chars.size


    image_background_np = np.array(image_background)
    image_chars_np = np.array(image_chars)
    h, w, deep = image_chars_np.shape
    alpha=image_chars_np[...,deep-1]
    alpha=alpha/255.0
    #image_background_np=image_background_np.astype(np.float32)
    image_chars_np = image_chars_np.astype(np.float32)

    for d in range(deep-1):
        image_background_np[..., d]=(1-alpha)* (image_background_np[..., d])+alpha*image_chars_np[...,d]
    image_background_np.astype(np.uint8)
    image_background = Image.fromarray(image_background_np)
    return  image_background

def readFromDir(path,extension='.png'):
    result=[]
    for file in os.listdir(path):
        if file.endswith(extension):
            result.append(os.path.join(path,file))
    return result

def readImagList(pathImage):
    result={}
    for i in range(10):
        pathtemp=pathImage+"/0000%2d/"%(48+i)
        samplelist = readFromDir(pathtemp, extension='.png')
        index=random.randint(0,len(samplelist)-1)
        result[str(i)]=samplelist[index]
    return result

def makePic(chars,pathImage,font_size=20,char_spacing=10):
    chars=list(chars)
    length=len(chars)
    rate = font_size / 20.0
    imagDic=readImagList(pathImage)
    bbimage = Image.open(imagDic[chars[0]])
    width,height =bbimage.size
    i = 0
    left = 0
    up = 0
    height=int(rate * height)
    down = height
    image_background = Image.new('RGB', ((50) * width+50*char_spacing, 15*height), '#FFFFFF')

    bigW,bigH=image_background.size
    up=int(bigH/2)
    right=int(bigW/2-(length/2)*rate*width-(length/2+5)*char_spacing)-char_spacing
    if right<-char_spacing:
        right=-char_spacing
    print image_background.size
    for i, char in enumerate(chars):
        char_image = Image.open(imagDic[char])

        width,height = char_image.size
        width = int(rate * width)
        height = int(rate * height)
        char_image = char_image.resize((width, height))
        down = up+height
        left = int(right + char_spacing)
        right = left + width
        #特殊间隔
        if (length==16 and i==6) or (length==19 and i%4 == 0):
            left=left+3*char_spacing
            right=right+3*char_spacing
        box = (int(left), int(up), int(right), int(down))
        print box
        region = image_background.crop(box)
        char_image = copyBck_1( region,char_image)
        image_background.paste(char_image,box)
    #image_background.show()
    return image_background

    #return image_background.crop(tepm.getbbox())

def adjustBck(box,img,char_size):
    smallH,smallW = char_size
    bigW, bigH = img.size
    # bigH,bigW=img.size
    # smallH,smallW=bbimage.size
    print img.size
    print "-----------"
    prop = 1
    # 随机选择背景高度
    if bigH > 2 * smallH:
        prop = random.uniform(1, 2)
    cropHeight = int(prop * smallH)
    print prop
    print cropHeight
    # (wew,weh,wew1,weh1)=(int(prop*wew),int(prop*weh),int(prop*wew1),int(prop*weh1))
    prop = (float(cropHeight) / float(bigH))
    (wew, weh, wew1, weh1)=box
    weh = int(prop * weh)
    weh1 = int(prop * weh1)
    # 在竖直方向随机裁剪背景
    up = random.randint(0, bigH - cropHeight)
    left = 0
    right = bigW
    down = up + cropHeight
    img = img.crop((left, up, right, down))

    print (wew, weh, wew1, weh1)

    # 在新的背景下随机粘贴卡号
    if weh + smallH >= weh1:
        return None
    heheight = random.randint(weh + smallH, weh1)
    weh1 = heheight
    weh = heheight - smallH
    return img,(wew, weh, wew1, weh1)



