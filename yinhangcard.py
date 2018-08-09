# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import sys
sys.path.append('..')
import numpy as np
import os
import random
from PIL import Image
import math



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

def readImagList(pathImage):
    result={}
    for file in os.listdir(pathImage):
        filePath=os.path.join(pathImage,file)
        if os.path.isdir(filePath):
            temp=[]
            for subfile in os.listdir(filePath):
                temp.append(os.path.join(filePath,subfile))
            result[file.decode("utf-8")]=temp
    return result




def makePic(charslist,imagDic,font_size=20,char_spacing=10,adjust_value={}):
    #替换“点”

    standardW=[30,34]
    stdW=random.randint(standardW[0],standardW[1])
    rate0 = font_size / 20.0
    result=[]
    for chars in charslist:
        chars=list(chars)
        length=len(chars)


        bbimage = Image.open(random.choice(imagDic[chars[0]]))

        width,height0 =bbimage.size
        height0=39
        i = 0
        left = 0
        up = 0


        image_background = Image.new('RGB', ((8) * width+5*char_spacing, 5*height0), '#FFFFFF')

        bigW,bigH=image_background.size
        down=int(bigH/2)
        #中心
        right=int(bigW/2-(length/2)*0.3*width-(length/2+5)*char_spacing)-char_spacing
        if right<-char_spacing:
            right=-char_spacing
        print image_background.size
        for i, char in enumerate(chars):
            if imagDic.has_key(char):
                char_image = Image.open(random.choice(imagDic[char]))
                width,height = char_image.size
                rate=rate0*float(stdW)/float(height)
                rate=1

                width = int(rate * width)
                height = int(rate * height)
                char_image = char_image.resize((width, height))
                up=down -height
                left = int(right + char_spacing)
                right = left + width
                #特殊间隔
                #if (length==16 and i==6) or (length==19 and i%4 == 0):
                 #   left=left+3*char_spacing
                  #  right=right+3*char_spacing
                flag=False
                delt=0
                if adjust_value.has_key(char):
                    delt=adjust_value[char]*height0
                up=up-delt
                down=down-delt
                box = (int(left), int(up), int(right), int(down))
                print box
                region = image_background.crop(box)
                char_image = copyBck( region,char_image)
                image_background.paste(char_image,box)
                up = up + delt
                down = down + delt
        #image_background.show()
        result.append(image_background)

    return result

    #return image_background.crop(tepm.getbbox())


def adjust3(charlist,boxs):
    try:
        if boxs[0][2]-charlist[0][1]<=boxs[0][0]:
            left0=boxs[0][0]
        else:left0=random.randint(boxs[0][0],boxs[0][2]-charlist[0][1])

        if boxs[2][2] - charlist[2][1] <= boxs[2][0]:
            left2 = boxs[2][0]
        else:left2=random.randint(boxs[2][0],boxs[2][2]-charlist[2][1])
        dis=left2-left0
        deltH=int(dis*math.tan(math.radians(2)))
        if boxs[0][3]-charlist[0][0]<=boxs[0][1]:
            down0=boxs[0][1]+charlist[0][0]
            print(boxs[0][3],charlist[0][0],boxs[0][1])
        else:down0=random.randint(boxs[0][1]+charlist[0][0],boxs[0][3])
        down2 = random.randint(down0-deltH, down0+deltH)
        down2 = max(down2, boxs[2][1]+charlist[2][0])
        down2 = min(down2, boxs[2][3])
        #if boxs[2][3]-charlist[2][0]<=boxs[2][1]:
         #   down2=boxs[2][1]+charlist[2][0]
        #else:down2=random.randint(boxs[2][1]+charlist[2][0],boxs[2][3])


    except Exception:
        pass
    right0=left0+charlist[0][1]
    center=(right0+left2)/2.0

    left1=int(center-charlist[1][1]/2.0)
    if left1<boxs[1][0]:
        left1=boxs[1][0]
    if left1>boxs[1][2]-charlist[1][1]:
        left1=boxs[1][2]-charlist[1][1]
    down1=int((float(down2-down0)/float(left2-left0))*(left1-left0)+down0)
    down1=max(down1,boxs[1][1])
    down1=min(down1,boxs[1][3])
    result=[]
    result.append((left0,down0-charlist[0][0],left0+charlist[0][1],down0))
    result.append((left1, down1- charlist[1][0], left1 + charlist[1][1], down1 ))
    result.append((left2, down2- charlist[2][0], left2 + charlist[2][1], down2 ))
    return result
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



