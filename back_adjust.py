# -*- coding:utf-8 -*-
import numpy as np
from numpy import random

class BackAdjust(object):
    def __init__(self,bckImage,charSize=None,oriBox=None,Height=None,Width=None):
        self.bckImage=bckImage
        self.charSize = charSize
        self.oriBox = oriBox
        self.Height = Height
        self.Width = Width
    def yinhangcard(self):
        img=self.bckImage
        char_size=self.charSize
        box=self.oriBox

        smallH, smallW = char_size
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
        (wew, weh, wew1, weh1) = box
        weh = int(prop * weh)
        weh1 = int(prop * weh1)
        # 在竖直方向随机裁剪背景
        up = random.randint(0, bigH - cropHeight)
        left = 0
        right = bigW
        down = up + cropHeight
        img = img.crop((left, up, right, down))

        print (wew, weh, wew1, weh1)
        # 返回调整后的背景图像和粘贴区域
        return img, (wew, weh, wew1, weh1)

    def signature(self):
        img=self.bckImage
        Height = self.Height
        Width = self.Width

        cropwit = random.randint(Width[0], Width[1])
        if cropwit > img.size[0]:
            r = float(cropwit) / img.size[0]
            print r, "ssssss"
            img = img.resize((cropwit, int(r * img.size[1])))
        croph = random.randint(Height[0], Height[1])
        if croph > img.size[1]:
            r = float(croph) / img.size[1]
            print r, "ssssss"
            img = img.resize((int(r * img.size[0]), croph))
        img = img.crop((0, 0, cropwit, croph))

        #返回调整后的背景图像和粘贴区域
        return img, (0, 0, img.size[0], img.size[1])