# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import sys
sys.path.append('..')
import argparse
import math
import numpy as np
import os
import random
import time
from PIL import Image
from extension import create_corpus
from PIL import ImageOps
from extension.hwcaptcha import getInsectionOfCharsAndFonts
from extension.hwcaptcha import getPrintCaptcha1,getPrintCaptcha,getHandwritingCaptcha
from extension.tools import progressbar
from extension.tools import readFromDir
from extension.tools import saveImage
from extension.hwcaptcha import makeDirDict
from skimage import exposure, util
import re

import json
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='')
    parser.add_argument('--save_pic_dir', help='保存图片的目录路径(以/结尾)')
    parser.add_argument('--sample', type=int, default=10, help='样本数量')
    parser.add_argument('--vertical', type=bool, default=False, help='是否竖向')
    parser.add_argument('--gen_pic_dir', help='图片源路径(以/结尾)')
    parser.add_argument('--fonts_idx', type=int, default=0)
    return parser.parse_args()
def create_amount_time(sample = 10, inf = 4, sup = 13):
    all_chars = u'0123456789'
    chars_list = []
    from random import *
    from time import *
    print '正在shengjin...'
    for loop in range(100000):
        date1 = (2010, 1, 1, 0, 0, 0, -1, -1, -1)
        time1 = mktime(date1)
        date2 = (2050, 1, 1, 0, 0, 0, -1, -1, -1)
        time2 = mktime(date2)
        random_time = uniform(time1, time2)  # uniform返回随机实数 time1 <= time < time2
        abc = strftime('%Y-%m-%d %H:%M:%S', localtime(random_time))
        numtype = randint(0, 2)
        if numtype == 0:
                  acadgd = abc.split()[0].replace('-', '')
        if numtype == 1:
                   acadgd = abc.split()[0].replace('-', ' ')
        if numtype == 2:
              acadgd = abc.split()[0].replace('-', '.')
        if numtype == 3:
              acadgd = abc.split()[0].replace('-', '/')
        if numtype == 4:
               aldfa = abc.split()[0].split('-')
               acadgd = aldfa[0] + u"年" + aldfa[1] + u"月" + aldfa[2] + u"日"
        # print acadgd
        chars_list.append(acadgd)
   
    return chars_list




from operator import mod
import random

def personcard():
    d1 = random.randint(1, 6)

    if d1 == 1 or d1 == 5 or d1 == 6:
        d2 = random.randint(0, 5)
    elif d1 == 2 or d1 == 4:
        d2 = random.randint(0, 3)
    elif d1 == 3:
        d2 = random.randint(0, 7)

    d3to4 = random.randint(1, 70)
    d5to6 = random.randint(1, 99)

    d_birth_year = random.randint(1935, 2017)
    d_birth_month = random.randint(1, 12)

    if d_birth_month == 2:
        d_birth_day = random.randint(1, 28)
    else:
        d_birth_day = random.randint(1, 30)

    d15to17 = random.randint(100, 700)
    d18 = random.randint(0, 9)

    return ('%d''%d''%02d''%02d''%02d''%02d''%02d''%02d''%d' \
           % (d1, d2, d3to4, d5to6, d_birth_year, \
              d_birth_month, d_birth_day, d15to17, d18))



def readfile_2():
    f = open("/home/user/wxb/GEN_DATA/cortext/beijing.txt", "r")
    lists = []
    abdc = []
    i = 0
    while True:
             
        
        i = i + 1

        if i == 20002:
            break
        # print i
        line = f.readline()

        # print i,"saa"
        if line:
            if i %2 == 0:
               pass
            else:

                str = line.strip()
                abdc.append(str)
            
        # print i
            # print str,len(str)       
    f.close()
    return  abdc

def handl(abc ,filecode):
   s= abc
   b = ''
   for tt in s:
      if tt in filecode:
          b = b +tt
          pass
      else:
           b = b + random.choice(filecode)
   return b
def passcard(abci):
   b=random.randint(1000000000000,9000000000000)

   return   ('%d' %b)
  
   return  bb

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

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
    image_background.show()
    return  image_background
def readFromDirPath(dir = None):
    if dir is None:
        print '目录为空!(tools->readFromDirIter)'
        exit()
    elif not dir.endswith('/'):
        print '目录应以/结尾!(tools->readFromDirIter)'
        exit()

    def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                list_name.append( file_path )
                listdir(file_path, list_name)
            else:
                # list_name.append(file_path)
                sys.stdout.write('\r')
                sys.stdout.write('已扫描%d个文件' % len(list_name))
                sys.stdout.flush()
    file_list_tmp = []
    listdir(path=dir, list_name=file_list_tmp)
    file_list = []
    for file_name in file_list_tmp:
            file_list.append(file_name)
    sys.stdout.write('\n')
    return file_list



if __name__ == '__main__':
    args = parse_args()
    if args.save_pic_dir is None:
        raise ValueError('没有指定保存路径(以/结尾)')
    # 读取参数
    
    gen_pic_dir = args.gen_pic_dir
    save_pic_dir = args.save_pic_dir
    # 显示参数配置
    print '保存图片路径:', save_pic_dir
    fonts1 = readFromDir(dir='../bfont/', extension='.TTF')
    fonts2 = readFromDir(dir='../bfont/', extension='.ttf')
    fonts = fonts1 + fonts2
    fonts.sort()
    samplelist = readFromDir(dir='../sampley/dianhuipingzheng/2/', extension='.jpg')
    print samplelist
    print fonts
    if args.fonts_idx >= len(fonts):
        raise ValueError('fonts_idx大于等于fonts长度%d' % len(fonts))
    font = fonts[args.fonts_idx]
    # print font
    chars_colors = [  '#2f404a','#2f404a','#2f404a','#2f404a','#2f404a',]
    back_colors = [ '#FFFFFF', '#FFFFFF', '#FFFFFF', ]
    add_chars = u'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫαβγ'
    listnmae = readfile_2()
    
    
    file_listpath = readFromDirPath(dir="/mnt/wxb/gen_pic_src_chars/") 
    #print file_listpath
    #print  file_listpath[1].split('/')
    mapdit = dict()
    filecode = []
    file16l = []
    for tt in file_listpath:

        # pathfime, imgelname = os.path.split(tt)
        # print imgelname.split("_")[0],imgelnametemp


        numdddd = tt.split('/')[len(tt.split('/')) - 1]
        #print numdddd
        abcddd = unichr(int(numdddd))
        # abcddd = unichr( int(   imgelname.split("_")[0] ,16))
        # if  imgelnametemp == imgelname.split("_")[0]:
        # print imgelnametemp ,abcdddi
        code = int( numdddd )
        code16  =  '%04x' % code

        filecode.append(code16)




 

    dir_dict = makeDirDict(dir=gen_pic_dir)
    #chars = create_corpus.common_used_character(tranditional = False, radical = False, alphabeta = True, number = True,
                                               # serial_number = False, symbol = False)
    #chars = getInsectionOfCharsAndFonts(chars + add_chars, font=font)
    #add_chars_exist = u''
    #for char in add_chars:
    #    if char in chars:
    #        add_chars_exist += char
    #if len(add_chars_exist) > 0:
    #    num_of_rome = len(chars) / len(add_chars_exist) / 6 - 1
    #    chars += add_chars_exist * num_of_rome
    #chars_list = create_corpus.character(chars = chars, inf=4, sup=17, sample=args.sample)
    begin_time = time.time()
    chars_list = create_amount_time(sample = 10000, inf=4, sup=17)
    total = len(chars_list)
    height_range = [31, 35]
    i = 0
    while i < 1 :
   # len(fonts):

     #args.fonts_idx = i
     i = i + 1
     
     print font
     for loop, chars in enumerate(chars_list):
        chars = passcard(listnmae)
        chars = chars.decode("utf-8" )
        chars =  handl(chars ,filecode)
        output_dir = 'FONTS_%03d/output_%03d' % (args.fonts_idx, int(loop / 10000))
        color_idx = random.randint(0, len(chars_colors) - 1)
        color_idx = 0
        if 1: 
            #continue

            #image= getPrintCaptcha(chars=chars, font_file=font, font_size =35 , 
            #                        char_color=chars_colors[color_idx], back_color=back_colors[color_idx])
            






            width1 = len(chars) * 15

            image = getHandwritingCaptcha(chars=chars, dir_dict=dir_dict,height=  30,width= width1,
                                            space_inf=-10, space_sup=0,
                                            offset=(random.randint(-5, 10), random.randint(-15, 15)))           

 




            xx1,yy1 = image.size
            image_background = Image.new('RGB', (4* xx1, yy1* 2), back_colors[color_idx] )
            xx2,yy2 = image_background.size

            bbimage = image
            print image.size
            print image_background.size

            box =   (xx2/2- xx1/2 ,yy2/2- yy1/2, xx2/2+  xx1/2 ,yy2/2+yy1/2    )
            print box
            #image_background.paste(image,box)
            #image = image_background


            image_np = np.array(image)
        

            if random.randint(0, 1) == 1:
                 image_np = exposure.adjust_gamma(image=image_np, gamma=math.exp(random.uniform(-1, 1)))
            if random.randint(0, 1) == 1:
                 image_np = util.random_noise(image_np, mode='gaussian', mean=0, var=0.001)
                 image = Image.fromarray(np.array(image_np * 255, dtype=np.uint8))
            else:
                image = Image.fromarray(image_np)
            w, h = image.size
            height_want = random.randint(height_range[0], height_range[1])
            r = float(height_want) / h
            image = image.resize((int(w * r), height_want))
            # saveImage(save_name=os.path.join(save_pic_dir, output_dir, chars + '.jpg'), image=image)
            #saveImage(save_name=os.path.join(save_pic_dir, output_dir, str(loop)+ '.jpg'), image=image)
        

            indexl  = random.randint( 0 , len(samplelist    ) - 1)
            fileimge = samplelist[   indexl  ]



            faldl = fileimge.replace(".jpg",".lar")
            with open( faldl, 'rb') as f:

                 setting = json.load(f,"gbk")
                 print  setting[0]["rect"]
                 tt = setting[0]
            if len(tt["rect"].split(",")) != 4:
                # img.show
                continue
            wew = int(tt["rect"].split(",")[0])
            weh = int(tt["rect"].split(",")[1])
            wew1 = int(tt["rect"].split(",")[2])
            weh1 = int(tt["rect"].split(",")[3]) 
                 


 
            heheight = random.randint(weh + h , weh1)
            weh1 = heheight
            weh = heheight - h



            if   w < wew1 -wew:
                wew1 = wew +w
            img =  Image.open( fileimge )

            height_want = weh1 - weh
            r = float(height_want) / h
            bbimage = bbimage.resize((int(w * r), height_want))

            region = img.crop((wew, weh, wew1, weh1))
            bbimage = ImageOps.invert(bbimage)         
   
            bbimage = copyBck( region,bbimage)
            img.paste( bbimage,(wew,weh,wew1,weh1))

        else :
            
            
            char1 = chars[ :18]


            char2 = chars[ 18:]

         

            image= getPrintCaptcha(chars=char1, font_file=font, font_size =35 ,
                                    char_color=chars_colors[color_idx], back_color=back_colors[color_idx])
            
                
            image1= getPrintCaptcha(chars=char2, font_file=font, font_size =35 ,char_color=chars_colors[color_idx], back_color=back_colors[color_idx])


    


            xx1,yy1 = image.size
            image_background = Image.new('RGB', (4* xx1, yy1* 2), back_colors[color_idx] )
            xx2,yy2 = image_background.size

            bbimage = image
            print image.size
            print image_background.size

            box =   (xx2/2- xx1/2 ,yy2/2- yy1/2, xx2/2+  xx1/2 ,yy2/2+yy1/2    )
            print box
            #image_background.paste(image,box)
            #image = image_background


            image_np = np.array(image)


            if random.randint(0, 1) == 1:
                 image_np = exposure.adjust_gamma(image=image_np, gamma=math.exp(random.uniform(-1, 1)))
            if random.randint(0, 1) == 1:
                 image_np = util.random_noise(image_np, mode='gaussian', mean=0, var=0.001)
                 image = Image.fromarray(np.array(image_np * 255, dtype=np.uint8))
            else:
                image = Image.fromarray(image_np)
            w, h = image.size
            w1, h1 = image1.size
            height_want = random.randint(height_range[0], height_range[1])
            r = float(height_want) / h
            image = image.resize((int(w * r), height_want))
            # saveImage(save_name=os.path.join(save_pic_dir, output_dir, chars + '.jpg'), image=image)
            #saveImage(save_name=os.path.join(save_pic_dir, output_dir, str(loop)+ '.jpg'), image=image)


            indexl  = random.randint( 0 , len(samplelist    ) - 1)
            fileimge = samplelist[   indexl  ]



            faldl = fileimge.replace(".jpg",".lar")
            with open( faldl, 'rb') as f:

                 setting = json.load(f,"gbk")
                 print  setting[0]["rect"]
                 tt = setting[0]
            if len(tt["rect"].split(",")) != 4:
                # img.show
                continue
            wew = int(tt["rect"].split(",")[0])
            weh = int(tt["rect"].split(",")[1])
            wew1 = int(tt["rect"].split(",")[2])
            weh1 = int(tt["rect"].split(",")[3])

            heheight = random.randint(weh + h , weh1- h- 4)
            weh1 = heheight
            weh = heheight - h
            if   w < wew1 -wew:
                wew1 = wew +w
            img =  Image.open( fileimge )

            height_want = weh1 - weh
            r = float(height_want) / h
            bbimage = bbimage.resize((int(w * r), height_want))
            bbimage1 = image1.resize((int(w1 * r), height_want))

            region = img.crop((wew, weh, wew1, weh1))
            region1 = img.crop((wew, weh1 + 4, wew + w1, weh1 + 4 + height_want))

            bbimage = copyBck( region,bbimage)
            bbimage1 = copyBck( region1,image1)

            img.paste( bbimage,(wew,weh,wew1,weh1))
            img.paste( bbimage1, (wew, weh1 + 4, wew + w1, weh1 + 4 + height_want) )


        mkdir(        save_pic_dir + '/'+  output_dir )
        img.save( os.path.join(save_pic_dir, output_dir, str(loop)+ '.jpg' ))
        output_dirrpr = 'FONTS_%03d' % (args.fonts_idx)
        output_dirtt = 'output_%03d' % ( int(loop / 10000))
        filesaveinfo = open(os.path.join(save_pic_dir, output_dirrpr, "EngSynthesisSample_" + str(loop / 10000)), "a+")
        chars = u"账号" + chars 
        filesaveinfo.write( output_dirtt  + "/" + str(loop) + ".jpg" + " " + chars.encode("utf-8") + "\t\n")
        filesaveinfo.close()
        progressbar(cur=loop + 1, total=total, begin_time=begin_time, cur_time=time.time(), info='loop:%d' % (loop + 1))

