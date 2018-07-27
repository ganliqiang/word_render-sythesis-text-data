# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import cv2
#import fontforge
import hashlib
import math
import numpy as np
import os
import random
import shutil
import time
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from tools import checkFileName
from tools import progressbar
from tools import readFromDir
from scipy.interpolate import griddata


table = np.zeros(256)
for i in range(256):
    table[i] = (255 - i)
table[table < 32] = 0

def makeDirDict(dir):
    print dir
    print '正在制作生成图片列表字典...'
    if not isinstance(dir, unicode):
        dir = dir.decode('utf-8')
    dir_dict = {}
    dir_list = readFromDir(dir=dir, extension='')
    dir_dict.setdefault('root_dir', dir)
    begin_time = time.time()
    total = len(dir_list)
    for i in range(total):
        dir = dir_list[i]
        dir_dict.setdefault(dir, readFromDir(dir=dir + '/', extension='.bmp'))
        #progressbar(cur=i + 1, total=total, begin_time=begin_time, cur_time=time.time())
    return dir_dict

def getInsectionOfCharsAndFonts(chars, font):
    if not os.path.exists(font):
        raise ValueError(font, 'is not exist.')
    file_name = os.path.join(os.path.split(font)[0], hashlib.md5(font + str(time.time())).hexdigest().upper() + '.ttf')
    file_name = checkFileName(save_name=file_name)
    shutil.copy(font, file_name)
    fnt = fontforge.open(file_name)
    os.remove(file_name)
    chars_set = set(chars)
    chars_set.update()
    fonts_set = set()
    for g in fnt.glyphs():
        code = g.unicode
        if code != -1:
            fonts_set.add(unichr(code))
    insection_list = list(chars_set & fonts_set)
    insection_list.sort()
    chars_output = ''.join(char for char in insection_list)
    return chars_output

def genGaussianKernal(dim = 7, sigma = 4):
    gaussian_kernel = np.zeros((dim, dim), dtype=np.float32)
    half_dim = (dim - 1) / 2
    s2 = 2.0 * sigma * sigma
    for i in range(-half_dim, half_dim + 1):
        m = i + half_dim
        for j in range(-half_dim, half_dim + 1):
            n = j + half_dim
            v = math.exp(-(float(i * i) + float(j * j)) / s2)
            gaussian_kernel[m, n] = v
    all = sum(sum(gaussian_kernel))
    gaussian_kernel = gaussian_kernel / all
    return gaussian_kernel

def _draw_character(file_name, crop=True, processing = True, block = True):
    im = Image.open(file_name)
    im_np = np.array(im)
    im = im.convert('L').point(table)
    if crop:
        im = im.crop(im.getbbox())
    im = pasteWithSide(image=im)
    if processing:
        im = randomProcessing(image=im)
    if block:
        im = pasteOnImage(image=im, width=72, height=72)
    return im

def getPrintCaptcha(chars = 'TEST', font_file = None, font_size = 48, char_color = (255, 255, 255), back_color = (0, 0, 0)):
    if font_file is None:
        font_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../fonts/msyh.ttc')
    font = truetype(font_file, font_size)
    image = Image.new('RGB', (0, 0))
    draw = Draw(image)
    w, h = draw.textsize(chars, font=font)
    im = Image.new('RGB', (w, h), color=back_color)
    Draw(im).text((0, 0), chars, font=font, fill=char_color)
    im = im.crop(im.getbbox())
    return im
def getPrintCaptcha1(chars= 'TEST', font_file = None, font_size = 48, vertical = False, char_color = (255, 255, 255), back_color = (0, 0, 0)):
    images = []
    max_ls = []
    for char in chars:
        img = getPrintCaptcha(chars=char, font_file=font_file, font_size=font_size, char_color=char_color, back_color=back_color)
        w, h = img.size
        max_l = max(w, h)
        max_ls.append(max_l)
        if vertical:
            img  = img.transpose(Image.ROTATE_90)
        images.append(img)
    max_ls = max(max_ls)
    positions = []
    for loop, img in enumerate(images):
        image = Image.new('RGB', (max_ls, max_ls), back_color)
        w, h = img.size
        r = min(float(max_ls) / w, float(max_ls) / h)
        img = img.resize((int(w * r), int(h * r)), resample=Image.BILINEAR)
        offset_w = (max_ls - int(w * r)) / 2
        offset_h = (max_ls - int(h * r)) / 2
        image.paste(img, (offset_w, offset_h))
        positions.append([offset_w, offset_h, int(w * r), int(h * r)])
        images[loop] = image
    random_space = random.randint(0, int(round(max_ls * 0.16)))
    space = []
    for i in range(len(images) - 1):
          print i
          space.append(random.randint(random_space, random_space))
    image_background = Image.new('RGB', (max_ls * len(chars) + sum(space), max_ls), back_color)
    offset_w = 0
    offset_h = 0
    for loop, positon in enumerate(positions):
        print loop
        img = images[loop]
        w, h = img.size
        image_background.paste(img, (offset_w, offset_h))
        positions[loop] = [positon[0] + offset_w, positon[1] + offset_h, positon[2], positon[3]]
        if space:
           offset_w = offset_w + w + space[i]
        else:
             offset_w = offset_w + w
    return image_background, positions
def getPrintCaptcha2(chars= 'TEST', font_file = None, font_size = 48, vertical = False, char_color = (255, 255, 255), back_color = (0, 0, 0)):
    images = []
    max_ls = []
    for char in chars:
        img = getPrintCaptcha(chars=char, font_file=font_file, font_size=font_size, char_color=char_color, back_color=back_color)
        w, h = img.size
        max_l = max(w, h)
        max_ls.append(max_l)
        if vertical:
            img  = img.transpose(Image.ROTATE_90)
        images.append(img)
    max_ls = max(max_ls)
    positions = []
    for loop, img in enumerate(images):
        image = Image.new('RGB', (max_ls, max_ls), back_color)
        w, h = img.size
        r = min(float(max_ls) / w, float(max_ls) / h)
        img = img.resize((int(w * r), int(h * r)), resample=Image.BILINEAR)
        offset_w = (max_ls - int(w * r)) / 2
        offset_h = (max_ls - int(h * r)) / 2
        image.paste(img, (offset_w, offset_h))
        positions.append([offset_w, offset_h, int(w * r), int(h * r)])
        images[loop] = image
    random_space = random.randint(0, int(round(max_ls * 0.76)))
    space = []
    for i in range(len(images) - 1):
        space.append(random.randint(random_space, random_space))
    image_background = Image.new('RGB', ((max_ls * len(chars) + sum(space))* 4, max_ls * 2), back_color)
    

    xx1,yy1 = image_background.size

    offset_w = xx1 / 3
    offset_h = yy1 / 3
    for loop, positon in enumerate(positions):
        img = images[loop]
        w, h = img.size
        image_background.paste(img, (offset_w, offset_h))
        positions[loop] = [positon[0] + offset_w, positon[1] + offset_h, positon[2], positon[3]]
        offset_w = offset_w + w + space[i]
    
    xx1,yy1 = image_background.size
    #image_background1 = Image.new('RGB', (4* xx1, yy1* 2), back_color)
    #xx2,yy2 = image_background1.size


    print image.size

    #box =   (xx2/2- xx1/2 ,yy2/2- yy1/2, xx2/2+  xx1/2 ,yy2/2+yy1/2    )
    #print box
    #image_background1.paste(image_background,box)
    


     
    return image_background, positions

def getPrintCaptcha3(chars= 'TEST', font_file = None, font_size = 48, vertical = False, char_color = (255, 255, 255), back_color = (0, 0, 0)):
    images = []
    max_ls = []
    for char in chars:
        img = getPrintCaptcha(chars=char, font_file=font_file, font_size=font_size, char_color=char_color, back_color=back_color)
        w, h = img.size
        max_l = max(w, h)
        max_ls.append(max_l)
        if vertical:
            img  = img.transpose(Image.ROTATE_90)
        images.append(img)
    max_ls = max(max_ls)
    positions = []
    for loop, img in enumerate(images):
        image = Image.new('RGB', (max_ls, max_ls), back_color)
        w, h = img.size
        r = min(float(max_ls) / w, float(max_ls) / h)
        img = img.resize((int(w * r), int(h * r)), resample=Image.BILINEAR)
        offset_w = (max_ls - int(w * r)) / 2
        offset_h = (max_ls - int(h * r)) / 2
        image.paste(img, (offset_w, offset_h))
        positions.append([offset_w, offset_h, int(w * r), int(h * r)])
        images[loop] = image
    random_space = random.randint(0, int(round(max_ls * 0.76)))
    space = []
    for i in range(len(images) - 1):
        space.append(random.randint(random_space, random_space))
    image_background = Image.new('RGB', ((max_ls * len(chars) + sum(space))* 2 , max_ls * 2), back_color)


    xx1,yy1 = image_background.size

    offset_w = 1
    offset_h = yy1 / 3 * 1
    for loop, positon in enumerate(positions):
        img = images[loop]
        w, h = img.size
        image_background.paste(img, (offset_w, offset_h))
        positions[loop] = [positon[0] + offset_w, positon[1] + offset_h, positon[2], positon[3]]
        offset_w = offset_w + w + space[i]

    xx1,yy1 = image_background.size
    #image_background1 = Image.new('RGB', (4* xx1, yy1* 2), back_color)
    #xx2,yy2 = image_background1.size


    print image.size

    #box =   (xx2/2- xx1/2 ,yy2/2- yy1/2, xx2/2+  xx1/2 ,yy2/2+yy1/2    )
    #print box
    #image_background1.paste(image_background,box)




    return image_background, positions



def getHandwritingCaptcha(chars = 'TEST', width = 200, height = 75, dir_dict = None, space_inf = -8, space_sup = -8,
                          hanzi = False, offset = None, crop = True, processing = True, block = True, pos = False):
    if dir_dict is None:
        raise ValueError('没有指定dir字典,使用函数makeDirDict(dir)生成!')
    file_name_list = []
    root_dir = dir_dict['root_dir']
    if hanzi:
        if not isinstance(chars, unicode):
            chars = chars.decode('utf-8')
        for c in chars:
            file_list = dir_dict[os.path.join(root_dir, c)]
            random_num = random.randint(0, len(file_list) - 1)
            file_name = file_list[random_num]
            file_name_list.append(file_name)
    else:
        for c in chars:
            ascii = ord(c)
            ascii=56
            file_list = dir_dict[os.path.join(root_dir, '%06d' % ascii)]
            random_num = random.randint(0, len(file_list) - 1)
            file_name = file_list[random_num]
            file_name_list.append(file_name)
    image = Image.new('RGB', (width, height))
    images = []
    h_resize = height
    for i in range(len(file_name_list)):
        im = _draw_character(file_name=file_name_list[i], crop=crop, processing=processing, block=block)
        im_np=np.array(im)
        w, h = im.size
        r = float(h_resize) / h
        im = im.resize((int(r * w), int(r * h)), resample=Image.BILINEAR)
        images.append(im)
    space = []
    for i in range(len(images)):
        space.append(random.randint(space_inf, space_sup))
    text_width = sum([im.size[0] for im in images]) + sum(space[0:-1])
    width_new = max(text_width, width)
    image = image.resize((width_new, height), resample=Image.BILINEAR)
    if offset is None:
        offset_w = (width_new - text_width) / 2
        offset_h = 0
    else:
        offset_w, offset_h = offset
    positions = []
    for i in range(len(images)):
        im = images[i]
        w, h = im.size
        image.paste(im, (offset_w, offset_h + int((height - h) / 2)), mask=im)
        positions.append([offset_w, offset_h + int((height - h) / 2), w, h])
        offset_w = offset_w + w + space[i]
    if width_new > width:
        image = image.resize((width, height), resample=Image.BILINEAR)
        r = float(width) / width_new
        for pos_idx, position in enumerate(positions):
            positions[pos_idx] = [int(round(position[0] * r)), position[1], int(round(position[2] * r)), position[3]]
    im = image.filter(ImageFilter.SMOOTH).convert('L')
    if pos:
        return im, positions
    else:
        return im

def preProcessing(image = None):
    image = image.convert('L')
    image = image.point(table)
    np_im = np.array(image, dtype=np.float32)
    if np_im.max() == np_im.min():
        np_im = np.array(np_im * 0, dtype=np.uint8)
    else:
        np_im = np.array((np_im - np_im.min()) * 255 / (np_im.max() - np_im.min()), dtype=np.uint8)
    image = Image.fromarray(np_im)
    image = image.crop(image.getbbox())
    image = pasteWithSide(image=image)
    return image

def randomProcessing(image = None):
    image = image.convert('L')
    w, h = image.size
    image = image.rotate(random.uniform(-5, 5), Image.BICUBIC, expand=1)
    dx = w * random.uniform(0., 0.1)
    dy = h * random.uniform(0., 0.1)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (x1, y1, -x1, h2 - y2, w2 + x2, h2 + y2, w2 - x2, -y1)
    image = image.resize((w2, h2), resample=Image.BILINEAR)
    image = image.transform((w, h), Image.QUAD, data)
    return image

def randomDistort(image = None, alpha = 1000):
    image = image.convert('L')
    image = pasteWithSide(image=image, padding=20)
    w, h = image.size
    dx = -1 + 2 * np.random.rand(h, w)
    dy = -1 + 2 * np.random.rand(h, w)
    gaussian_kernel = genGaussianKernal(dim=7, sigma=4)
    fdx = np.array(cv2.filter2D(np.array(dx), -1, gaussian_kernel))
    fdy = np.array(cv2.filter2D(np.array(dy), -1, gaussian_kernel))
    n = sum(sum(np.square(fdx) + np.square(fdy)))
    fdx = fdx * alpha / n
    fdy = fdy * alpha / n
    x, y = np.meshgrid(range(w), range(h))
    x_new = x - fdx
    y_new = y - fdy
    x_new[x_new < 0] = 0
    y_new[y_new < 0] = 0
    x_new[x_new > w - 1] = w - 1
    y_new[y_new > h - 1] = h - 1
    x = x.reshape(x.size)
    y = y.reshape(y.size)
    x_new = x_new.reshape(x_new.size)
    y_new = y_new.reshape(y_new.size)
    image_np = np.array(image)
    image_np = image_np.reshape(image_np.size)
    image_np = griddata((x, y), image_np, (x_new, y_new), method='nearest').reshape((h, w))
    image = Image.fromarray(image_np[10:h-10, 10:w-10])
    image = image.crop(image.getbbox())
    return image

def pasteOnImage(image = None, width = 200, height = 75, background = None, offset = None):
    image = image.convert('L')
    if background is None:
        image_background = Image.new('L', (width, height))
    else:
        image_background = background
    if offset is None:
        w, h = image.size
        r = min(float(width) / w, float(height) / h)
        image = image.resize((int(w * r), int(h * r)), resample=Image.BILINEAR)
        offset_w = int((width - w * r) / 2)
        offset_h = int((height - h * r) / 2)
        image_background.paste(image, (offset_w, offset_h))
    else:
        offset_w, offset_h = offset
        image_background.paste(image, (offset_w, offset_h))
    return image_background

def pasteWithSide(image = None, padding = 1):
    image = image.convert('L')
    w, h = image.size
    image_background = Image.new('L', (w + 2 * padding, h + 2 * padding))
    image_background.paste(image, (padding, padding))
    return image_background

def getHandwritingData(image = None, width = 200, height = 75):
    image = image.convert('L')
    image = preProcessing(image=image)
    # image = randomDistort(image=image, alpha=random.randint(1, 1000))
    # image = randomProcessing(image=image)
    image = pasteOnImage(image=image, width=width, height=height)
    return image
