# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Sqrt5


import glob
import hashlib
import math
import os
import sys


def progressbar(cur, total, begin_time = 0, cur_time = 0, info = ''):
    sys.stdout.write('\r')
    if begin_time == 0 and cur_time == 0 or cur == 0:
        sys.stdout.write("[%-30s]\t%6.2f%%" %
                         ('=' * int(math.floor(cur * 30 / total)),
                         ('=' * int(math.floor(cur * 30 / total)),
                          float(cur * 100) / total)))
    else:
        sys.stdout.write("[%-30s]  %6.2f%%   COST: %.2fs   ETA: %.2fs   %s     " %
                         ('=' * int(math.floor(cur * 30 / total)),
                          float(cur * 100) / total,
                          (cur_time - begin_time),
                          (cur_time - begin_time) * (total - cur) / cur,
                          info))
    if cur == total:
        print ''
    sys.stdout.flush()
    return

def readFromDir(dir = None, extension = None):
    if dir is None:
        raise ValueError('目录为空')
    elif not dir.endswith('/'):
        raise ValueError('目录应以/结尾')
    if extension is None:
        raise ValueError('没用指定扩展名')
    file_list = glob.glob(os.path.join(os.path.split(dir)[0], '*' + extension))
    return file_list

def readFromDirIter(dir = None, extension = None):
    if dir is None:
        raise ValueError('目录为空')
    elif not dir.endswith('/'):
        raise ValueError('目录应以/结尾')
    if extension is None:
        raise ValueError('没用指定扩展名')
    def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)
                sys.stdout.write('\r')
                sys.stdout.write('已扫描%d个文件' % len(list_name))
                sys.stdout.flush()
    file_list_tmp = []
    listdir(path=dir, list_name=file_list_tmp)
    file_list = []
    for file_name in file_list_tmp:
        if file_name.endswith(extension):
            file_list.append(file_name)
    sys.stdout.write('\n')
    return file_list

def checkDir(dir = None):
    if dir is None:
        raise ValueError('没有指定目录')
    elif not dir.endswith('/'):
        raise ValueError('目录应以/结尾')
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def splitext(name):
    index = name.rfind('.')
    return name[0:index], name[index:]

def checkFileName(save_name = None):
    if save_name is None:
        raise ValueError('没有指定保存名称')
    dir = os.path.split(save_name)[0]
    checkDir(dir=dir + '/')
    name, ext = splitext(os.path.split(save_name)[1])
    cnt = 0
    save_name = os.path.join(dir, '%s_No.%04d%s' % (name, cnt, ext))
    while os.path.exists(save_name):
        cnt += 1
        save_name = os.path.join(dir, '%s_No.%04d%s' % (name, cnt, ext))
    return save_name

def saveImage(save_name = None, image = None):
    if save_name is None:
        raise ValueError('没有指定save_name')
    if image is None:
        raise ValueError('没有指定image')
    dir = os.path.split(save_name)[0]
    checkDir(dir=dir + '/')
    name, ext = splitext(os.path.split(save_name)[1])
    cnt = 0
    save_name = os.path.join(dir, '%s_No.%04d%s' % (name, cnt, ext))
    while os.path.exists(save_name):
        cnt += 1
        save_name = os.path.join(dir, '%s_No.%04d%s' % (name, cnt, ext))
    image.save(save_name)
    return save_name

def md5sum(file_name):
    f = open(file_name, 'rb')
    md5 = hashlib.md5(f.read()).hexdigest().upper()
    f.close()
    return md5

def dictInverse(input_dict):
    output_dict = {}
    for k, v in input_dict.items():
        for value in v:
            if value not in output_dict:
                output_dict[value] = k
            else:
                raise ValueError(k, value, v)
    return output_dict