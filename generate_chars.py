# -*- coding:utf-8 -*-
import numpy as np
import os
import re
import random
import time
from create_corpus import create_amount_in_word,create_amount_in_figure

class CharsGenerator(object):
    def __init__(self, char_dir=None, charsSet=None, charLen=1):
        self.char_dir=char_dir
        self.len=charLen
        self.charsSet=charsSet

    #出生
    def persion_birth(self,path=None):
        while True:
            date1 = (2010, 1, 1, 0, 0, 0, -1, -1, -1)
            time1 = time.mktime(date1)
            date2 = (2050, 1, 1, 0, 0, 0, -1, -1, -1)
            time2 = time.mktime(date2)
            random_time = random.uniform(time1, time2)  # uniform返回随机实数 time1 <= time < time2
            abc = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(random_time))
            numtype = random.randint(0, 2)
            numtype = 4
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
            yield acadgd

    #姓名
    def persion_name(self,path):
        f = open(path, "r")
        lists = []
        abdc = []
        i = 0
        while True:
            i = i + 1
            if i == 1028295:
                break
            # print i
            line = f.readline()
            # print i,"saa"
            if line:
                # print line
                str = line.decode('utf-8')
                # print str,len(str)

                matchObj = re.match(r'http://.+resgain.net/name/(.+).html', str, re.M | re.I)

                if matchObj:
                    # print "matchObj.group() : ", matchObj.group()
                    # print "matchObj.group(1) : ", matchObj.group(1)

                    abdc.append(matchObj.group(1))
                    # print  len(abdc), "www"
        f.close()
        while True:
            yield random.choice(abdc)

    #公民身份号码
    def persion_id(self,path=None):
        while True:
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

            yield ('%d''%d''%02d''%02d''%02d''%02d''%02d''%02d''%d' \
                   % (d1, d2, d3to4, d5to6, d_birth_year, \
                      d_birth_month, d_birth_day, d15to17, d18))


    #电汇凭证--------
    #NO
    def dianhuipingzheng_id(self,path=None):
        while True:
            yield str(random.randint(10000000, 99999999))

    #账号
    def dianhuipingzheng_num(self,path=None):
        #b = random.randint(1000000000000, 9000000000000)
        pass
    #全称
    def dianhuipingzheng_name(self,path=None):
        pass


    #开户许可证----------
    #核准号
    def kaihuxukezheng_num(self,path=None):
        while True:
            s = []
            for ch in xrange(0x41, 0x5A):
                s.append(unichr(ch))
            a = random.choice(s)
            b = random.randint(100000000000000, 900000000000000)

            yield (a + ('%d' % b))

    # 开户银行
    def kaihuxukezheng_bank(self,path=None):
        return self.shuiwudengjizheng_1(path)

    #税务登记证--------
    #纳税人名称
    def shuiwudengjizheng_1(self,path=None):
        f = open(path)
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
                if i % 2 == 0:
                    pass
                else:

                    str = line.strip()
                    str = str.decode('utf-8')
                    abdc.append(str)

            # print i
            # print str,len(str)
        f.close()
        while True:
            yield random.choice(abdc)

    #地址
    def shuiwudengjizheng_2(self,path=None):
        return self.shuiwudengjizheng_1(path)

    #银行卡号------------
    def yinhangcard_num(self,length):
        while True:
            leng=random.choice(length)
            if leng == 16:
                # 16位
                id = random.randint(1000000000000000, 9999999999999999)
            elif leng==19:
                id = random.randint(1000000000000000000, 9999999999999999999)
            else:
                raise ValueError("chars' length must be 19 or 16")
            yield str(id)

    #营业执照------------
    #住所
    def yingyezhichao_addr(self,path=None):
        return self.shuiwudengjizheng_1(path)
    #注册资本
    def yingyezhichao_money(self,path=None):
        for num in  create_amount_in_figure(prefix=u''):
            yield (('%s' % num) + u"万元")

    #注册资本 大写金额
    def yingyezhichao_money_big(self,path=None):
        return create_amount_in_word()


    #增值税专用发票----------
    #""
    def zhuangyongtax_none(self,path=None):
        while True:
            b = random.randint(1000000000, 9999999999)

            yield ('%d' % b)
    #No
    def zhuangyongtax_No(self,path=None):
        while True:
            b = random.randint(10000000, 99999999)

            yield ('%d' % b)
    #￥
    def zhuangyongtax_money(self,path=None):
        while True:
            b = random.randint(0, 99999999)

            c = random.randint(0, 99)

            yield (u"￥" + ('%d' % b) + "." + ('%d' % c))

    #名称
    def zhuangyongtax_name(self,path=None):
        return self.shuiwudengjizheng_1(path)

    #开票日期
    def zhuangyongtax_date(self,path=None):
        return self.persion_birth(path)


    #组织机构代码证-----------------
    #代码
    def zujijigou_code(self,path=None):
        while True:
            b = random.randint(00000000, 99999999)
            c = random.randint(0, 9)
            yield (('%d' % b) + '-' + ('%d' % c))
    #机构名称
    def zujijigou_addr(self,path=None):
        return self.shuiwudengjizheng_1(path)