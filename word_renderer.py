# Max Jaderberg 12/5/14
# -*- coding:utf-8 -*-
# Module for rendering words
# BLEND MODES: http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
# Rendered words have three colours - base char (0) and border/shadow (128) and background (255)

# TO RUN ON TITAN: use source ~/Work/virtual_envs/paintings_env/bin/activate

import sys
import pygame
import os
import re
from pygame.locals import *
import numpy as n
from pygame import freetype
import math
from matplotlib import pyplot
from PIL import Image
from scipy import ndimage, interpolate
import scipy.cluster
from matplotlib import cm
import random
from scipy.io import loadmat
import time
import h5py
import pandas as pd
import cv2
import json
from skimage import exposure, util
from yinhangcard import makePic,adjust3
from back_adjust import BackAdjust

def wait_key():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                return


def rgb2gray(rgb):
    # RGB -> grey-scale (as in Matlab's rgb2grey)
    try:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    except IndexError:
        try:
            gray = rgb[:, :, 0]
        except IndexError:
            gray = rgb[:, :]
    return gray


def resize_image(im, r=None, newh=None, neww=None, filtering=Image.BILINEAR):
    dt = im.dtype
    I = Image.fromarray(im)
    if r is not None:
        h = im.shape[0]
        w = im.shape[1]
        newh = int(round(r * h))
        neww = int(round(r * w))
    if neww is None:
        neww = int(newh * im.shape[1] / float(im.shape[0]))
    if newh > im.shape[0]:
        I = I.resize([neww, newh], Image.ANTIALIAS)
    else:
        I.thumbnail([neww, newh], filtering)
    return n.array(I).astype(dt)


def matrix_mult(A, B):
    C = n.empty((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i, j] = n.sum(A[i, :] * B[:, j])
    return C


def save_screen_img(pg_surface, fn, quality=100):
    imgstr = pygame.image.tostring(pg_surface, 'RGB')
    im = Image.fromstring('RGB', pg_surface.get_size(), imgstr)
    im.save(fn, quality=quality)
    print
    fn


MJBLEND_NORMAL = "normal"
MJBLEND_ADD = "add"
MJBLEND_SUB = "subtract"
MJBLEND_MULT = "multiply"
MJBLEND_MULTINV = "multiplyinv"
MJBLEND_SCREEN = "screen"
MJBLEND_DIVIDE = "divide"
MJBLEND_MIN = "min"
MJBLEND_MAX = "max"


def grey_blit1(src, dst, blend_mode=MJBLEND_NORMAL):
    """
    This is for grey + alpha images
    """
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    # blending with alpha http://stackoverflow.com/questions/1613600/direct3d-rendering-2d-images-with-multiply-blending-mode-and-alpha
    # blending modes from: http://www.linuxtopia.org/online_books/graphics_tools/gimp_advanced_guide/gimp_guide_node55.html
    dt = dst.dtype
    src = src.astype(n.single)
    dst = dst.astype(n.single)
    out = n.empty(src.shape, dtype='float')
    alpha = n.index_exp[:, :, 1]
    rgb = n.index_exp[:, :, 0]
    src_a = src[alpha] / 255.0
    dst_a = dst[alpha] / 255.0
    out[alpha] = src_a + dst_a * (1 - src_a)
    old_setting = n.seterr(invalid='ignore')

    src_pre = src[rgb] * src_a
    dst_pre = dst[rgb] * dst_a
    # blend:
    blendfuncs = {MJBLEND_NORMAL: lambda s, d, sa_: s + d * sa_,
                  MJBLEND_ADD: lambda s, d, sa_: n.minimum(255, s + d),
                  MJBLEND_SUB: lambda s, d, sa_: n.maximum(0, s - d),
                  MJBLEND_MULT: lambda s, d, sa_: s * d * sa_ / 255.0,
                  MJBLEND_MULTINV: lambda s, d, sa_: (255.0 - s) * d * sa_ / 255.0,
                  MJBLEND_SCREEN: lambda s, d, sa_: 255 - (1.0 / 255.0) * (255.0 - s) * (255.0 - d * sa_),
                  MJBLEND_DIVIDE: lambda s, d, sa_: n.minimum(255, d * sa_ * 256.0 / (s + 1.0)),
                  MJBLEND_MIN: lambda s, d, sa_: n.minimum(d * sa_, s),
                  MJBLEND_MAX: lambda s, d, sa_: n.maximum(d * sa_, s), }
    out[rgb] = blendfuncs[blend_mode](src_pre, dst_pre, (1 - src_a))
    out[rgb] /= out[alpha]
    n.seterr(**old_setting)
    out[alpha] *= 255
    n.clip(out, 0, 255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype(dt)
    return out
def grey_blit(src, dst, blend_mode=MJBLEND_NORMAL):
    """
    This is for grey + alpha images
    """
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    # blending with alpha http://stackoverflow.com/questions/1613600/direct3d-rendering-2d-images-with-multiply-blending-mode-and-alpha
    # blending modes from: http://www.linuxtopia.org/online_books/graphics_tools/gimp_advanced_guide/gimp_guide_node55.html
    dt = dst.dtype
    src = src.astype(n.single)
    dst = dst.astype(n.single)
    out = n.empty(src.shape, dtype='float')
    alpha = n.index_exp[:, :, 1]
    rgb = n.index_exp[:, :, 0]
    src_a = src[alpha] / 255.0
    dst_a = dst[alpha] / 255.0
    out[alpha] = src_a + dst_a * (1 - src_a)
    old_setting = n.seterr(invalid='ignore')
    src_pre = src[rgb] * src_a
    dst_pre = dst[rgb] * dst_a
    # blend:
    blendfuncs = {MJBLEND_NORMAL: lambda s, d, sa_: s + d * sa_, MJBLEND_ADD: lambda s, d, sa_: n.minimum(255, s + d),
        MJBLEND_SUB: lambda s, d, sa_: n.maximum(0, s - d), MJBLEND_MULT: lambda s, d, sa_: s * d * sa_ / 255.0,
        MJBLEND_MULTINV: lambda s, d, sa_: (255.0 - s) * d * sa_ / 255.0,
        MJBLEND_SCREEN: lambda s, d, sa_: 255 - (1.0 / 255.0) * (255.0 - s) * (255.0 - d * sa_),
        MJBLEND_DIVIDE: lambda s, d, sa_: n.minimum(255, d * sa_ * 256.0 / (s + 1.0)),
        MJBLEND_MIN: lambda s, d, sa_: n.minimum(d * sa_, s), MJBLEND_MAX: lambda s, d, sa_: n.maximum(d * sa_, s), }
    out[rgb] = blendfuncs[blend_mode](src_pre, dst_pre, (1 - src_a))
    out[rgb] /= out[alpha]
    n.seterr(**old_setting)
    out[alpha] *= 255
    n.clip(out, 0, 255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype(dt)
    return out


class Corpus(object):
    """
    Defines a corpus of words
    """
    # valid_ascii = [48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]
    valid_ascii = [36, 37, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100,
                   101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                   121, 122]

    def __init__(self):
        pass


class TestCorpus(Corpus):
    """
    Just a test corpus from a text file
    """
    CORPUS_FN = "./corpus.txt"

    def __init__(self, args={'unk_probability': 0}):
        self.corpus_text = ""
        pattern = re.compile('[^a-zA-Z0-9 ]')
        for line in open(self.CORPUS_FN):
            line = line.replace('\n', ' ')
            line = pattern.sub('', line)
            self.corpus_text = self.corpus_text + line
        self.corpus_text = ''.join(c for c in self.corpus_text if c.isalnum() or c.isspace())
        self.corpus_list = self.corpus_text.split()
        self.unk_probability = args['unk_probability']

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = n.random.randint(0, len(self.corpus_list))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.corpus_list[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = n.random.randint(0, diff)
                        samp = samp[starti:starti + length]
                    break
            else:
                break
            idx = n.random.randint(0, len(self.corpus_list))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")
        if n.random.rand() < self.unk_probability:
            # change some letters to make it random
            ntries = 0
            while True:
                ntries += 1
                if len(samp) > 2:
                    n_to_change = n.random.randint(2, len(samp))
                else:
                    n_to_change = max(1, len(samp) - 1)
                idx_to_change = n.random.permutation(len(samp))[0:n_to_change]
                samp = list(samp)
                for idx in idx_to_change:
                    samp[idx] = chr(random.choice(self.valid_ascii))
                samp = "".join(samp)
                if samp not in self.corpus_list:
                    idx = len(self.corpus_list)
                    break
                if ntries > 10:
                    idx = self.corpus_list.index(samp)
                    break
        return samp, idx


class SVTCorpus(TestCorpus):
    CORPUS_FN = "/Users/jaderberg/Data/TextSpotting/DataDump/svt1/svt_lex_lower.txt"


class FileCorpus(TestCorpus):
    def __init__(self, args):
        self.CORPUS_FN = args['fn']
        TestCorpus.__init__(self, args)


class NgramCorpus(TestCorpus):
    """
    Spits out a word sample, dictionary label, and ngram encoding labels
    """

    def __init__(self, args):
        words_fn = args['encoding_fn_base'] + '_words.txt'
        idx_fn = args['encoding_fn_base'] + '_idx.txt'
        values_fn = args['encoding_fn_base'] + '_values.txt'

        self.words = self._load_list(words_fn)
        self.idx = self._load_list(idx_fn, split=' ', tp=int)
        self.values = self._load_list(values_fn, split=' ', tp=int)

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = n.random.randint(0, len(self.words))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.words[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = n.random.randint(0, diff)
                        samp = samp[starti:starti + length]
                    break
            else:
                break
            idx = n.random.randint(0, len(self.words))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")

        return samp, {'word_label': idx, 'ngram_labels': self.idx[idx], 'ngram_counts': self.values[idx], }

    def _load_list(self, listfn, split=None, tp=str):
        arr = []
        for l in open(listfn):
            l = l.strip()
            if split is not None:
                l = [tp(x) for x in l.split(split)]
            else:
                l = tp(l)
            arr.append(l)
        return arr


class RandomCorpus(Corpus):
    """
    Generates random strings
    """

    def __init__(self, args={'min_length': 1, 'max_length': 23}):
        self.min_length = args['min_length']
        self.max_length = args['max_length']

    def get_sample(self, length=None):
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        samp = ""
        for i in range(length):
            samp = samp + chr(random.choice(self.valid_ascii))
        return samp, length


class FontState(object):
    """
    Defines the random state of the font rendering
    """
    #size = [60, 10]  # normal dist mean, std
    underline = 0.1
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.02778, 0.05333]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = 1.0
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 1
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, path="/home/ubuntu/Datasets/SVT/",fontSize=25,isRandom=False,fontPic=False):
        self.fonts = [os.path.join(path, f.strip()) for f in os.listdir(path)]
        self.size=[fontSize,fontSize]
        self.isRandom=isRandom
        if fontPic:
            result = {}
            for file in os.listdir(path):
                filePath = os.path.join(path, file)
                if os.path.isdir(filePath):
                    temp = []
                    for subfile in os.listdir(filePath):
                        temp.append(os.path.join(filePath, subfile))
                    result[file.decode("utf-8")] = temp
            if result.has_key("point"):
                result[u"."] = result.pop("point")
            self.fontPic = result

    def get_sample(self):
        """
        Samples from the font state distribution
        """
        if not self.isRandom:
            return {'font': self.fonts[int(n.random.randint(0, len(self.fonts)))],
                'size': self.size[0], 'underline': False,
                'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1] * n.random.randn() +
                                                     self.underline_adjustment[0])),
                'strong': False, 'oblique': False,
                'strength': self.strength[0],
                'char_spacing': 0,
                'border': False, 'random_caps': False,
                'capsmode': random.choice(self.capsmode), 'curved': False,
                'random_kerning': False,
                'random_kerning_amount': self.random_kerning_amount, }
        else:
            return {'font': self.fonts[int(n.random.randint(0, len(self.fonts)))],
                'size': self.size[0], 'underline': n.random.rand() < self.underline,
                'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1] * n.random.randn() +
                                                     self.underline_adjustment[0])),
                'strong': n.random.rand() < self.strong, 'oblique': n.random.rand() < self.oblique,
                'strength': (self.strength[1] - self.strength[0]) * n.random.rand() + self.strength[0],
                'char_spacing': int(self.kerning[3] * (n.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
                'border': n.random.rand() < self.border, 'random_caps': n.random.rand() < self.random_caps,
                'capsmode': random.choice(self.capsmode), 'curved': n.random.rand() < self.curved,
                'random_kerning': n.random.rand() < self.random_kerning,
                'random_kerning_amount': self.random_kerning_amount, }

class AffineTransformState(object):
    """
    Defines the random state for an affine transformation
    """
    proj_type = Image.AFFINE
    rotation = [0, 5]  # rotate normal dist mean, std
    skew = [0, 0]  # skew normal dist mean, std

    def sample_transformation(self, imsz):
        #theta = math.radians(self.rotation[1] * n.random.randn() + self.rotation[0])
        theta = math.radians(random.uniform(-5,5))
        ca = math.cos(theta)
        sa = math.sin(theta)
        R = n.zeros((3, 3))
        R[0, 0] = ca
        R[0, 1] = -sa
        R[1, 0] = sa
        R[1, 1] = ca
        R[2, 2] = 1
        S = n.eye(3, 3)
        S[0, 1] = math.tan(math.radians(self.skew[1] * n.random.randn() + self.skew[0]))
        A = matrix_mult(R, S)
        x = imsz[1] / 2
        y = imsz[0] / 2
        return (A[0, 0], A[0, 1], -x * A[0, 0] - y * A[0, 1] + x, A[1, 0], A[1, 1], -x * A[1, 0] - y * A[1, 1] + y)


class PerspectiveTransformState(object):
    """
    Defines teh random state for a perspective transformation
    Might need to use http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    """
    proj_type = Image.PERSPECTIVE
    a_dist = [1, 0.01]
    b_dist = [0, 0.005]
    c_dist = [0, 0.005]
    d_dist = [1, 0.01]
    e_dist = [0, 0.0005]
    f_dist = [0, 0.0005]

    def v(self, dist):
        return dist[1] * n.random.randn() + dist[0]

    def sample_transformation(self, imsz):
        x = imsz[1] / 2
        y = imsz[0] / 2
        a = self.v(self.a_dist)
        b = self.v(self.b_dist)
        c = self.v(self.c_dist)
        d = self.v(self.d_dist)
        e = self.v(self.e_dist)
        f = self.v(self.f_dist)

        # scale a and d so scale kept same
        # a = 1 - e*x
        # d = 1 - f*y

        z = -e * x - f * y + 1
        A = n.zeros((3, 3))
        A[0, 0] = a + e * x
        A[0, 1] = b + f * x
        A[0, 2] = -a * x - b * y - e * x * x - f * x * y + x
        A[1, 0] = c + e * y
        A[1, 1] = d + f * y
        A[1, 2] = -c * x - d * y - e * x * y - f * y * y + y
        A[2, 0] = e
        A[2, 1] = f
        A[2, 2] = z
        # print a,b,c,d,e,f
        # print z
        A = A / z
        #(A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 1])
        import numpy as np
        return np.float64(A)


class ElasticDistortionState(object):
    """
    Defines a random state for elastic distortions
    """
    displacement_range = 1
    alpha_dist = [[15, 30], [0, 2]]
    sigma = [[8, 2], [0.2, 0.2]]
    min_sigma = [4, 0]

    def sample_transformation(self, imsz):
        choices = len(self.alpha_dist)
        c = int(n.random.randint(0, choices))
        sigma = max(self.min_sigma[c], n.abs(self.sigma[c][1] * n.random.randn() + self.sigma[c][0]))
        alpha = n.random.uniform(self.alpha_dist[c][0], self.alpha_dist[c][1])
        dispmapx = n.random.uniform(-1 * self.displacement_range, self.displacement_range, size=imsz)
        dispmapy = n.random.uniform(-1 * self.displacement_range, self.displacement_range, size=imsz)
        dispmapx = alpha * ndimage.gaussian_filter(dispmapx, sigma)
        dispmaxy = alpha * ndimage.gaussian_filter(dispmapy, sigma)
        return dispmapx, dispmaxy


class BorderState(object):
    outset = 0.5
    width = [4, 4]  # normal dist
    position = [[0, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]]

    def get_sample(self):
        p = self.position[int(n.random.randint(0, len(self.position)))]
        w = max(1, int(self.width[1] * n.random.randn() + self.width[0]))
        return {'outset': n.random.rand() < self.outset, 'width': w,
            'position': [int(-1 * n.random.uniform(0, w * p[0] / 1.5)), int(-1 * n.random.uniform(0, w * p[1] / 1.5))]}


class ColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """
    IMFN = "/home/ubuntu/Datasets/text-renderer/image_24_results.png"

    def __init__(self):
        self.im = rgb2gray(n.array(Image.open(self.IMFN)))

    def get_sample(self, n_colours):
        # print 'Inside Color State'
        a = self.im.flatten()

        codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
        # get std of centres
        vecs, dist = scipy.cluster.vq.vq(a, codes)

        colours = []

        for i in range(n_colours):
            try:
                code = codes[i]
                std = n.std(a[vecs == i])
                colours.append(std * n.random.randn() + code)
            except IndexError:
                print
                "\tcolour error"
                colours.append(int(sum(colours) / float(len(colours))))
        # choose randomly one of each colour
        return n.random.permutation(colours)


class TrainingCharsColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """

    def __init__(self, matfn="/home/ubuntu/Datasets/SVT/icdar_2003_train.txt"):
        # self.ims = loadmat(matfn)['images']
        # with open(matfn) as f: self.ims = f.read().splitlines()
        #list_fn = list(pd.read_csv(matfn, sep='\t')['Image_Path'])
        list_fn=[os.path.join(matfn,file.strip()) for file in os.listdir(matfn) if file.endswith(".jpg")]
        self.ims = list_fn

    def get_sample(self, n_colours):
        curs = 0
        while True:
            curs += 1
            if curs > 1000:
                print
                "problem with colours"
                break

            # im_sample=cv2.imread(random.choice(self.ims),0)
            # im = cv2.normalize(im_sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
            imfn = random.choice(self.ims)
            im = rgb2gray(n.array(Image.open(imfn)))

            # im = self.ims[...,n.random.randint(0, self.ims.shape[2])]

            a = im.flatten()
            codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
            if len(codes) != n_colours:
                continue
            # get std of centres
            vecs, dist = scipy.cluster.vq.vq(a, codes)
            colours = []
            for i, code in enumerate(codes):
                std = n.std(a[vecs == i])
                colours.append(std * n.random.randn() + code)
            break
        # choose randomly one of each colour
        return n.random.permutation(colours)


class FillImageState(object):
    """
    Handles the images used for filling the background, foreground, and border surfaces
    """
    DATA_DIR = '/home/ubuntu/Pictures/'
    IMLIST = ['maxresdefault.jpg', 'alexis-sanchez-arsenal-wallpaper-phone.jpg', 'alexis.jpeg']
    blend_amount = [0.0, 0.25]  # normal dist mean, std
    blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
    blend_order = 0.5
    min_textheight = 16.0  # minimum pixel height that you would find text in an image

    def get_sample(self, surfarr):
        """
        The image sample returned should not have it's aspect ratio changed, as this would never happen in real world.
        It can still be resized of course.
        """
        # load image
        imfn = random.choice(self.IMLIST)
        baseim = n.array(Image.open(imfn))

        # choose a colour channel or rgb2gray

        if baseim.ndim == 3:
            if n.random.rand() < 0.25:
                baseim = rgb2gray(baseim)
            else:
                baseim = baseim[..., n.random.randint(0, 3)]
        else:
            assert (baseim.ndim == 2)

        imsz = baseim.shape
        surfsz = surfarr.shape

        # don't resize bigger than if at the original size, the text was less than min_textheight
        max_factor = float(surfsz[0]) / self.min_textheight
        # don't resize smaller than it is smaller than a dimension of the surface
        min_factor = max(float(surfsz[0] + 5) / float(imsz[0]), float(surfsz[1] + 5) / float(imsz[1]))
        # sample a resize factor
        factor = max(min_factor, min(max_factor, ((max_factor - min_factor) / 1.5) * n.random.randn() + max_factor))
        sampleim = resize_image(baseim, factor)
        imsz = sampleim.shape
        # sample an image patch
        good = False
        curs = 0
        while not good:
            curs += 1
            if curs > 1000:
                print
                "difficulty getting sample"
                break
            try:
                x = n.random.randint(0, imsz[1] - surfsz[1])
                y = n.random.randint(0, imsz[0] - surfsz[0])
                good = True
            except ValueError:
                # resample factor
                factor = max(min_factor,
                             min(max_factor, ((max_factor - min_factor) / 1.5) * n.random.randn() + max_factor))
                sampleim = resize_image(baseim, factor)
                imsz = sampleim.shape
        imsample = (n.zeros(surfsz) + 255).astype(surfarr.dtype)
        imsample[..., 0] = sampleim[y:y + surfsz[0], x:x + surfsz[1]]
        imsample[..., 1] = surfarr[..., 1].copy()
        #imsample[..., 1] = sampleim[y:y + surfsz[0], x:x + surfsz[1]]
        #imsample[..., 2] = sampleim[y:y + surfsz[0], x:x + surfsz[1]]
        return {'image': imsample, 'blend_mode': random.choice(self.blend_modes),
            'blend_amount': min(1.0, n.abs(self.blend_amount[1] * n.random.randn() + self.blend_amount[0])),
            'blend_order': n.random.rand() < self.blend_order, }


# class SVTFillImageState(FillImageState):
#    def __init__(self, data_dir, gtmat_fn):
#        self.DATA_DIR = data_dir
#        gtmat = loadmat(gtmat_fn)['gt']
#        with open(gtmat_fn) as f: self.IMLIST = f.read().splitlines()

class SVTFillImageState(FillImageState):

    def __init__(self, label_data_dir,random_data_dir,isNoise):
        list_fn = []
        if os.path.exists(label_data_dir):
            for file in os.listdir(label_data_dir):
                if file.endswith(".jpg"):
                    list_fn.append(os.path.join(label_data_dir,file))
        self.label_back = list_fn
        if isNoise:
            list_fn = []
            if os.path.exists(random_data_dir):
                for file in os.listdir(random_data_dir):
                    if file.endswith(".jpg"):
                        list_fn.append(os.path.join(random_data_dir, file))
            self.IMLIST = list_fn

        # print list_fn
        # for gtmat_fn in list_fn:
        #    with open(gtmat_fn) as f:
        #        self.IMLIST += (f.read().splitlines())
        #        print self.IMLIST
    def get_sample_p(self, surfarr1,outconfig=None):
        import numpy as np

        try :
            fileimge = random.choice(self.label_back)

            img = Image.open(fileimge)
        except Exception:
            if len(surfarr1)>1:
                return None
            surfarr=surfarr1[0]
            surfarr=surfarr.astype(np.uint8)
            surfarr[:] = 255 - surfarr[:]
            surfarr = surfarr.reshape((surfarr.shape[0], surfarr.shape[1], 1))
            return np.concatenate((surfarr, surfarr, surfarr), axis=2)
        faldl = fileimge.replace(".jpg", ".lar")
        boxs=[]
        #设置粘贴区域
        try:
            with open(faldl, 'rb') as f:
                setting = json.load(f, "gbk")

                print  setting[0]["rect"]
                for tt in setting[:len(surfarr1)]:
                    if len(tt["rect"].split(",")) != 4:
                        # img.show
                        return
                    left = int(tt["rect"].split(",")[0])
                    up = int(tt["rect"].split(",")[1])
                    right = int(tt["rect"].split(",")[2])
                    down = int(tt["rect"].split(",")[3])
                    boxs.append((left, up, right, down))
        except Exception:
            (left, up, right, down)=(2,2,img.size[0], img.size[1])
            boxs.append((left, up, right, down))
        char_shape_list=[surfarr.shape for surfarr in surfarr1]
        boxslist=boxs
        if len(boxs)>2:
            boxslist=adjust3(char_shape_list,boxs)


        for surfarr,(left, up, right, down) in zip(surfarr1,boxslist):
            box0 = (left, up, right, down)

            #背景调整
            try:
                adjust=BackAdjust(img,charSize=surfarr.shape[:2],oriBox=box0,Height=outconfig["output_height"],Width=outconfig["output_width"])
                adjust_fun=getattr(adjust,outconfig["back_adjust"])
                img, (left, up, right, down)=adjust_fun()
            except Exception:
                #未设置背景调整
                pass

            h, w = surfarr.shape[:2]
            if up + h > down or left + w > right:
                print("error:the font size is too big")
                #粘贴区域不足以容下字符图
                raise ValueError
            #生成随机区域
            if  outconfig["ver_random"]:
                down = random.randint(up + h, down)
                up = down - h
            else:down=up+h

            if  outconfig["hor_random"]:
                right = random.randint(left + w, right)
                left = right - w
            else:right=left+w
            alpha = surfarr

            #转成白底黑字
            alpha[:] = 255 - alpha[:]
            alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
            surfarr = np.concatenate((alpha, alpha, alpha), axis=2)

            region = img.crop((left, up, right, down))
            region = np.array(region)

            surfarr = surfarr.astype(np.float32)
            surfarr[:] = surfarr[:] / 255.0
            if outconfig["red"]:
                surfarr[...,0]=1
            region=region*surfarr
            print(region.shape[0])
            region = region.astype(np.uint8)
            img.paste(Image.fromarray(region), (left, up, right, down))
            #img.show()
        if not outconfig["keep_label"] and False:
            a = random.randint(  0, 3)
            b = random.randint(  0, 3)
            c = random.randint(  0, 3)
            d = random.randint(  0, 3)
            left=max(left-a,0)
            up = max(up - b, 1)
            right=min(img.size[0],right+ c)
            down = min(img.size[1], down + d)
            img = img.crop(( left, up, right,down))
            return np.array(img)
        return np.array(img)

    def get_sample_p_blur(self, surfarr):
        fileimge=random.choice(self.IMLIST)
        #return surfarr

        h,w = surfarr.shape[:2]
        if not self.isRandom:
            faldl = fileimge.replace(".jpg", ".lar")
            with open(faldl, 'rb') as f:

                setting = json.load(f, "gbk")
                print  setting[0]["rect"]
                tt = setting[0]
            if len(tt["rect"].split(",")) != 4:
                # img.show
                return
            left = int(tt["rect"].split(",")[0])
            up = int(tt["rect"].split(",")[1])
            right = int(tt["rect"].split(",")[2])
            down = int(tt["rect"].split(",")[3])

            if up + h>=down:
                print("error:the font size is too big")
            heheight = random.randint(up + h, down)
            down = heheight
            up = heheight - h

            import numpy as np
            img = Image.open(fileimge)
            #surfarr=Image.fromarray(surfarr)
            #surfarr = surfarr.convert('L')

            #surfarr=np.array(surfarr)

            height_want = down - up
            r = float(height_want) / h


            surfarr=np.reshape(surfarr,(height_want, int(w * r),surfarr.shape[2]))
            if   w < right -left:
                right = left +w
            region = img.crop((left, up, right, down))
            region=np.array(region)

            rr=np.max(surfarr[:,:,0])
            rr2 = np.max(surfarr[:, :, 1])
            rr3 = np.max(surfarr[:, :, 2])
            surfarr=Image.fromarray(surfarr)

            surfarr=np.array(surfarr)
            surfarr = rgb2gray(surfarr)
            temp=np.reshape(surfarr,(surfarr.shape[0],surfarr.shape[1],1))
            surfarr=np.concatenate((temp,np.zeros(temp.shape),np.zeros(temp.shape)),axis=2)
            #mean=np.mean(surfarr)
            surfarr[surfarr>35]=255-surfarr[surfarr>35]

            surfarr[surfarr <= 35] = 255
            #surfarr[:]=255-surfarr[:]
            surfarr[:]=surfarr[:]/255.0
            #surfarr=np.concatenate((surfarr, np.ones((surfarr.shape[0],surfarr.shape[1],1))), axis=2)

            if surfarr.shape[:2]==region.shape[:2]:
                region=region*surfarr
            region=region.astype(np.uint8)
            img.paste(Image.fromarray(region), (left, up, right, down))
            #返回3通道图像
            return np.array(img)
        else:return None



class DistortionState(object):
    blur = [0, 1]
    sharpen = 0
    sharpen_amount = [30, 10]
    noise = 4
    resample = 0.1
    resample_range = [24, 32]

    def get_sample(self):
        return {'blur': n.abs(self.blur[1] * n.random.randn() + self.blur[0]),
            'sharpen': n.random.rand() < self.sharpen,
            'sharpen_amount': self.sharpen_amount[1] * n.random.randn() + self.sharpen_amount[0], 'noise': self.noise,
            'resample': n.random.rand() < self.resample,
            'resample_height': int(n.random.uniform(self.resample_range[0], self.resample_range[1]))}


class SurfaceDistortionState(DistortionState):
    noise = 8
    resample = 0


class BaselineState(object):
    curve = lambda this, a: lambda x: a * x * x
    differential = lambda this, a: lambda x: 2 * a * x
    a = [1, 0.1]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        a = self.a[1] * n.random.randn() + self.a[0]
        return {'curve': self.curve(a), 'diff': self.differential(a), }


class WordRenderer(object):
    def __init__(self, sz=(800, 200), corpus=TestCorpus, fontstate=FontState, colourstate=ColourState,
                 fillimstate=FillImageState,info=None):
        # load corpus
        self.corpus = corpus() if isinstance(corpus, type) else corpus
        # load fonts
        self.fontstate = fontstate() if isinstance(fontstate, type) else fontstate
        # init renderer
        pygame.init()
        self.sz = sz
        self.screen = None

        self.perspectivestate = PerspectiveTransformState()
        self.affinestate = AffineTransformState()
        self.borderstate = BorderState()
        self.colourstate = colourstate() if isinstance(colourstate, type) else colourstate
        self.fillimstate = fillimstate() if isinstance(fillimstate, type) else fillimstate
        self.diststate = DistortionState()
        self.surfdiststate = SurfaceDistortionState()
        self.baselinestate = BaselineState()
        self.elasticstate = ElasticDistortionState()
        self.extraInfo=info

    def invert_surface(self, surf):
        pixels = pygame.surfarray.pixels2d(surf)
        pixels ^= 2 ** 32 - 1
        del pixels

    def invert_arr(self, arr):
        arr ^= 2 ** 32 - 1
        return arr

    def apply_perspective_surf(self, surf):
        self.invert_surface(surf)
        data = pygame.image.tostring(surf, 'RGBA')
        img = Image.fromstring('RGBA', surf.get_size(), data)
        img = img.transform(img.size, self.affinestate.proj_type, self.affinestate.sample_transformation(img.size),
                            Image.BICUBIC)
        img = img.transform(img.size, self.perspectivestate.proj_type,
                            self.perspectivestate.sample_transformation(img.size), Image.BICUBIC)
        im = n.array(img)
        # pyplot.imshow(im)
        # pyplot.show()
        surf = pygame.surfarray.make_surface(im[..., 0:3].swapaxes(0, 1))
        self.invert_surface(surf)
        return surf

    def get_rect(self,w,h,angle):
        halfW=w/2.0
        halfH=h/2.0
        #angle1=math.atan(float(h)/w)
        #try:
         #   angle = self.extraInfo["noise_config"]["rotateAngle"]
        #except Exception:
         #   angle=5
        #random_angle=random.uniform(0,angle)
        random_angle=math.radians(angle)

        length=halfW*math.tan(random_angle)
        #length=random.randint(-1,1)*length
        point1=[0,length]
        point2=[w-1,0]
        point3 = [0, h+length]
        point4 = [w-1, h -1]

        import numpy as np
        return np.float32([point1,point2,point3,point4])

    def fandom_dlt_ratio(self, maxdlt, deno, orilen):
        return n.random.randint(-maxdlt, maxdlt) / (1.0*deno)*orilen
    def apply_perspective_arr(self, arr,pts31, filtering=Image.BICUBIC):
        img = Image.fromarray(arr)


        arr = n.array(img)
        #tmp = sum(arr[0, :])
        #if tmp != 0:
         #   raise ValueError('Wrong Pic')

        #img = img.transform(img.size, self.affinestate.proj_type, affstate, filtering)
        #img = img.transform(img.size, self.perspectivestate.proj_type, perstate, filtering)
        
        import numpy as np
        i = 0
        # while i < 10000:
        #    i = i+1
        #    abc=
        #    print abc
        maxdlt = 100
        dltx0 = self.fandom_dlt_ratio( maxdlt, 1000, img.width)
        dltx1 = self.fandom_dlt_ratio( maxdlt, 1000, img.width)
        dltx2 = self.fandom_dlt_ratio(maxdlt, 1000, img.width)
        dltx3 = self.fandom_dlt_ratio(maxdlt, 1000, img.width)
        dlty0 = self.fandom_dlt_ratio(maxdlt, 1000, img.height)
        dlty1 = self.fandom_dlt_ratio(maxdlt, 1000, img.height)

        dlty2 = self.fandom_dlt_ratio(maxdlt, 1000, img.height)
        dlty3 = self.fandom_dlt_ratio(maxdlt, 1000, img.height)

        #dlty2 = self.get_rect(img.width, img.height, 5)
        #dlty3 = self.get_rect(img.width, img.height, 5)
        #pts3 = np.float32([ [dltx0, dlty0], [img.width*(1.0 -n.random.randint(-maxdlt,maxdlt)/1000.0)  , 0], [0, img.height*(1.0-  n.random.randint(-maxdlt,maxdlt)/1000.0) ], [img.width*(1.0 -n.random.randint(-maxdlt,maxdlt)/1000.0), img.height*(1.0 -  n.random.rand()  ] ])
        if dltx0 < 0:
            dltx0 = 0
        if dlty0 <= 0:
            dlty0 = 0
        if  dlty1 <= 0:
            dlty1 = 0
        if img.width+dltx1  < img.width:
            dltx1 = 0
        if img.height + dlty3 > img.height:
            dlty3 = 0
        if img.width+dltx3 > img.width:
            dltx3 = 0


        pts3 = np.float32([[dltx0, dlty0], [img.width+dltx1, dlty1], [dltx2, img.height+dlty2], [img.width+dltx3, img.height+dlty3]])
        #pts31 = self.get_rect(img.width, img.height)
        pts4 = np.float32([ [0, 0], [img.width, 0], [0, img.height], [img.width, img.height ] ])

        abd = cv2.getPerspectiveTransform( pts4, pts31)
        #bba = cv2.cvtColor( arr, cv2.COLOR_GRAY2RGB)
        #adlldla = cv2.cvtColor( bba,cv2.COLOR_RGB2GRAY )
        rows = img.height
        cols = img.width

        # cv2.imshow("OpenCV",adlldla)
        # wait_key()

        imga = cv2.warpPerspective( arr ,abd , (cols, rows))

        # cv2.imshow("OpenCV", adlldla)
        # wait_key()


        img = imga



        arr = n.array(img)
        #tmp = sum(arr[0,:]) + sum(arr[-1,:]) + sum(arr[:, 0]) + sum(arr[:, -1])
        #if tmp != 0:
         #   raise ValueError('Wrong Pic')
        return arr

    def apply_perspective_rectim(self, rects, arr, affstate, perstate):
        rectarr = n.zeros(arr.shape)
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1] + rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0] + rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = (i + 1) * 10
        rectarr = self.apply_perspective_arr(rectarr, affstate, perstate, filtering=Image.NONE)
        newrects = []
        for i, _ in enumerate(rects):
            try:
                newrects.append(pygame.Rect(self.get_bb(rectarr, eq=(i + 1) * 10)))
            except ValueError:
                pass
        return newrects

    def resize_rects(self, rects, arr, outheight):
        rectarr = n.zeros((arr.shape[0], arr.shape[1]))
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1] + rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0] + rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = (i + 1) * 10
        rectarr = resize_image(rectarr, newh=outheight, filtering=Image.NONE)
        newrects = []
        for i, _ in enumerate(rects):
            try:
                newrects.append(pygame.Rect(self.get_bb(rectarr, eq=(i + 1) * 10)))
            except ValueError:
                pass
        return newrects

    def get_image(self):
        data = pygame.image.tostring(self.screen, 'RGBA')
        return n.array(Image.fromstring('RGBA', self.screen.get_size(), data))

    def copyBck_1(image_background, image_chars):
        w, h = image_background.size
        image_chars = image_chars.resize((w, h), Image.ANTIALIAS)

        # image_chars = image_chars.convert('L')
        print image_chars.size

        image_background_np = np.array(image_background)
        image_chars_np = np.array(image_chars)
        h, w, deep = image_chars_np.shape
        for d in range(deep - 1):
            for i in range(h - 1):
                for j in range(w - 1):
                    aa = float(image_chars_np[i, j, deep - 1]) / 255
                    image_background_np[i, j, d] = int(
                        (1 - aa) * float(image_background_np[i, j, d]) + aa * float(image_chars_np[i, j, d]))
        image_background = Image.fromarray(image_background_np)
        return image_background

    def get_ga_image(self, surf):
        import numpy as np
        r = pygame.surfarray.pixels_red(surf)
        dsg=np.max(r)
        a = pygame.surfarray.pixels_alpha(surf)
        dsgfs = np.max(a)
        b=pygame.surfarray.pixels_blue(surf)
        bb=np.max(b)
        g = pygame.surfarray.pixels_green(surf)
        dsfeg = np.max(g)
        r = r.reshape((r.shape[0], r.shape[1], 1))
        a = a.reshape(r.shape)
        g = g.reshape(r.shape)
        b = b.reshape(r.shape)
        return n.concatenate(( r,a), axis=2).swapaxes(0, 1)

    def get_ga_image_p(self, surf):
        import numpy as np
        r = pygame.surfarray.pixels_red(surf)
        dsg = np.max(r)
        a = pygame.surfarray.pixels_alpha(surf)
        dsgfs = np.max(a)
        b = pygame.surfarray.pixels_blue(surf)
        dsfeg = np.max(b)
        r = r.reshape((r.shape[0], r.shape[1], 1))
        a = a.reshape(r.shape)
        b = b.reshape(r.shape)
        return n.concatenate((r, a, b), axis=2).swapaxes(0, 1)

    def arr_scroll(self, arr, dx, dy):
        arr = n.roll(arr, dx, axis=1)
        arr = n.roll(arr, dy, axis=0)
        return arr

    def get_bordershadow(self, bg_arr, colour):
        """
        Gets a border/shadow with the movement state [top, right, bottom, left].
        Inset or outset is random.
        """
        bs = self.borderstate.get_sample()
        outset = bs['outset']
        width = bs['width']
        position = bs['position']

        # make a copy
        border_arr = bg_arr.copy()
        # re-colour
        border_arr[..., 0] = colour
        if outset:
            # dilate black (erode white)
            border_arr[..., 1] = ndimage.grey_dilation(border_arr[..., 1], size=(width, width))
            border_arr = self.arr_scroll(border_arr, position[0], position[1])

            # canvas = 255*n.ones(bg_arr.shape)
            # canvas = grey_blit(border_arr, canvas)
            # canvas = grey_blit(bg_arr, canvas)
            # pyplot.imshow(canvas[...,0], cmap=cm.Greys_r)
            # pyplot.show()

            return border_arr, bg_arr
        else:
            # erode black (dilate white)
            border_arr[..., 1] = ndimage.grey_erosion(border_arr[..., 1], size=(width, width))
            return bg_arr, border_arr

    def add_colour(self, canvas, fg_surf, border_surf=None):
        cs = self.colourstate.get_sample(2 + (border_surf is not None))
        # replace background
        pygame.PixelArray(canvas).replace((255, 255, 255), (cs[0], cs[0], cs[0]), distance=1.0)
        # replace foreground
        pygame.PixelArray(fg_surf).replace((0, 0, 0), (cs[1], cs[1], cs[1]), distance=0.99)

    def get_bb(self, arr, eq=None):
        if eq is None:
            v = n.nonzero(arr > 0)
        else:
            v = n.nonzero(arr == eq)
        xmin = v[1].min()
        xmax = v[1].max()+3
        ymin = v[0].min()
        ymax = v[0].max()+3
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    def stack_arr(self, arrs):
        shp = list(arrs[0].shape)
        shp.append(1)
        tup = []
        for arr in arrs:
            tup.append(arr.reshape(shp))
        return n.concatenate(tup, axis=2)

    def imcrop(self, arr, rect):
        if arr.ndim > 2:
            return arr[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], ...]
        else:
            return arr[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


    def add_fillimage_p(self, arr,outconfig=None):

        return self.fillimstate.get_sample_p(arr,outconfig)





    def add_fillimage(self, arr):
        """
        Adds a fill image to the array.
        For blending this might be useful:
        - http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
        - http://stackoverflow.com/questions/5605174/python-pil-function-to-divide-blend-two-images
        """
        try :
            fis = self.fillimstate.get_sample(arr)
            image = fis['image']
            blend_mode = fis['blend_mode']
            blend_amount = fis['blend_amount']
            blend_order = fis['blend_order']

            # change alpha of the image
            if blend_amount > 0:
                if blend_order:
                    # image[...,1] *= blend_amount
                    image[..., 1] = (image[..., 1] * blend_amount).astype(int)
                    arr = grey_blit(image, arr, blend_mode=blend_mode)
                else:
                    # arr[...,1] *= (1 - blend_amount)
                    arr[..., 1] = (arr[..., 1] * (1 - blend_amount)).astype(int)
                    arr = grey_blit(arr, image, blend_mode=blend_mode)
        except Exception:
            surfsz = arr.shape
            imsample = (n.zeros(surfsz) + 255).astype(arr.dtype)
            arr=imsample
        # pyplot.imshow(image[...,0], cmap=cm.Greys_r)
        # pyplot.show()

        return arr

    def mean_val(self, arr):
        return n.mean(arr[arr[..., 1] > 0, 0].flatten())

    def surface_distortions(self, arr):
        ds = self.surfdiststate.get_sample()
        blur = ds['blur']

        origarr = arr.copy()
        arr = n.minimum(n.maximum(0, arr + n.random.normal(0, ds['noise'], arr.shape)), 255)
        # make some changes to the alpha
        arr[..., 1] = ndimage.gaussian_filter(arr[..., 1], ds['blur'])
        ds = self.surfdiststate.get_sample()
        arr[..., 0] = ndimage.gaussian_filter(arr[..., 0], ds['blur'])
        if ds['sharpen']:
            newarr_ = ndimage.gaussian_filter(origarr[..., 0], blur / 2)
            arr[..., 0] = arr[..., 0] + ds['sharpen_amount'] * (arr[..., 0] - newarr_)

        return arr

    def global_distortions(self, arr):
        # http://scipy-lectures.github.io/advanced/image_processing/#image-filtering
        ds = self.diststate.get_sample()

        blur = ds['blur']
        sharpen = ds['sharpen']
        sharpen_amount = ds['sharpen_amount']
        noise = ds['noise']

        newarr = n.minimum(n.maximum(0, arr + n.random.normal(0, noise, arr.shape)), 255)
        if blur > 0.1:
            newarr = ndimage.gaussian_filter(newarr, blur)
        if sharpen:
            newarr_ = ndimage.gaussian_filter(arr, blur / 2)
            newarr = newarr + sharpen_amount * (newarr - newarr_)

        if ds['resample']:
            sh = newarr.shape[0]
            newarr = resize_image(newarr, newh=ds['resample_height'])
            newarr = resize_image(newarr, newh=sh)

        return newarr

    def get_rects_union_bb(self, rects, arr):
        rectarr = n.zeros((arr.shape[0], arr.shape[1]))
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1] + rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0] + rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = 10
        return self.get_bb(rectarr)

    def apply_distortion_maps(self, arr, dispx, dispy):
        """
        Applies distortion maps generated from ElasticDistortionState
        """
        origarr = arr.copy()
        xx, yy = n.mgrid[0:dispx.shape[0], 0:dispx.shape[1]]
        xx = xx + dispx
        yy = yy + dispy
        coords = n.vstack([xx.flatten(), yy.flatten()])
        arr = ndimage.map_coordinates(origarr, coords, order=1, mode='nearest')
        return arr.reshape(origarr.shape)

    def getPrintCaptcha(self,font,display_text_list,bg_surf,char_spacing,spaceH,size,curved=False,label=" ",adjust_value={}):
        #display_text_list=[['1','2','3','4','5','6','7']]
        display_text = display_text_list[0]
        if "1" in display_text:
            dksj=0
        mid_idx = int(math.floor(len(display_text) / 2))
        curve = [0 for c in display_text]
        rotations = [0 for c in display_text]


        if curved and len(display_text) > 1:
            bs = self.baselinestate.get_sample()
            for i, c in enumerate(display_text[mid_idx + 1:]):
                curve[mid_idx + i + 1] = bs['curve'](i + 1)
                rotations[mid_idx + i + 1] = -int(math.degrees(math.atan(bs['diff'](i + 1) / float(size / 2))))
            for i, c in enumerate(reversed(display_text[:mid_idx])):
                curve[mid_idx - i - 1] = bs['curve'](-i - 1)
                rotations[mid_idx - i - 1] = -int(math.degrees(math.atan(bs['diff'](-i - 1) / float(size / 2))))
            mean_curve = sum(curve) / float(len(curve) - 1)
            curve[mid_idx] = -1 * mean_curve
            # curve=[5*i for i in curve]
        curve = [0] * len(curve)
        # render text (centered)
        char_bbs = []
        # display_text[mid_idx]="N"
        # place middle char

        # render chars to the right
        bg_centerx, bg_centery = (bg_surf.get_rect().centerx, bg_surf.get_rect().centery)
        lineNum = len(display_text_list)
        rotations = [0 for c1 in display_text_list[0]]

        wid=font.get_rect("1")[2]
        flag = False

        adjust = 0
        '''
        adjust_value[u"《"]=int(2.0 * wid)
        adjust_value[u"|"] = int(1.5 * wid)
        adjust_value[u"“"] = int(2.5 * wid)
        adjust_value[u"〈"] = int(2.0 * wid)
        adjust_value[u"￥"] = int(1 * wid)
        adjust_value[u"‘"] = int(2.5 * wid)
        adjust_value[u"·"] = int(2.0 * wid)
        adjust_value[u"（"]=int(1.7 * wid)
        adjust_value[u"１"] = int(1.7 * wid)
        adjust_value[u"I"] = int(1* wid)
        adjust_value[u"1"] = int(0.5 * wid)
        '''
        '''
        if label=="NO":
            #adjust = int(0.7 * wid)
            adjust=0
        if label == "No":
            adjust = int(0.5 * wid)
        if label == "核准号":
            adjust=int(0.5 * wid)
        if label == "代码":
            adjust = int(0.6 * wid)
        '''

        rect = bg_surf.get_rect()
        for i in range(lineNum):
            if len(display_text_list[i]) <= mid_idx:
                rect.centery += (rect.height + spaceH + curve[mid_idx])
                continue

            c = display_text_list[i][mid_idx]

            rect = font.get_rect(c)
            rect.centerx = bg_centerx
            rect.centery = bg_centery
            #if c == u"（" or c == u"１":
             #   rect.x = rect.x - int(1.7 * rect[2])
            rect.centery += i * (rect.height + spaceH + curve[mid_idx])
            print(rect,)
            if i == 0:

                if adjust_value.has_key(c.encode("utf-8")):#c == "1"
                    adjust=int(adjust_value[c.encode("utf-8")] * wid)
                    flag = True
                    rect.x = rect.x - adjust
                last_rect = rect
            bbrect = font.render_to(bg_surf, rect, c, rotation=rotations[mid_idx])  # fgcolor=(200,255,255)
            #if c == u"（" or c == u"１":
             #   rect.x = rect.x + int(1.7 * rect[2])
            if flag:
                rect.x = rect.x + adjust
                flag = False

            bbrect.x = rect.x
            bbrect.y = rect.y - rect.height
            char_bbs.append(bbrect)

        rect.centery -= (i) * (rect.height + spaceH + curve[mid_idx])
        # "1"比较特殊，后面字符位置需要调整
        rect = pygame.Rect(last_rect)
        char_fact = 1.0
        for i, c in enumerate(display_text[mid_idx + 1:]):

            last_rect.topleft = (last_rect.topright[0] + char_spacing * char_fact, last_rect.topleft[1])
            for ii in range(lineNum):
                # if fs['random_kerning'] and False:
                #   char_fact += fs['random_kerning_amount'] * n.random.randn()
                char_fact = 1.0
                if len(display_text_list[ii]) <= mid_idx + i + 1:
                    # newrect = font.get_rect("0")
                    # newrect.centery += ii * (newrect.height + spaceH + curve[mid_idx - i - 1])
                    continue
                c = display_text_list[ii][mid_idx + i + 1]

                newrect = font.get_rect(c)

                newrect.topleft = last_rect.topleft
                #if c == u"（" or c == u"１":
                 #   newrect.x = newrect.x - int(1.7 * newrect[2])

                newrect.centery += ii * (newrect.height + spaceH + curve[mid_idx + i + 1])
                print(newrect, c)
                if ii == 0:
                    ghgd = u"《"
                    utf=c.encode("utf-8")
                    if adjust_value.has_key(c.encode("utf-8")):  # c == "1"
                        adjust=int(adjust_value[c.encode("utf-8")] * wid)
                        flag = True
                        newrect.x = newrect.x - adjust
                # ??
                try:
                    bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx + i + 1])
                except ValueError:
                    bbrect = font.render_to(bg_surf, newrect, c)
                #if c == u"（" or c == u"１":
                 #   newrect.x = newrect.x + int(1.7 * newrect[2])
                bbrect.x = newrect.x
                bbrect.y = newrect.y - newrect.height
                char_bbs.append(bbrect)
                newrect.centery -= ii * (newrect.height + spaceH + curve[mid_idx + i + 1])

                if ii == 0:
                    if flag :
                        #复位
                        newrect.x = newrect.x + adjust
                        flag = False
                    last_rect = newrect
        # render chars to the left
        last_rect = rect
        #last_rect.topright = (last_rect.topleft[0] - char_spacing * char_fact, last_rect.topleft[1])
        for i, c in enumerate(reversed(display_text[:mid_idx])):
            char_fact = 1.0
            last_rect.topright = (last_rect.topleft[0] - char_spacing * char_fact, last_rect.topleft[1])

            for ii in range(lineNum):
                char_fact = 1.0
                if len(display_text_list[ii]) <= mid_idx - i - 1:
                    # newrect = font.get_rect("0")
                    # newrect.centery += ii * (newrect.height + spaceH + curve[mid_idx - i - 1])
                    continue
                c = display_text_list[ii][mid_idx - i - 1]
                newrect = font.get_rect(c)
                if ii == 0:
                    newrect.topleft = (last_rect.topright[0] - newrect[2], last_rect.topright[1])
                else:
                    newrect.topleft = (last_rect.topleft[0], last_rect.topright[1])

                # newrect.y = last_rect.y
               # if c == u"（" or c == u"１":
                #    newrect.x = newrect.x - int(1.5 * newrect[2])
                newrect.centery += ii * (newrect.height + spaceH + curve[mid_idx - i - 1])
                # ??
                # newrect.centery = max(0 + newrect.height * 1,
                #  min(self.sz[1] - newrect.height * 1, newrect.centery + curve[mid_idx - i - 1]))
                print(newrect, c)
                if ii == 0:
                    if adjust_value.has_key(c.encode("utf-8")):  # c == "1"
                        adjust =int(adjust_value[c.encode("utf-8")] * wid)
                        flag = True
                        newrect.x = newrect.x - adjust
                try:
                    bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx - i - 1])
                except ValueError:
                    bbrect = font.render_to(bg_surf, newrect, c)
                #if c == u"（" or c == u"１":
                 #   newrect.x = newrect.x + int(1.5 * newrect[2])
                if ii == 0:
                    if flag :
                        newrect.x = newrect.x + adjust
                        flag = False
                bbrect.x = newrect.x
                bbrect.y = newrect.y - newrect.height
                char_bbs.append(bbrect)
                newrect.centery -= ii * (newrect.height + spaceH + curve[mid_idx - i - 1])

                last_rect = newrect
            last_rect = newrect
        return bg_surf,char_bbs






    def generate_sample(self, display_text=None,noise=True, outheight=None, pygame_display=False,
                        random_crop=False, substring_crop=-1, char_annotations=False):
        """
        This generates the full text sample
        """

        if self.screen is None and pygame_display:
            self.screen = pygame.display.set_mode(self.sz)
            pygame.display.set_caption('WordRenderer')

        # clear bg
        # bg_surf = pygame.Surface(self.sz, SRCALPHA, 32)
        # bg_surf = bg_surf.convert_alpha()

        if display_text is None:
            # get the text to render

            # abc = "0123456789ABCDEF"
            # idx = n.random.randint(0, len(abc))
            # i = 0
            # listsamp = []
            #
            # while i < 8:
            #     idx = n.random.randint(0, len(abc))
            #     listsamp.append(abc[idx])
            #     i = i + 1
            # sample = "".join(listsamp)




            # print linestr
            # print len(linestr)
            i = 0

            label = 0
            # display_text, label = self.corpus.get_sample(length=display_text_length)
        else:
            label = 0

        # print "generating sample for \"%s\"" % display_text
        #display_text="41250131"
        # get a new font state
        info = self.extraInfo
        if info is None:
            return
        #display_text=u"城东‘区省“区"

        if isinstance(display_text,list):
            testuncodelist = display_text
            testuncode=display_text[0]
            for strr in display_text[1:]:
                testuncode+=" "+strr
        else:
            testuncode=display_text
            testuncodelist=[display_text]

        '''
        display_text = display_text.strip()
        testuncode=display_text
        testuncodelist=[]
        testuncode=testuncode.strip(u"日")
        temp=testuncode.split("年")
        testuncodelist.append(temp[0])
        temp=temp[1].split(u"月")
        testuncodelist.extend(temp)
        '''

        '''
        display_text=list(display_text)
        display_text_list=[]
        temp=[]
        lineChars=info["char_config"]["lineChars"]
        if lineChars==0:
            lineChars=len(display_text)
        for c in display_text:
            temp.append(c)
            if len(temp)>lineChars-1:
                display_text_list.append(temp)
                temp=[]
        if len(temp)>0:
            display_text_list.append(temp)
        '''
        fs = self.fontstate.get_sample()
        #fs['size']=30
        # cl# !/usr/bin/env python
        # # -*- coding:utf-8 -*-
        # # Author: Sqrt5
        #
        #
        # import sys
        # sys.path.append('..')
        # import argparse
        # import math
        # import numpy as np
        # import os
        # import random
        # import time
        # from PIL import Image
        # from extension import create_corpus
        # from extension.hwcaptcha import getInsectionOfCharsAndFonts
        # from extension.hwcaptcha import getPrintCaptcha1,getPrintCaptcha
        # from extension.tools import progressbar
        # from extension.tools import readFromDir
        # from extension.tools import saveImage
        # from skimage import exposure, util
        # import re
        #
        # import json
        # def parse_args():
        #     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='')
        #     parser.add_argument('--save_pic_dir', help='保存图片的目录路径(以/结尾)')
        #     parser.add_argument('--sample', type=int, default=10, help='样本数量')
        #     parser.add_argument('--vertical', type=bool, default=False, help='是否竖向')
        #     parser.add_argument('--fonts_idx', type=int, default=0)
        #     return parser.parse_args()
        # def create_amount_time(sample = 10, inf = 4, sup = 13):
        #     all_chars = u'0123456789'
        #     chars_list = []
        #     from random import *
        #     from time import *
        #     print '正在shengjin...'
        #     for loop in range(60000):
        #         date1 = (2010, 1, 1, 0, 0, 0, -1, -1, -1)
        #         time1 = mktime(date1)
        #         date2 = (2050, 1, 1, 0, 0, 0, -1, -1, -1)
        #         time2 = mktime(date2)
        #         random_time = uniform(time1, time2)  # uniform返回随机实数 time1 <= time < time2
        #         abc = strftime('%Y-%m-%d %H:%M:%S', localtime(random_time))
        #         numtype = randint(0, 2)
        #         if numtype == 0:
        #                   acadgd = abc.split()[0].replace('-', '')
        #         if numtype == 1:
        #                    acadgd = abc.split()[0].replace('-', ' ')
        #         if numtype == 2:
        #               acadgd = aear bg
        # bg_surf = pygame.Surface(self.sz, SRCALPHA, 32)



        # print 'hey inside generate_sample'

        # colour state
        #cs = self.colourstate.get_sample(2 + fs['border'])
        cs = [96,98]
        # print cs

        # print 'hey inside generate_sample222'

        # baseline state


        try:
            interval=info["char_config"]["spaceVer"]
        except Exception:
            interval=[5,5]
        spaceH=random.randint(interval[0],interval[1])
        try:
            char_spacing=info["char_config"]["char_spacing"]
        except Exception:
            print("error:char_spacing could not be found")
            char_spacing=10
        try:
            fontPic=info["fontPic"]
        except Exception:
            fontPic=False
        bg_surf = pygame.Surface((round(2.0 * fs['size'] * len(display_text)), self.sz[1] * 5), SRCALPHA, 32)
        char_bbs = []
        char_bbs.append(bg_surf.get_rect())

        if not fontPic:
            font = freetype.Font(fs['font'], size=fs['size'])

            # random params
            # display_text = fs['capsmode'](display_text) if fs['random_caps'] else display_text

            font.underline = fs['underline']
            font.underline_adjustment = fs['underline_adjustment']
            font.strong = fs['strong']

            font.oblique = fs['oblique']
            font.strength = fs['strength']
            #char_spacing = fs['char_spacing']
            font.antialiased = True
            font.origin = True
            bound_list = []
            lineChars = info["char_config"]["lineChars"]
            for chars in testuncodelist:
                display_text_list = []
                temp = []
                if lineChars == 0:
                    lineChars = len(chars)
                for c in chars:
                    temp.append(c)
                    if len(temp) > lineChars - 1:
                        display_text_list.append(temp)
                        temp = []
                if len(temp) > 0:
                    display_text_list.append(temp)
                bg_surf = pygame.Surface((round(2.0 * fs['size'] * len(display_text)), self.sz[1] * 5), SRCALPHA, 32)

                char_bbs.append(bg_surf.get_rect())
                print(bg_surf.get_size())
                #黑底白字
                bg_surf, char_bbs=self.getPrintCaptcha(font,display_text_list,bg_surf,char_spacing,spaceH,fs['size'],fs["curved"],label=info["label"],adjust_value=info["char_config"]["leftAdjust"])
                bg_arr = self.get_ga_image(bg_surf)  # 0,1（有值）
                bound=bg_arr[..., 1]
                img = Image.fromarray(255-bound)
                #img.show()
                bound_list.append(bound)
        else:
            try :
                upAdjust=info["char_config"]["upAdjust"]
            except Exception:
                upAdjust={}
            try :
                height=info["char_config"]["height"]
            except Exception:
                height=None
            bg_surf=makePic(testuncodelist,self.fontstate.fontPic, font_size=fs['size'], char_spacing=char_spacing,adjust_value=upAdjust,standardH=height)
            bound_list=[]
            for bg in bg_surf:
                # 3通道
                bg_arr = 255 - n.array(bg)
                bound = rgb2gray(bg_arr)
                bound = bound.astype(n.uint8)
                bound_list.append(bound)

        #curve = [last_rect[3] + spaceH] * len(curve) + curve

        # show
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # self.screen.fill((255,255,255))
        # self.screen.blit(bg_surf, (0,0))
        # # for bb in char_bbs:
        # #     pygame.draw.rect(self.screen, (255,0,0), bb, 2)
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/0.jpg')
        # wait_key()


        bb = pygame.Rect(self.get_bb(bound))
        #bg_arr = self.get_ga_image_p(bg_surf)
        # colour text
        #bg_arr[..., 0] = cs[0]

        # # do elastic distortion described by http://research.microsoft.com/pubs/68920/icdar03.pdf
        # dispx, dispy = self.elasticstate.sample_transformation(bg_arr[...,0].shape)
        # bg_arr[...,1] = self.apply_distortion_maps(bg_arr[...,1], dispx, dispy)
        #
        # # show
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # canvas = 255*n.ones(bg_arr.shape)
        # globalcanvas = grey_blit(bg_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # wait_key()

        noise = info["noise_config"]["isNoise"]
        try:
            value = info["bold"]
        except Exception:
            value = 0.8
        try:
            red = info["red"]
        except Exception:
            red = False
        try:
            keeplabel = info["keep_label"]
        except Exception:
            keeplabel = True
        if not noise:
            for i,bound in enumerate(bound_list):
                bb = pygame.Rect(self.get_bb(bound))
                bound = self.imcrop(bound, bb)
                img=Image.fromarray(bound)
                #img.show()
                bound = bound *value
                bound_list[i]=bound

            #bg_arr = exposure.adjust_gamma(image=bg_arr, gamma=math.exp(5))
            bound = self.add_fillimage_p(bound_list,info["output_config"])


            if bound.ndim<3:
                rgb_canvas = self.stack_arr((bound, bound, bound))
            else:
                rgb_canvas=bound
            #canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0, 1))
            # for char_bb in char_bbs:
            #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
            #dgfs=bg_surf.get_size()
            #self.screen = pygame.display.set_mode(canvas_surf.get_size())
            #self.screen.blit(canvas_surf, (0, 0))
            #pygame.display.flip()

            print testuncode
            return {'image': bound, 'text': testuncode, 'label': label,
                    'chars': n.array([[c.x, c.y, c.width, c.height] for c in char_bbs])}
        flag = True
        try:
            if info["noise_config"]["useOriBck"]:
                flag = True
        except Exception:
            pass
        angle = info["noise_config"]["rotateAngle"]
        random_angle = random.uniform(-angle, angle)
        for i, bound in enumerate(bound_list):

            #img=Image.fromarray(bound)
            pts31 = self.get_rect(bound.shape[1], bound.shape[0],random_angle)
            bound = self.apply_perspective_arr(bound,pts31)
            bb = pygame.Rect(self.get_bb(bound))
            bound = self.imcrop(bound, bb)
            bound = bound * value
            bound_list[i] = bound
            img = Image.fromarray(bound)
            #img.show()
            pass
        if flag:
            l1_arr = self.add_fillimage_p(bound_list, info["output_config"])
            l1_arr = l1_arr[..., 1]
        # border/shadow
        '''
        if fs['border']:
            l1_arr, l2_arr = self.get_bordershadow(bg_arr, cs[2])
        else:
            #灰度图
            l1_arr = bound

        # show individiual layers (fore, bord, back)
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l2_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/1.jpg')
        # wait_key()
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l1_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/2.jpg')
        # wait_key()
        # self.screen.fill((cs[1],cs[1],cs[1]))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/3.jpg')
        # wait_key()

        # show
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # canvas[...,0] = cs[1]
        # globalcanvas = grey_blit(l1_arr, canvas)
        # if fs['border']:
        #     globalcanvas = grey_blit(l2_arr, globalcanvas)
        # globalcanvas = globalcanvas[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
        # # for char_bb in char_bbs:
        # #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
        # self.screen = pygame.display.set_mode(canvas_surf.get_size())
        # self.screen.blit(canvas_surf, (0, 0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/4.jpg')
        # wait_key()

        # do rotation and perspective distortion
        temp_bb= pygame.Rect(self.get_bb(l1_arr))
        if temp_bb[2]>270 or temp_bb[2]<230 or temp_bb[3]<62 or temp_bb[3]>72:
            pass
            #return None
        affstate = self.affinestate.sample_transformation(l1_arr.shape)
        perstate = self.perspectivestate.sample_transformation(l1_arr.shape)
        l1_arr = self.apply_perspective_arr(l1_arr, affstate, perstate)
        if fs['border']:
             l2_arr[..., 1] = self.apply_perspective_arr(l2_arr[..., 1], affstate, perstate)
        if char_annotations:
             char_bbs = self.apply_perspective_rectim(char_bbs, l1_arr[..., 1], affstate, perstate)
             # order char_bbs by left to right
             xvals = [bb.x for bb in char_bbs]
             idx = [i[0] for i in sorted(enumerate(xvals), key=lambda x: x[1])]
             char_bbs = [char_bbs[i] for i in idx]

        if n.random.rand() < substring_crop and len(display_text) > 4 and char_annotations:
            # randomly crop to just a sub-string of the word
            start = n.random.randint(0, len(display_text) - 1)
            stop = n.random.randint(min(start + 1, len(display_text)), len(display_text))
            display_text = display_text[start:stop]
            char_bbs = char_bbs[start:stop]
            # get new bb of image
            bb = pygame.Rect(self.get_rects_union_bb(char_bbs, l1_arr))
        else:
            # get bb of text
            if fs['border']:
                bb = pygame.Rect(self.get_bb(grey_blit(l2_arr, l1_arr)[..., 1]))
            else:
                bb = pygame.Rect(self.get_bb(l1_arr))
        if random_crop:
            #bb.inflate_ip(10 * n.random.randn() + 15, 10 * n.random.randn() + 15)
            pass
        else:
            inflate_amount = int(0.4 * bb[3])
            bb.inflate_ip(inflate_amount, inflate_amount)

        # crop image
        #bb[2]=2*bb[2]
        #bb[3] = 2 * bb[3]

        import numpy as np
        a1 = l1_arr[..., 1]

        l1_arr = self.imcrop(l1_arr, bb)
        l1_arr = l1_arr * value
        flag=False
        try:
            if info["noise_config"]["useOriBck"]:
                flag=True
        except Exception:
            pass
        if flag:
            l1_arr = self.add_fillimage_p(l1_arr, info["output_config"])
            l1_arr = l1_arr[..., 1]
        '''
        #
        #gray=rgb2gray(l1_arr)
        import numpy as np
        #3通道图
        gray=l1_arr
        gray = gray.reshape((gray.shape[0], gray.shape[1], 1))

        #2通道
        l1_arr=n.concatenate((n.zeros(gray.shape)+np.random.randint(150,160),gray),axis=2)
        if fs['border']:
            l1_arr, l2_arr = self.get_bordershadow(bg_arr, cs[2])
            l2_arr = self.imcrop(l2_arr, bb)
        if char_annotations:
            # adjust char bbs
            for char_bb in char_bbs:
                char_bb.move_ip(-bb.x, -bb.y)
        canvas = (255 * n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        canvas[..., 0] = cs[1]

        # show
        # globalcanvas = grey_blit(l1_arr, canvas)
        # if fs['border']:
        #     globalcanvas = grey_blit(l2_arr, globalcanvas)
        # globalcanvas = globalcanvas[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
        # # for char_bb in char_bbs:
        # #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
        # self.screen = pygame.display.set_mode(canvas_surf.get_size())
        # self.screen.blit(canvas_surf, (0, 0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/5.jpg')
        # wait_key()


        # add in natural images
        #self.fillimstate.isRandom=True
        try:
            if True:
                #随机背景
                canvas = self.add_fillimage(canvas)
            else:
                #canvas = self.add_fillimage_p(canvas)
                pass


            #l1_arr = self.add_fillimage(l1_arr)
            if fs['border']:
                l2_arr = self.add_fillimage(l2_arr)
        except Exception:
            print
            "\tfillimage error"
            return None

        # add per-surface distortions
        #l1_arr = self.surface_distortions(l1_arr)
        if fs['border']:
            l2_arr = self.surface_distortions(l2_arr)

        # compose global image
        blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
        count = 0
        try:
            convertGray=info["gray"]
        except Exception:
            convertGray=True

        while True:
            globalcanvas = grey_blit(l1_arr, canvas, blend_mode=random.choice(blend_modes))
            #globalcanvas = grey_blit(l1_arr, canvas, blend_mode=MJBLEND_SCREEN)
            if fs['border']:
                globalcanvas = grey_blit(l2_arr, globalcanvas, blend_mode=random.choice(blend_modes))
            if convertGray:
                globalcanvas = globalcanvas[..., 0]
                #globalcanvas = rgb2gray(globalcanvas)
            std = n.std(globalcanvas.flatten())
            count += 1
            # print count
            if std > 20:
                break
            if count > 10:
                print
                "\tcan't get good contrast"
                return None
        #canvas=canvas.astype('uint8')
        #canvas = canvas[..., 2]
        canvas = globalcanvas

        # do elastic distortion described by http://research.microsoft.com/pubs/68920/icdar03.pdf
        # dispx, dispy = self.elasticstate.sample_transformation(canvas.shape)
        # canvas = self.apply_distortion_maps(canvas, dispx, dispy)

        #canvas = resize_image(canvas, newh=64, neww=270)
        # add global distortions
        canvas = self.global_distortions(canvas)

        # noise removal
        if convertGray :
            filterSize=(3,3)
        else:
            filterSize=(3,3,canvas.shape[2])
        canvas = ndimage.filters.median_filter(canvas, size=filterSize)
        # resize
        #outheight=64
        outheight=None
        if outheight is not None:
            if char_annotations:
                char_bbs = self.resize_rects(char_bbs, canvas, outheight)
            canvas = resize_image(canvas, newh=outheight,neww=270)

        # FINISHED, SHOW ME SOMETHING
        pygame_display=False
        canvas=canvas.astype(int)
        if pygame_display:
            if canvas.ndim<3:
                rgb_canvas = self.stack_arr((canvas, canvas, canvas))
            else :
                rgb_canvas=canvas
            #rgb_canvas=rgb_canvas[...,1]
            canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0, 1))
            # for char_bb in char_bbs:
            #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
            #dgfs=bg_surf.get_size()
            self.screen = pygame.display.set_mode(canvas_surf.get_size())
            self.screen.blit(canvas_surf, (0, 0))
            pygame.display.flip()
        print "--------------------"

        # pyplot.imshow(self.get_image())
        # pyplot.show()

        # print char_bbs[0]
        aa1=canvas[...,0]
        aa2 = canvas[..., 1]
        return {'image': canvas, 'text': testuncode, 'label': label,
            'chars': n.array([[c.x, c.y, c.width, c.height] for c in char_bbs])}


if __name__ == "__main__":

    fillimstate = SVTFillImageState("/Users/jaderberg/Data/TextSpotting/DataDump/svt1",
                                    "/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat")
    fs = FontState()
    # fs.border = 1.0

    # corpus = SVTCorpus({'unk_probability': 0.5})
    corpus = RandomCorpus({'min_length': 1, 'max_length': 10})
    WR = WordRenderer(fontstate=fs, fillimstate=fillimstate, colourstate=TrainingCharsColourState, corpus=corpus)
    while True:
        data = WR.generate_sample(pygame_display=True, substring_crop=0, random_crop=True, char_annotations=True)
        if data is not None:
            print
            data['text']
        # save_screen_img(WR.screen, '/Users/jaderberg/Desktop/6.jpg', 70)
        wait_key()

        # WR = WordRenderer(fontstate=fs, fillimstate=fillimstate, corpus=Corpus, colourstate=TrainingCharsColourState)
        # towrite = "Worcester College Rocks".split()
        # perrow = 4.0
        # rows = math.ceil(len(towrite)/perrow)
        # cols = int(perrow)
        # num = len(towrite)
        # for i, w in enumerate(towrite):
        #     pyplot.subplot(rows, cols, i+1)
        #     pyplot.imshow(WR.generate_sample(display_text=w, outheight=32)['image'], cmap=cm.Greys_r)
        #     pyplot.axis('off')
        # pyplot.tight_layout()
        # pyplot.show()
