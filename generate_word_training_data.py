#a Max Jaderberg 16/5/14
# -*- coding:utf-8 -*-
# Generates training data using WordRenderer
import sys
import os
import shutil
from titan_utils import is_cluster, get_task_id, crange
from generate_chars import CharsGenerator
from word_renderer import WordRenderer, FontState, FileCorpus, TrainingCharsColourState, SVTFillImageState, wait_key, NgramCorpus, RandomCorpus
from scipy.io import savemat
from PIL import Image
import numpy as n
import tarfile
import h5py
import pandas as pd
import lmdb
import cv2
import argparse
import json

reload(sys)
sys.setdefaultencoding( "utf-8" )
def parse_args(args):
    parser = argparse.ArgumentParser(prog="GameServer")
    parser.add_argument('configfile', nargs=1,type=str, help='')
    parser.add_argument('--startNum', default=0, type=int, help='')
    parser.add_argument('--endNum', default=1, type=int, help='')
    #parser.add_argument('--save_pic_dir', default="/home/user/", type=str, help='')
    return parser.parse_args(args)

def parse(filename):
    configfile = open(filename)
    jsonconfig = json.load(configfile)
    configfile.close()
    return jsonconfig






SETTINGS = {
    #####################################
    'RAND10': {
        'corpus_class': RandomCorpus,
        'corpus_args': {'min_length': 1, 'max_length': 10},
        'fontstate':{
            'font_list': ["/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/font_path_list_ch.txt",
                      "/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/font_path_list_ch.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/home/user/codedd/Synthetic_Data_Engine_For_Text_Recognition/SVT/icdar_2003_train.txt",
                             "/home/user/codedd/Synthetic_Data_Engine_For_Text_Recognition/SVT/icdar_2003_train.txt"],
        'fillimstate': {
            'data_dir': ["/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/img",
                         "/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/img"],
            'gtmat_fn': ["/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt",
                         "/home/user/wxb/GEN_DATA/czt/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt"],
        }
    },
    #####################################
    'RAND23': {
        'corpus_class': RandomCorpus,
        'corpus_args': {'min_length': 1, 'max_length': 23},
        'fontstate':{
            'font_list': ["/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/font_path_list_ch.txt",
                      "/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/font_path_list_ch.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/icdar_2003_train.txt",
                             "/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/icdar_2003_train.txt"],
        'fillimstate': {
            'data_dir': ["/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/img",
                         "/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt/svt1/img"],
            'gtmat_fn': ["/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt",
                         "/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt"],
        }
    },
}


#------------GET Labels--------------------------------------------------

def get_labels(input_list):
    out_list=[]
    
    for x in input_list:
        names=os.path.basename(x)
        res=names.partition('_')[2].partition('_')[0]
        out_list.append(res)
        
    return out_list


#---------------CREATING LMDB DATASET--------------------------------------------
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = n.fromstring(imageBin, dtype=n.uint8)

    if imageBuf.size==0:
        return False

    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
            #txn.put(k.encode(),v.encode())


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

#---------------------CREATING TRAINING DATA-----------

def create_synthetic_data(lmdb_path,imfolder_path,dataset,NUM_TO_GENERATE,lmdb_path_pre,fileindex,info,label):

    #NUM_PER_FOLDER = 10 #1000
    SAMPLE_HEIGHT = 32
    QUALITY = [80, 10]

    iscluster = int(is_cluster())

    settings = SETTINGS[dataset]


    if not os.path.exists(imfolder_path):
    	os.makedirs(imfolder_path)


    ngram_mode = settings.get('ngram_mode', False)

    # init providers
    if 'corpus_class' in settings:
        corp_class = settings['corpus_class']
    else:
        corp_class = FileCorpus
    if 'corpus_args' in settings:
        corpus = corp_class(settings['corpus_args'])
    else:
        corpus = corp_class()
    
    fontPath=info["char_config"]["FontDir"]
    if len(sys.argv) == 4 and False:
         fontabc = sys.argv[3]
         fontstate = FontState(font_list = fontabc )
         fontstate.random_caps = 1
         print "wangxiaobo"
    else:
        try:
            fontPic = info["fontPic"]
        except Exception:
            fontPic = False
        #字体路径
        fontstate = FontState(path=fontPath,fontSize=info["char_config"]["FontSize"],isRandom=info["noise_config"]["isNoise"],fontPic=fontPic)
        fontstate.random_caps = settings['fontstate']['random_caps']
    colourstate = TrainingCharsColourState(info["trainingchars_fn"])
    if not isinstance(settings['fillimstate'], list):
        #背景

        fillimstate = SVTFillImageState(info["labelBackgdir"],info["noise_config"]["randomBackgdir"],info["noise_config"]["isNoise"])
    else:
        # its a list of different fillimstates to combine
        states = []
        for i, fs in enumerate(settings['fillimstate']):
            s = SVTFillImageState(fs['data_dir'][iscluster], fs['gtmat_fn'][iscluster])
            # move datadir to imlist
            s.IMLIST = [os.path.join(s.DATA_DIR, l) for l in s.IMLIST]
            states.append(s)
        fillimstate = states.pop()
        for fs in states:
            fillimstate.IMLIST.extend(fs.IMLIST)

    # take substrings
    try:
        substr_crop = settings['substrings']
    except KeyError:
        substr_crop = -1

    # init renderer
    sz = (800,200)
    WR = WordRenderer(sz=sz, corpus=corpus, fontstate=fontstate, colourstate=colourstate, fillimstate=fillimstate,info=info)

    count=0

    #Declating the Image_Name List and Label List
    im_list=[]
    label_list=[]
    generator=getattr(CharsGenerator(),info["Generator"])

    i = 0
   # filesaveinfo = open( lmdb_path_pre + "EngSynthesisSample_" +str(fileindex) , "w")
    for  display_text1 in generator(info["char_config"]["charDir"]):
        if i > NUM_TO_GENERATE:
            break

    # for i in crange(range(0, NUM_TO_GENERATE)):
        print 'Creating Image :', count
        # gen sample
        #data = WR.generate_sample(display_text=display_text1, lineChars=info["lineChars"], outheight=SAMPLE_HEIGHT,
        #                          random_crop=True, substring_crop=substr_crop, char_annotations=(substr_crop > 0))
        try:
            data = WR.generate_sample(display_text=display_text1,outheight=SAMPLE_HEIGHT, random_crop=True, substring_crop=substr_crop, char_annotations=(substr_crop>0))
        except Exception:
            # print "\tERROR","wangxiaobo"
            continue

        if data is None:
            print "\tcould not generate good sample"
            continue

        if not ngram_mode:
            fnstart = "%s_%s_%d" % ('synthetic', data['text'], data['label'])
        else:
            fnstart = "%s_%s_%d" % ('synthetic', data['text'], data['label']['word_label'])

        # save with random compression
        quality = min(80, max(0, int(QUALITY[1]*n.random.randn() + QUALITY[0])))

        try:
            arr = data['image']
            import numpy as np
            arr=arr.astype(np.uint8)
            print(arr.shape)
            #rr1 = np.max(arr[..., 0])
            #rr2 = np.max(arr[..., 1])
            #rr3 = np.max(arr[..., 2])
            img = Image.fromarray(arr)
        except Exception:
            print "\tbad image generated"
            continue

        if img.mode != 'RGB':
            img = img.convert('RGB')
 #       imfn = os.path.join(imfolder_path, fnstart + ".jpg")

        print imfolder_path+str(count)+'.jpg'
        try:
           
           img.save(imfolder_path+str(count)+'.jpg',quality=quality)
        except Exception:
           print

           continue
        print 'Creating Image :', count,"complete:",float(fileindex+0.00005*i)/10
        filesaveinfo = open( lmdb_path_pre + "EngSynthesisSample_" +str(fileindex) , "a+")
        if label=="￥" or label=="tax1" or (not info["output_config"]["keep_label"]):
            label=" "
        filesaveinfo.write(str(fileindex) +"/" +str(count) + ".jpg" + " " +label.decode("utf-8" )+" "+data['text'] + "\t\n")
        filesaveinfo.close() 
        # Save Data for LMDB
        im_list.append(str(count)+'.jpg')
        label_list.append(data['text'])

        count=count+1
        i = i + 1

    #Saving the Dataframe
    im_list = [imfolder_path+x for x in im_list]

    print 'Length of Image Path List: ', len(im_list)
    print 'Length of Image Label List: ', len(label_list)

    print im_list[0]
    print type(im_list[0])
    print label_list
    print type(label_list[0])
    #filesaveinfo.close()
    #df_synthetic=pd.DataFrame(columns=['Image_Path','Image_Label'])
    #df_synthetic['Image_Path']=im_list
    #df_synthetic['Image_Label']=label_list
    #df_synthetic.to_csv('Synthetic_data_info.csv',sep='\t',index=None)

    #Creating LMDB Dataset using create_dataset function

    print 'Creating LMDB Dataset'
    #createDataset(lmdb_path, im_list, label_list, lexiconList=None, checkValid=True)

    print 'Finished creating LMDB Synthetic_Data_Engine_For_Text_Recognition'

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input
def main(argv):
    args = parse_args(argv[1:])
    config = parse(args.configfile[0])
    config=byteify(config)
    #className=args.className
    #label=args.label
    info=config
    if info is None:
        return

    label=info["label"]
    
    if len(sys.argv) >1 or len(sys.argv) == 4 :
        i = int(args.startNum)
        MAXI = int(args.endNum)
    else:
        i = info["startNum"]
        MAXI = info["endNum"]
    print "-------------gd"
    print MAXI

    train_im_folder_path=info["output_config"]["output_dir"]
    val_im_folder_path='/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/text-renderer/vgg_synthetic_custom_val/'

	#Setting LMDB Folder Path
    train_lmdb_path='/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/text-renderer/synth90k_custom_train_lmdb'
    val_lmdb_path='/home/user/wxb/syn/Synthetic_Data_Engine_For_Text_Recognition/text-renderer/synth90k_custom_val_lmdb'

	#Number of Training and Val Images to Generate
    NUM_TO_GENERATE_TRAIN = 10000
    NUM_TO_GENERATE_VAL = 60000



	#Type of Data to Generate
    dataset_type='RAND10'

	#Creating the Training Data
    print 'NUM_TO_GENERATE_TRAIN',NUM_TO_GENERATE_TRAIN

    while i < MAXI:
        train_lmdb_path_1 = train_im_folder_path + str(i)+'/'
        train_im_folder_path_1 = train_im_folder_path + str(i)+'/'
        create_synthetic_data(train_lmdb_path_1,train_im_folder_path_1,dataset_type,NUM_TO_GENERATE_TRAIN,train_im_folder_path,i,info,label)
        i = i + 1
	#Creating the Validation Data
	# print 'NUM_TO_GENERATE_VAL',NUM_TO_GENERATE_VAL
	# create_synthetic_data(val_lmdb_path,val_im_folder_path,dataset_type,NUM_TO_GENERATE_VAL)

	print "FINISHED! Creating Training and Validation Data"


if __name__ == '__main__':
	main(sys.argv)

