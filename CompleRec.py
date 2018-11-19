import tensorflow as tf
import pandas as pd
import gzip
import struct
import json
from pprint import pprint
import numpy as np
import pickle
import re, math
from collections import Counter
import json
import random
from itertools import permutations, islice
import argparse
import scipy.sparse

from contextlib import closing
import shelve
from operator import itemgetter 
import Encore
from sklearn.model_selection import KFold


def parase_args():
    parser = argparse.ArgumentParser(description="Run Encore.")
    parser.add_argument('--ImageDim', type=int, default=4096,
                        help='Image Dimension.')
    parser.add_argument('--TexDim', type=int, default=100,
                        help='Text Dimension.')
    parser.add_argument('--ImageEmbDim', type=int, default=10,
                        help='Image Embedding Dimension.')
    parser.add_argument('--TexEmDim', type=int, default=10,
                        help='Text Embedding Dimension.')
    parser.add_argument('--HidDim', type=int, default=100,
                        help='Hidden Dimension.')
    parser.add_argument('--FinalDim', type=int, default=10,
                        help='Fainl Dimension.')
    parser.add_argument('--learningrate', type=float, default=0.0001,
                        help='Learning Rate.')
    parser.add_argument('--trainchoice', nargs='?', default="Yes",
                        help='Training or Testing.')

    return parser.parse_args()



def main():
    args = parase_args()
    ImageDim, TexDim, ImageEmbDim, TexEmDim, HidDim, FinalDim, learningrate, trainchoice = args.ImageDim, args.TexDim, args.ImageEmbDim, args.TexEmDim, args.HidDim, args.FinalDim, args.learningrate, args.trainchoice
    

    with tf.device('/gpu:0'):
        encore = Encore.EncoreCell(ImageDim, TexDim, ImageEmbDim, TexEmDim, HidDim, FinalDim)

        encoreloss, thresh = encore.train()
        testacc = encore.predict_ratings()


        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        #optimizer = tf.train.AdamOptimizer(learning_rate = learningrate)
        #print("AdadeltaOptimizer")
        train_op = optimizer.minimize(encoreloss)


        # Start training
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)


        #import data
        #AlsoBoughtRelationDic: {"bought_together":{item: [list of also bought items/bought together items]}}
        #AlsoBoughtInfoDic{item:[textual word count with smallercase, textual word count, text vector]}
        with open('BoughtTogetherDic.pickle') as f:
            AlsoBoughtRelationDic,AlsoBoughtInfoDic = pickle.load(f)

        #TextVecDic: {item: [learned word vector]}
        with open('TextVecDic.pickle') as f:
            TextVecDic = pickle.load(f)

        #learned Bayesian score
        with open('RatingScore.pickle') as f:
            dscore = pickle.load(f)
        
        #image feature
        imagefeature = shelve.open('ImageBoughtTogetherDic.shelf')
        
        #train, test data
        with open("datakeys.pickle") as f:
            SubTraingKeys, Testkeys = pickle.load(f)



        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        if trainchoice == "Yes":  #five cross to train data
            print("train")
            kf = KFold(n_splits=5)
            
            result_acc = []

            for train_index, vali_index in kf.split(SubTraingKeys): #cross validation
                Trainkeys, Valikeys = np.array(SubTraingKeys)[train_index], np.array(SubTraingKeys)[vali_index]

                runtimes = 10
                for _ in range(runtimes):  #training times
                    random.shuffle(Trainkeys)
                    for key in Trainkeys:
                        print(key)
                        try:
                            mi = np.transpose(imagefeature[key])
                            ti = TextVecDic[key].reshape((1,TexDim))
                            #print("try")
                        except:
                            continue
                        
                        #ti = np.transpose(TextVecDic[key])
                        checklist = AlsoBoughtRelationDic['bought_together'][key]
                        ichecklist = 0
                        
                        for it in checklist:
                            print("test checklist is", it)
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                print("begin")
                                #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:1})
                                _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                                                        encore.xj:mj, 
                                                                                        encore.tti:ti, 
                                                                                        encore.ttj:tj, 
                                                                                        encore.rscore:score, 
                                                                                        encore.y:1})
                                print('distance1 = %f', distance)
                                ichecklist = ichecklist + 1

                            except:
                                continue

                        flaglen = len(checklist)

                        NotRelationQ = set(AlsoBoughtInfoDic.keys()) - set(checklist)  #it can be changed !!!!!!!!!!!!!!!!!!!!!!!

                        Qlist = list(NotRelationQ)

                        Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                        Qchecklist = list(itemgetter(*Qindex)(Qlist))
                        
                        ical = 0
                        
                        for it in Qchecklist:
                            if ical == ichecklist:
                                break
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:0})
                                _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                                encore.xj: mj, 
                                                                encore.tti: ti, 
                                                                encore.ttj:tj, 
                                                                encore.rscore: score, 
                                                                encore.y:0})

                                print('distance2 = %f', distance)
                                ical = ical + 1
                                #print("run2")
                            except:
                                continue


                

                print("begin test")
                for key in Valikeys[:2]:
                        print(key)
                        try:
                           mi = np.transpose(imagefeature[key])
                           ti = TextVecDic[key].reshape((1,TexDim))
                        except:
                           continue
                        #ti = TextVecDic[key]
                        checklist = AlsoBoughtRelationDic['bought_together'][key]
                        ichecklist = 0

                        for it in checklist:
                            print("checklist is ", it)
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                                                      encore.xj: mj, 
                                                                                      encore.tti: ti, 
                                                                                      encore.ttj:tj, 
                                                                                      encore.rscore: score, 
                                                                                      encore.y:1})

                                print("test1 thresh is", test_thresh )
                                result_acc.append(cal_acc)
                                ichecklist = ichecklist + 1
                                #print("runtest")
                            except:
                                continue

                        flaglen = len(checklist)

                        NotRelationQ = set(AlsoBoughtInfoDic.keys()) - set(checklist)

                        Qlist = list(NotRelationQ)

                        Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                        Qchecklist = list(itemgetter(*Qindex)(Qlist))
                        
                        ical = 0

                        for it in Qchecklist:
                            if ical == ichecklist:
                                break
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                              encore.xj: mj, 
                                                              encore.tti: ti, 
                                                              encore.ttj:tj, 
                                                              encore.rscore: score, 
                                                              encore.y:0})


                                print("test2 thresh is", test_thresh )
                                result_acc.append(cal_acc)
                                ical = ical + 1
                                #print("runText2")
                            except:
                                continue


            print("TexDim is %d" % TexDim)
            print("acc is")
            print(np.mean(result_acc))
            print("learning rate is")
            print(learningrate)
            print("runtime is %d"%runtimes)


        
        else:
            runtimes = 10
            for _ in range(runtimes):  #training times
                random.shuffle(SubTraingKeys)
                for key in SubTraingKeys[:10]:
                    print(key)
                    try:
                        mi = np.transpose(imagefeature[key])
                        ti = TextVecDic[key].reshape((1,TexDim))
                        #print("try")
                    except:
                        continue
                    
                    #ti = np.transpose(TextVecDic[key])
                    checklist = AlsoBoughtRelationDic['bought_together'][key]
                    ichecklist = 0
                    
                    for it in checklist:
                        print("test checklist is", it)
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            print("begin")
                            #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:1})
                            _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                                                    encore.xj: mj, 
                                                                                    encore.tti: ti, 
                                                                                    encore.ttj:tj, 
                                                                                    encore.rscore: score, 
                                                                                    encore.y:1})
                            print('distance1 = %f', distance)
                            ichecklist = ichecklist + 1
                            #tf.Print(sigma, [sigma], message="sigma is:")
                            #print("run")
                        except:
                            continue

                    flaglen = len(checklist)

                    NotRelationQ = set(AlsoBoughtInfoDic.keys()) - set(checklist)  #it can be changed !!!!!!!!!!!!!!!!!!!!!!!

                    Qlist = list(NotRelationQ)

                    Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                    Qchecklist = list(itemgetter(*Qindex)(Qlist))
                    
                    ical = 0
                    
                    for it in Qchecklist[:2]:
                        if ical == ichecklist:
                            break
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:0})
                            _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                            encore.xj: mj, 
                                                            encore.tti: ti, 
                                                            encore.ttj:tj, 
                                                            encore.rscore: score, 
                                                            encore.y:0})

                            print('distance2 = %f', distance)
                            ical = ical + 1
                            #print("run2")
                        except:
                            continue


            result_acc = []

            print("begin test")
            for key in Testkeys:
                    print(key)
                    try:
                       mi = np.transpose(imagefeature[key])
                       ti = TextVecDic[key].reshape((1,TexDim))
                    except:
                       continue
                    #ti = TextVecDic[key]
                    checklist = AlsoBoughtRelationDic['bought_together'][key]
                    ichecklist = 0

                    for it in checklist:
                        print("checklist is ", it)
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                                                  encore.xj: mj, 
                                                                                  encore.tti: ti, 
                                                                                  encore.ttj:tj, 
                                                                                  encore.rscore: score, 
                                                                                  encore.y:1})

                            print("test1 thresh is", test_thresh )
                            result_acc.append(cal_acc)
                            ichecklist = ichecklist + 1
                            #print("runtest")
                        except:
                            continue

                    flaglen = len(checklist)

                    NotRelationQ = set(AlsoBoughtInfoDic.keys()) - set(checklist)

                    Qlist = list(NotRelationQ)

                    Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                    Qchecklist = list(itemgetter(*Qindex)(Qlist))
                    
                    ical = 0

                    for it in Qchecklist:
                        if ical == ichecklist:
                            break
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                          encore.xj: mj, 
                                                          encore.tti: ti, 
                                                          encore.ttj:tj, 
                                                          encore.rscore: score, 
                                                          encore.y:0})


                            print("test2 thresh is", test_thresh )
                            result_acc.append(cal_acc)
                            ical = ical + 1
                            #print("runText2")
                        except:
                            continue


            print("TexDim is %d" % TexDim)
            print("acc is")
            print(np.mean(result_acc))
            print("learning rate is")
            print(learningrate)
            print("runtime is %d"%runtimes)



        with open('EncoreParameter.pickle', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([sess.run(encore.Em), sess.run(encore.Et), sess.run(encore.W), sess.run(encore.E), sess.run(encore.b1), sess.run(encore.c)], f)



if __name__ == "__main__":
    main()
