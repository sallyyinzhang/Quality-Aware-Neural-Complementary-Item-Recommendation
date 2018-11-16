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

from contextlib import closing
import shelve
from operator import itemgetter 

stdEm = 0.01
stdEt = 0.1
stdW = 0.1
stdE = 0.01


class EncoreCell:
    """
    complementary recommendation

    @inproceedings{zhang2018quality,
      title={Quality-aware neural complementary item recommendation},
      author={Zhang, Yin and Lu, Haokai and Niu, Wei and Caverlee, James},
      booktitle={Proceedings of the 12th ACM Conference on Recommender Systems},
      pages={77--85},
      year={2018},
      organization={ACM}
    }

    """

    def __init__(self, ImageDim, TexDim, ImageEmbDim, TexEmDim, HidDim, FinalDim):

        self.ImageDim = ImageDim
        self.TexDim = TexDim
        self.ImageEmbDim = ImageEmbDim
        self.TexEmDim = TexEmDim
        self.HidDim = HidDim
        self.FinalDim = FinalDim


        self.xi = tf.placeholder(tf.float32, shape=(None, ImageDim)) #image info
        self.xj = tf.placeholder(tf.float32, shape=(None, ImageDim))
        self.tti = tf.placeholder(tf.float32, shape=(None, TexDim))
        self.ttj = tf.placeholder(tf.float32, shape=(None, TexDim))
        self.rscore = tf.placeholder(tf.float32, shape=(None, 1))



        self.y = tf.placeholder(tf.float32)

        
        self.Em = tf.Variable(tf.truncated_normal([self.ImageDim, self.ImageEmbDim],stddev=stdEm))
    
        self.Et = tf.Variable(tf.truncated_normal([self.TexDim, self.TexEmDim],stddev=stdEt))

        self.W = tf.Variable(tf.truncated_normal([(self.ImageEmbDim + self.TexEmDim + 1), HidDim],stddev=stdW))

        self.E = tf.Variable(tf.truncated_normal([self.HidDim, self.FinalDim],stddev=stdE))


        self.b1 = tf.Variable(tf.zeros([1,self.HidDim]), dtype=tf.float32)
        self.c = tf.Variable([0.0], dtype = tf.float32)
        #c = tf.Variable(tf.random_normal([1], stddev = 0.1))


    def train(self):
        dm = tf.matmul((self.xi - self.xj),self.Em)
        dt = tf.matmul((self.tti - self.ttj),self.Et)
        tempdis = tf.concat([dm, dt], 1)

        NewVec = tf.reshape(tf.concat([tempdis, self.rscore], 1), [1,(self.ImageEmbDim + self.TexEmDim + 1)])
        #d = rho2 * tf.reduce_sum(tf.square(tf.matmul((xi - xj),Em))) + rho * tf.reduce_sum(tf.square(tf.matmul((tti - ttj),Et)))
        #thresh = tf.reduce_sum(tf.square(tf.matmul((xi - xj),Em))) - c
        #d = tf.norm(tf.matmul((xi - xj), Em))
        dij = tf.matmul(NewVec, self.W) + self.b1
        dtem = tf.reduce_sum(tf.square(tf.matmul(tf.tanh(dij), self.E)))
        thresh = dtem - self.c
        sigma = 1/(1 + tf.exp(thresh))

        loss = -tf.reduce_sum(self.y*tf.log(sigma) + (1 - self.y)*tf.log((1 - sigma)))

        return loss, thresh

    def predict_ratings(self):
        dm = tf.matmul((self.xi - self.xj),self.Em)
        dt = tf.matmul((self.tti - self.ttj),self.Et)
        tempdis = tf.concat([dm, dt], 1)

        NewVec = tf.reshape(tf.concat([tempdis, self.rscore], 1), [1,(self.ImageEmbDim + self.TexEmDim + 1)])
        #d = rho2 * tf.reduce_sum(tf.square(tf.matmul((xi - xj),Em))) + rho * tf.reduce_sum(tf.square(tf.matmul((tti - ttj),Et)))
        #thresh = tf.reduce_sum(tf.square(tf.matmul((xi - xj),Em))) - c
        #d = tf.norm(tf.matmul((xi - xj), Em))
        dij = tf.matmul(NewVec, self.W) + self.b1
        dtem = tf.reduce_sum(tf.square(tf.matmul(tf.tanh(dij), self.E)))
        thresh = dtem - self.c


        pos_acc = tf.less(thresh, 0)
        neg_acc = tf.greater_equal(thresh, 0)
        acc = self.y*tf.cast(pos_acc, tf.float32) + (1 - self.y)*tf.cast(neg_acc, tf.float32)

        return acc


