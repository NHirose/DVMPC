#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From Seigo Ito topic no doki

#ROS
import rospy
import message_filters
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import Odometry

#PIL
from image_converter import decode, encode
import ImageDraw
import Image as PILImage

#chainer
import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable
from chainer import function
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.utils import type_check

#others
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import time
import numpy as np
import math
import numpy

i = 0
j = 0
ini = 0
count_safe = 0
count = 0
Lpast = 1.0
vel_res = 0.0
wvel_res = 0.0
vel_ref = 0.0
wvel_ref = 0.0
Lth = 0.18

swopt = 1
vwkeep = cuda.to_gpu(np.zeros((16,), dtype=np.float32))

#VUNet360
model_file_dec = 'nn_model/vunet_360_dec.h5'
model_file_enc = 'nn_model/vunet_360_enc.h5'

#GONet
model_file_gen = 'nn_model/featlayer_gen_single.h5'
model_file_dis = 'nn_model/featlayer_dis_single.h5'
model_file_invg = 'nn_model/featlayer_invg_single.h5'
model_file_fl = 'nn_model/classlayer_retrain.h5' #after re-training with simdata

#dvmpc
model_file_dvmpc = 'nn_model/dvmpc.h5'

nz = 100
ratio = 0.1

#center of picture
#yoko
xc = 310
#tate
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XYf = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]
XYb = [(xc+xplus-xyoffset, yc-xyoffset), (xc+xplus+xyoffset, yc+xyoffset)]

# resize parameters
rsizex = 128
rsizey = 128

#load mask
mask_br360 = np.loadtxt(open("mask_360view.csv", "rb"), delimiter=",", skiprows=0)
#print mask_br.shape
mask_brr = mask_br360.reshape((1,1,128,256)).astype(np.float32)
mask_brr1 = mask_br360.reshape((1,128,256)).astype(np.float32)
mask_brrc = np.concatenate((mask_brr1, mask_brr1, mask_brr1), axis=0)

Lmin = 100

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        _, h, w = image.shape
        imagex = (image - 128)/128

        return imagex, label

class Vmpc(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = L.Convolution2D(512, 16, 4, 2, 1, initialW=w)
        super(Vmpc, self).__init__(**layers)

    def __call__(self, x):
        hs = F.leaky_relu(self.c0(x))
        for i in range(1,8):
            hs = self['c%d'%i](hs)
        return hs

class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(528, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c1'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c2'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c3'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(512, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(256, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(128, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(64, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs, vres, wres):
        z = F.concat((hs[-1],vres,wres), axis=1)
        h = self.c0(z)
        for i in range(1,8):
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = F.tanh(self.c7(h))
        return h

class Generator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 8*8*512, initialW=initializer),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initializer),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initializer),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initializer),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=initializer),
            bn0l = L.BatchNormalization(8*8*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 8, 8))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x

class invG(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(invG, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, nz, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h))) 
        h = F.relu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

class ELU(function.Function):

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,

def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)

class Discriminator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, 2, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))
        h = elu(self.bn1(self.c1(h)))
        h = elu(self.bn2(self.c2(h))) 
        h = elu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return h

class FL(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(FL, self).__init__(
            l_img = L.Linear(3*128*128, 1, initialW=initializer),
            l_dis = L.Linear(512*8*8, 1, initialW=initializer),
            l_fdis = L.Linear(512*8*8, 1, initialW=initializer),
            l_FL = L.Linear(3, 1, initialW=initializer),
        )
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.absolute(img_error), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        g = F.reshape(F.absolute(dis_error), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        ghf = F.sigmoid(self.l_FL(F.concat((h,g,f), axis=1)))
        return ghf

def preprocess_image(msg):
    cv2_msg_img_L = bridge.imgmsg_to_cv2(msg)
    cv_imgc_L = bridge.cv2_to_imgmsg(cv2_msg_img_L, 'rgb8')
    pil_img_L = encode(cv_imgc_L)
    fg_img = PILImage.new('RGBA', pil_img_L.size, (0, 0, 0, 255))
    draw=ImageDraw.Draw(fg_img)
    draw.ellipse(XYf, fill = (0, 0, 0, 0))
    draw.ellipse(XYb, fill = (0, 0, 0, 0))
    pil_img_L.paste(fg_img, (0, 0), fg_img.split()[3])
    img_msg_L = decode(pil_img_L)
    cv2_img_L = bridge.imgmsg_to_cv2(img_msg_L, 'rgb8')
    cv_cutimg_F = cv2_img_L[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
    cv_cutimg_B = cv2_img_L[yc-xyoffset:yc+xyoffset, xc+xplus-xyoffset:xc+xplus+xyoffset]

    cv_cutimg_FF = cv2.transpose(cv_cutimg_F)
    cv_cutimg_F = cv2.flip(cv_cutimg_FF, 1)
    cv_cutimg_Bt = cv2.transpose(cv_cutimg_B)
    cv_cutimg_B = cv2.flip(cv_cutimg_Bt, 0)
    cv_cutimg_BF = cv2.flip(cv_cutimg_Bt, -1)

    #resize image
    cv_resize_F = cv2.resize(cv_cutimg_F,(rsizex, rsizey)) #front image
    cv_resize_B = cv2.resize(cv_cutimg_B,(rsizex, rsizey)) #back image

    cv_resize_n = np.concatenate((cv_resize_F, cv_resize_B), axis=1)
      
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 128)/128

    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img

def callback(msg_1):
    global i
    global j
    global timagev
    global Nline
    global count
    global Lpast
    global inputgpu
    global prefv
    global opt_biasv
    global swopt
    global Lmin
    global vwkeep
    global Lth
    global mask_brrc
    global imgbt, imgrt, imggt    

    j = j + 1
    #print j
    if j == 1:
        cur_img = preprocess_image(msg_1) #current image

        #standard deviation and mean for current image
        imgbc = (np.reshape(cur_img[0][0],(1,128,256)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,256)) + 1.0)*0.5
        imgrc = (np.reshape(cur_img[0][2],(1,128,256)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgbc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgrc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgbc-mask_brr1*mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mask_brr1*mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgrc-mask_brr1*mean_cbgr[2]))/countm)

        #standard deviation and mean for subgoal image
        imgrt = (np.reshape(goal_img[0][0],(1,128,256)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,256)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,256)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgbt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgrt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgbt-mask_brr1*mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mask_brr1*mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgrt-mask_brr1*mean_tbgr[2]))/countm)

        #mean_ct[0] = (mean_cbgr[0] + mean_tbgr[0])*0.5
        #mean_ct[1] = (mean_cbgr[1] + mean_tbgr[1])*0.5
        #mean_ct[2] = (mean_cbgr[2] + mean_tbgr[2])*0.5
        #std_ct[0] = (std_cbgr[0] + std_tbgr[0])*0.5
        #std_ct[1] = (std_cbgr[1] + std_tbgr[1])*0.5
        #std_ct[2] = (std_cbgr[2] + std_tbgr[2])*0.5
        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        xcg = F.clip(Variable(cuda.to_gpu(cur_img)), -1.0, 1.0)

        imgbtt = (imgbt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgrtt = (imgrt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]
        goalt_img = np.array((np.reshape(np.concatenate((imgbtt, imggtt, imgrtt), axis = 0), (1,3,128,256))*mask_c - 0.5)*2.0, dtype=np.float32)
        timage = F.clip(Variable(cuda.to_gpu(goalt_img)), -1.0, 1.0)

        #current image
        xcgf = F.get_item(xcg,(slice(0,batchsize,1),slice(0,3,1),slice(0,128,1),slice(0,128,1)))
        xcgb = F.flip(F.get_item(xcg,(slice(0,batchsize,1),slice(0,3,1),slice(0,128,1),slice(128,256,1))), axis=3)
        xcgx = F.concat((xcgf, xcgb), axis=1)

        #subgoal image
        xpgf = F.get_item(timage,(slice(0,batchsize,1),slice(0,3,1),slice(0,128,1),slice(0,128,1)))
        xpgb_wof = F.get_item(timage,(slice(0,batchsize,1),slice(0,3,1),slice(0,128,1),slice(128,256,1)))
        xpgb = F.flip(xpgb_wof, axis=3)

        xcpg = F.concat((xcgf, xcgb, xpgf, xpgb), axis=1)

        #GONet
        with chainer.using_config('train', False):
            img_gen = gen(invg(xcgf))
            dis_real = dis(xcgf)
            dis_gen = dis(img_gen)
            outputc = fl(xcgf-img_gen, dis_real-dis_gen, dis_real)

        #DVMPC
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            vwres = dvmpc(xcpg)

        vwsp = F.separate(vwres, axis=1)
        v1 = F.reshape(vwsp[0],(batchsize,1,1,1))
        v2 = F.reshape(vwsp[1],(batchsize,1,1,1))
        v3 = F.reshape(vwsp[2],(batchsize,1,1,1))
        v4 = F.reshape(vwsp[3],(batchsize,1,1,1))
        v5 = F.reshape(vwsp[4],(batchsize,1,1,1))
        v6 = F.reshape(vwsp[5],(batchsize,1,1,1))
        v7 = F.reshape(vwsp[6],(batchsize,1,1,1))
        v8 = F.reshape(vwsp[7],(batchsize,1,1,1))
        w1 = F.reshape(vwsp[8],(batchsize,1,1,1))
        w2 = F.reshape(vwsp[9],(batchsize,1,1,1))
        w3 = F.reshape(vwsp[10],(batchsize,1,1,1))
        w4 = F.reshape(vwsp[11],(batchsize,1,1,1))
        w5 = F.reshape(vwsp[12],(batchsize,1,1,1))
        w6 = F.reshape(vwsp[13],(batchsize,1,1,1))
        w7 = F.reshape(vwsp[14],(batchsize,1,1,1))
        w8 = F.reshape(vwsp[15],(batchsize,1,1,1))

        vresg = F.tanh(F.concat((v1, v2, v3, v4, v5, v6, v7, v8), axis=1))*0.5
        wresg = F.tanh(F.concat((w1, w2, w3, w4, w5, w6, w7, w8), axis=1))*1.0
        #print "linear", v1, v2, v3, v4, v5, v6, v7, v8
        #print "angular", w1, w2, w3, w4, w5, w6, w7, w8
        
        #VUNet-360
        with chainer.using_config('train', False):
            z = enc(xcgx)
            x = dec(z, vresg, wresg)

        xsp = F.separate(x, axis=1)

        apf0f_f = F.reshape(xsp[0],(batchsize,1,128,128))
        apf1f_f = F.reshape(xsp[1],(batchsize,1,128,128))
        apf2f_f = F.reshape(xsp[2],(batchsize,1,128,128))
        apf3f_f = F.reshape(xsp[3],(batchsize,1,128,128))
        apf4f_f = F.reshape(xsp[4],(batchsize,1,128,128))
        apf5f_f = F.reshape(xsp[5],(batchsize,1,128,128))
        apf6f_f = F.reshape(xsp[6],(batchsize,1,128,128))
        apf7f_f = F.reshape(xsp[7],(batchsize,1,128,128))
        apf8f_f = F.reshape(xsp[8],(batchsize,1,128,128))
        apf9f_f = F.reshape(xsp[9],(batchsize,1,128,128))
        apf10f_f = F.reshape(xsp[10],(batchsize,1,128,128))
        apf11f_f = F.reshape(xsp[11],(batchsize,1,128,128))
        apf12f_f = F.reshape(xsp[12],(batchsize,1,128,128))
        apf13f_f = F.reshape(xsp[13],(batchsize,1,128,128))
        apf14f_f = F.reshape(xsp[14],(batchsize,1,128,128))
        apf15f_f = F.reshape(xsp[15],(batchsize,1,128,128))

        apf0b_f = F.reshape(xsp[16],(batchsize,1,128,128))
        apf1b_f = F.reshape(xsp[17],(batchsize,1,128,128))
        apf2b_f = F.reshape(xsp[18],(batchsize,1,128,128))
        apf3b_f = F.reshape(xsp[19],(batchsize,1,128,128))
        apf4b_f = F.reshape(xsp[20],(batchsize,1,128,128))
        apf5b_f = F.reshape(xsp[21],(batchsize,1,128,128))
        apf6b_f = F.reshape(xsp[22],(batchsize,1,128,128))
        apf7b_f = F.reshape(xsp[23],(batchsize,1,128,128))
        apf8b_f = F.reshape(xsp[24],(batchsize,1,128,128))
        apf9b_f = F.reshape(xsp[25],(batchsize,1,128,128))
        apf10b_f = F.reshape(xsp[26],(batchsize,1,128,128))
        apf11b_f = F.reshape(xsp[27],(batchsize,1,128,128))
        apf12b_f = F.reshape(xsp[28],(batchsize,1,128,128))
        apf13b_f = F.reshape(xsp[29],(batchsize,1,128,128))
        apf14b_f = F.reshape(xsp[30],(batchsize,1,128,128))
        apf15b_f = F.reshape(xsp[31],(batchsize,1,128,128))

        w1f_f = F.reshape(xsp[32],(batchsize,1,128,128))
        w2f_f = F.reshape(xsp[33],(batchsize,1,128,128))
        w3f_f = F.reshape(xsp[34],(batchsize,1,128,128))
        w4f_f = F.reshape(xsp[35],(batchsize,1,128,128))
        w5f_f = F.reshape(xsp[36],(batchsize,1,128,128))
        w6f_f = F.reshape(xsp[37],(batchsize,1,128,128))
        w7f_f = F.reshape(xsp[38],(batchsize,1,128,128))
        w8f_f = F.reshape(xsp[39],(batchsize,1,128,128))

        w1b_f = F.reshape(xsp[40],(batchsize,1,128,128))
        w2b_f = F.reshape(xsp[41],(batchsize,1,128,128))
        w3b_f = F.reshape(xsp[42],(batchsize,1,128,128))
        w4b_f = F.reshape(xsp[43],(batchsize,1,128,128))
        w5b_f = F.reshape(xsp[44],(batchsize,1,128,128))
        w6b_f = F.reshape(xsp[45],(batchsize,1,128,128))
        w7b_f = F.reshape(xsp[46],(batchsize,1,128,128))
        w8b_f = F.reshape(xsp[47],(batchsize,1,128,128))

        apf_f1f_f = F.concat((apf0f_f, apf1f_f), axis=1)
        apf_f2f_f = F.concat((apf2f_f, apf3f_f), axis=1)
        apf_f3f_f = F.concat((apf4f_f, apf5f_f), axis=1)
        apf_f4f_f = F.concat((apf6f_f, apf7f_f), axis=1)
        apf_f5f_f = F.concat((apf8f_f, apf9f_f), axis=1)
        apf_f6f_f = F.concat((apf10f_f, apf11f_f), axis=1)
        apf_f7f_f = F.concat((apf12f_f, apf13f_f), axis=1)
        apf_f8f_f = F.concat((apf14f_f, apf15f_f), axis=1)

        apf_f1b_f = F.concat((apf0b_f, apf1b_f), axis=1)
        apf_f2b_f = F.concat((apf2b_f, apf3b_f), axis=1)
        apf_f3b_f = F.concat((apf4b_f, apf5b_f), axis=1)
        apf_f4b_f = F.concat((apf6b_f, apf7b_f), axis=1)
        apf_f5b_f = F.concat((apf8b_f, apf9b_f), axis=1)
        apf_f6b_f = F.concat((apf10b_f, apf11b_f), axis=1)
        apf_f7b_f = F.concat((apf12b_f, apf13b_f), axis=1)
        apf_f8b_f = F.concat((apf14b_f, apf15b_f), axis=1)
           
        genL1f_f = F.spatial_transformer_sampler(xcgf, apf_f1f_f)
        genL2f_f = F.spatial_transformer_sampler(xcgf, apf_f2f_f)
        genL3f_f = F.spatial_transformer_sampler(xcgf, apf_f3f_f)
        genL4f_f = F.spatial_transformer_sampler(xcgf, apf_f4f_f)
        genL5f_f = F.spatial_transformer_sampler(xcgf, apf_f5f_f)
        genL6f_f = F.spatial_transformer_sampler(xcgf, apf_f6f_f)
        genL7f_f = F.spatial_transformer_sampler(xcgf, apf_f7f_f)
        genL8f_f = F.spatial_transformer_sampler(xcgf, apf_f8f_f)

        genL1b_f = F.spatial_transformer_sampler(xcgb, apf_f1b_f)
        genL2b_f = F.spatial_transformer_sampler(xcgb, apf_f2b_f)
        genL3b_f = F.spatial_transformer_sampler(xcgb, apf_f3b_f)
        genL4b_f = F.spatial_transformer_sampler(xcgb, apf_f4b_f)
        genL5b_f = F.spatial_transformer_sampler(xcgb, apf_f5b_f)
        genL6b_f = F.spatial_transformer_sampler(xcgb, apf_f6b_f)
        genL7b_f = F.spatial_transformer_sampler(xcgb, apf_f7b_f)
        genL8b_f = F.spatial_transformer_sampler(xcgb, apf_f8b_f)

        mask1_f = F.concat((w1f_f, w1b_f), axis=1)
        mask2_f = F.concat((w2f_f, w2b_f), axis=1)
        mask3_f = F.concat((w3f_f, w3b_f), axis=1)
        mask4_f = F.concat((w4f_f, w4b_f), axis=1)
        mask5_f = F.concat((w5f_f, w5b_f), axis=1)
        mask6_f = F.concat((w6f_f, w6b_f), axis=1)
        mask7_f = F.concat((w7f_f, w7b_f), axis=1)
        mask8_f = F.concat((w8f_f, w8b_f), axis=1)

        mask_soft1_f = F.softmax(mask1_f, axis=1)
        mask_soft2_f = F.softmax(mask2_f, axis=1)
        mask_soft3_f = F.softmax(mask3_f, axis=1)
        mask_soft4_f = F.softmax(mask4_f, axis=1)
        mask_soft5_f = F.softmax(mask5_f, axis=1)
        mask_soft6_f = F.softmax(mask6_f, axis=1)
        mask_soft7_f = F.softmax(mask7_f, axis=1)
        mask_soft8_f = F.softmax(mask8_f, axis=1)

        mask_sep1_f = F.separate(mask_soft1_f, axis=1)
        mask_sep2_f = F.separate(mask_soft2_f, axis=1)
        mask_sep3_f = F.separate(mask_soft3_f, axis=1)
        mask_sep4_f = F.separate(mask_soft4_f, axis=1)
        mask_sep5_f = F.separate(mask_soft5_f, axis=1)
        mask_sep6_f = F.separate(mask_soft6_f, axis=1)
        mask_sep7_f = F.separate(mask_soft7_f, axis=1)
        mask_sep8_f = F.separate(mask_soft8_f, axis=1)

        mask_1f_f = F.reshape(mask_sep1_f[0],(batchsize,1,128,128))
        mask_1b_f = F.reshape(mask_sep1_f[1],(batchsize,1,128,128))
        mask_2f_f = F.reshape(mask_sep2_f[0],(batchsize,1,128,128))
        mask_2b_f = F.reshape(mask_sep2_f[1],(batchsize,1,128,128))
        mask_3f_f = F.reshape(mask_sep3_f[0],(batchsize,1,128,128))
        mask_3b_f = F.reshape(mask_sep3_f[1],(batchsize,1,128,128))
        mask_4f_f = F.reshape(mask_sep4_f[0],(batchsize,1,128,128))
        mask_4b_f = F.reshape(mask_sep4_f[1],(batchsize,1,128,128))
        mask_5f_f = F.reshape(mask_sep5_f[0],(batchsize,1,128,128))
        mask_5b_f = F.reshape(mask_sep5_f[1],(batchsize,1,128,128))
        mask_6f_f = F.reshape(mask_sep6_f[0],(batchsize,1,128,128))
        mask_6b_f = F.reshape(mask_sep6_f[1],(batchsize,1,128,128))
        mask_7f_f = F.reshape(mask_sep7_f[0],(batchsize,1,128,128))
        mask_7b_f = F.reshape(mask_sep7_f[1],(batchsize,1,128,128))
        mask_8f_f = F.reshape(mask_sep8_f[0],(batchsize,1,128,128))
        mask_8b_f = F.reshape(mask_sep8_f[1],(batchsize,1,128,128))
        genL1x_f = F.scale(genL1f_f,mask_1f_f,axis=0) + F.scale(genL1b_f,mask_1b_f,axis=0)
        genL2x_f = F.scale(genL2f_f,mask_2f_f,axis=0) + F.scale(genL2b_f,mask_2b_f,axis=0)
        genL3x_f = F.scale(genL3f_f,mask_3f_f,axis=0) + F.scale(genL3b_f,mask_3b_f,axis=0)
        genL4x_f = F.scale(genL4f_f,mask_4f_f,axis=0) + F.scale(genL4b_f,mask_4b_f,axis=0)
        genL5x_f = F.scale(genL5f_f,mask_5f_f,axis=0) + F.scale(genL5b_f,mask_5b_f,axis=0)
        genL6x_f = F.scale(genL6f_f,mask_6f_f,axis=0) + F.scale(genL6b_f,mask_6b_f,axis=0)
        genL7x_f = F.scale(genL7f_f,mask_7f_f,axis=0) + F.scale(genL7b_f,mask_7b_f,axis=0)
        genL8x_f = F.scale(genL8f_f,mask_8f_f,axis=0) + F.scale(genL8b_f,mask_8b_f,axis=0)

        xap_f = F.concat((genL1x_f, genL2x_f, genL3x_f, genL4x_f, genL5x_f, genL6x_f, genL7x_f, 5.0*genL8x_f), axis=1)

        apf0f_b = F.reshape(xsp[48],(batchsize,1,128,128))
        apf1f_b = F.reshape(xsp[49],(batchsize,1,128,128))
        apf2f_b = F.reshape(xsp[50],(batchsize,1,128,128))
        apf3f_b = F.reshape(xsp[51],(batchsize,1,128,128))
        apf4f_b = F.reshape(xsp[52],(batchsize,1,128,128))
        apf5f_b = F.reshape(xsp[53],(batchsize,1,128,128))
        apf6f_b = F.reshape(xsp[54],(batchsize,1,128,128))
        apf7f_b = F.reshape(xsp[55],(batchsize,1,128,128))
        apf8f_b = F.reshape(xsp[56],(batchsize,1,128,128))
        apf9f_b = F.reshape(xsp[57],(batchsize,1,128,128))
        apf10f_b = F.reshape(xsp[58],(batchsize,1,128,128))
        apf11f_b = F.reshape(xsp[59],(batchsize,1,128,128))
        apf12f_b = F.reshape(xsp[60],(batchsize,1,128,128))
        apf13f_b = F.reshape(xsp[61],(batchsize,1,128,128))
        apf14f_b = F.reshape(xsp[62],(batchsize,1,128,128))
        apf15f_b = F.reshape(xsp[63],(batchsize,1,128,128))

        apf0b_b = F.reshape(xsp[64],(batchsize,1,128,128))
        apf1b_b = F.reshape(xsp[65],(batchsize,1,128,128))
        apf2b_b = F.reshape(xsp[66],(batchsize,1,128,128))
        apf3b_b = F.reshape(xsp[67],(batchsize,1,128,128))
        apf4b_b = F.reshape(xsp[68],(batchsize,1,128,128))
        apf5b_b = F.reshape(xsp[69],(batchsize,1,128,128))
        apf6b_b = F.reshape(xsp[70],(batchsize,1,128,128))
        apf7b_b = F.reshape(xsp[71],(batchsize,1,128,128))
        apf8b_b = F.reshape(xsp[72],(batchsize,1,128,128))
        apf9b_b = F.reshape(xsp[73],(batchsize,1,128,128))
        apf10b_b = F.reshape(xsp[74],(batchsize,1,128,128))
        apf11b_b = F.reshape(xsp[75],(batchsize,1,128,128))
        apf12b_b = F.reshape(xsp[76],(batchsize,1,128,128))
        apf13b_b = F.reshape(xsp[77],(batchsize,1,128,128))
        apf14b_b = F.reshape(xsp[78],(batchsize,1,128,128))
        apf15b_b = F.reshape(xsp[79],(batchsize,1,128,128))

        w1f_b = F.reshape(xsp[80],(batchsize,1,128,128))
        w2f_b = F.reshape(xsp[81],(batchsize,1,128,128))
        w3f_b = F.reshape(xsp[82],(batchsize,1,128,128))
        w4f_b = F.reshape(xsp[83],(batchsize,1,128,128))
        w5f_b = F.reshape(xsp[84],(batchsize,1,128,128))
        w6f_b = F.reshape(xsp[85],(batchsize,1,128,128))
        w7f_b = F.reshape(xsp[86],(batchsize,1,128,128))
        w8f_b = F.reshape(xsp[87],(batchsize,1,128,128))

        w1b_b = F.reshape(xsp[88],(batchsize,1,128,128))
        w2b_b = F.reshape(xsp[89],(batchsize,1,128,128))
        w3b_b = F.reshape(xsp[90],(batchsize,1,128,128))
        w4b_b = F.reshape(xsp[91],(batchsize,1,128,128))
        w5b_b = F.reshape(xsp[92],(batchsize,1,128,128))
        w6b_b = F.reshape(xsp[93],(batchsize,1,128,128))
        w7b_b = F.reshape(xsp[94],(batchsize,1,128,128))
        w8b_b = F.reshape(xsp[95],(batchsize,1,128,128))

        apf_b1f_b = F.concat((apf0f_b, apf1f_b), axis=1)
        apf_b2f_b = F.concat((apf2f_b, apf3f_b), axis=1)
        apf_b3f_b = F.concat((apf4f_b, apf5f_b), axis=1)
        apf_b4f_b = F.concat((apf6f_b, apf7f_b), axis=1)
        apf_b5f_b = F.concat((apf8f_b, apf9f_b), axis=1)
        apf_b6f_b = F.concat((apf10f_b, apf11f_b), axis=1)
        apf_b7f_b = F.concat((apf12f_b, apf13f_b), axis=1)
        apf_b8f_b = F.concat((apf14f_b, apf15f_b), axis=1)

        apf_b1b_b = F.concat((apf0b_b, apf1b_b), axis=1)
        apf_b2b_b = F.concat((apf2b_b, apf3b_b), axis=1)
        apf_b3b_b = F.concat((apf4b_b, apf5b_b), axis=1)
        apf_b4b_b = F.concat((apf6b_b, apf7b_b), axis=1)
        apf_b5b_b = F.concat((apf8b_b, apf9b_b), axis=1)
        apf_b6b_b = F.concat((apf10b_b, apf11b_b), axis=1)
        apf_b7b_b = F.concat((apf12b_b, apf13b_b), axis=1)
        apf_b8b_b = F.concat((apf14b_b, apf15b_b), axis=1)
           
        genL1f_b = F.spatial_transformer_sampler(xcgf, apf_b1f_b)
        genL2f_b = F.spatial_transformer_sampler(xcgf, apf_b2f_b)
        genL3f_b = F.spatial_transformer_sampler(xcgf, apf_b3f_b)
        genL4f_b = F.spatial_transformer_sampler(xcgf, apf_b4f_b)
        genL5f_b = F.spatial_transformer_sampler(xcgf, apf_b5f_b)
        genL6f_b = F.spatial_transformer_sampler(xcgf, apf_b6f_b)
        genL7f_b = F.spatial_transformer_sampler(xcgf, apf_b7f_b)
        genL8f_b = F.spatial_transformer_sampler(xcgf, apf_b8f_b)
        genL1b_b = F.spatial_transformer_sampler(xcgb, apf_b1b_b)
        genL2b_b = F.spatial_transformer_sampler(xcgb, apf_b2b_b)
        genL3b_b = F.spatial_transformer_sampler(xcgb, apf_b3b_b)
        genL4b_b = F.spatial_transformer_sampler(xcgb, apf_b4b_b)
        genL5b_b = F.spatial_transformer_sampler(xcgb, apf_b5b_b)
        genL6b_b = F.spatial_transformer_sampler(xcgb, apf_b6b_b)
        genL7b_b = F.spatial_transformer_sampler(xcgb, apf_b7b_b)
        genL8b_b = F.spatial_transformer_sampler(xcgb, apf_b8b_b)
        mask1_b = F.concat((w1f_b, w1b_b), axis=1)
        mask2_b = F.concat((w2f_b, w2b_b), axis=1)
        mask3_b = F.concat((w3f_b, w3b_b), axis=1)
        mask4_b = F.concat((w4f_b, w4b_b), axis=1)
        mask5_b = F.concat((w5f_b, w5b_b), axis=1)
        mask6_b = F.concat((w6f_b, w6b_b), axis=1)
        mask7_b = F.concat((w7f_b, w7b_b), axis=1)
        mask8_b = F.concat((w8f_b, w8b_b), axis=1)

        mask_soft1_b = F.softmax(mask1_b, axis=1)
        mask_soft2_b = F.softmax(mask2_b, axis=1)
        mask_soft3_b = F.softmax(mask3_b, axis=1)
        mask_soft4_b = F.softmax(mask4_b, axis=1)
        mask_soft5_b = F.softmax(mask5_b, axis=1)
        mask_soft6_b = F.softmax(mask6_b, axis=1)
        mask_soft7_b = F.softmax(mask7_b, axis=1)
        mask_soft8_b = F.softmax(mask8_b, axis=1)

        mask_sep1_b = F.separate(mask_soft1_b, axis=1)
        mask_sep2_b = F.separate(mask_soft2_b, axis=1)
        mask_sep3_b = F.separate(mask_soft3_b, axis=1)
        mask_sep4_b = F.separate(mask_soft4_b, axis=1)
        mask_sep5_b = F.separate(mask_soft5_b, axis=1)
        mask_sep6_b = F.separate(mask_soft6_b, axis=1)
        mask_sep7_b = F.separate(mask_soft7_b, axis=1)
        mask_sep8_b = F.separate(mask_soft8_b, axis=1)

        mask_1f_b = F.reshape(mask_sep1_b[0],(batchsize,1,128,128))
        mask_1b_b = F.reshape(mask_sep1_b[1],(batchsize,1,128,128))
        mask_2f_b = F.reshape(mask_sep2_b[0],(batchsize,1,128,128))
        mask_2b_b = F.reshape(mask_sep2_b[1],(batchsize,1,128,128))
        mask_3f_b = F.reshape(mask_sep3_b[0],(batchsize,1,128,128))
        mask_3b_b = F.reshape(mask_sep3_b[1],(batchsize,1,128,128))
        mask_4f_b = F.reshape(mask_sep4_b[0],(batchsize,1,128,128))
        mask_4b_b = F.reshape(mask_sep4_b[1],(batchsize,1,128,128))
        mask_5f_b = F.reshape(mask_sep5_b[0],(batchsize,1,128,128))
        mask_5b_b = F.reshape(mask_sep5_b[1],(batchsize,1,128,128))
        mask_6f_b = F.reshape(mask_sep6_b[0],(batchsize,1,128,128))
        mask_6b_b = F.reshape(mask_sep6_b[1],(batchsize,1,128,128))
        mask_7f_b = F.reshape(mask_sep7_b[0],(batchsize,1,128,128))
        mask_7b_b = F.reshape(mask_sep7_b[1],(batchsize,1,128,128))
        mask_8f_b = F.reshape(mask_sep8_b[0],(batchsize,1,128,128))
        mask_8b_b = F.reshape(mask_sep8_b[1],(batchsize,1,128,128))
        genL1x_b = F.scale(genL1f_b,mask_1f_b,axis=0) + F.scale(genL1b_b,mask_1b_b,axis=0)
        genL2x_b = F.scale(genL2f_b,mask_2f_b,axis=0) + F.scale(genL2b_b,mask_2b_b,axis=0)
        genL3x_b = F.scale(genL3f_b,mask_3f_b,axis=0) + F.scale(genL3b_b,mask_3b_b,axis=0)
        genL4x_b = F.scale(genL4f_b,mask_4f_b,axis=0) + F.scale(genL4b_b,mask_4b_b,axis=0)
        genL5x_b = F.scale(genL5f_b,mask_5f_b,axis=0) + F.scale(genL5b_b,mask_5b_b,axis=0)
        genL6x_b = F.scale(genL6f_b,mask_6f_b,axis=0) + F.scale(genL6b_b,mask_6b_b,axis=0)
        genL7x_b = F.scale(genL7f_b,mask_7f_b,axis=0) + F.scale(genL7b_b,mask_7b_b,axis=0)
        genL8x_b = F.scale(genL8f_b,mask_8f_b,axis=0) + F.scale(genL8b_b,mask_8b_b,axis=0)

        xap_b = F.concat((genL1x_b, genL2x_b, genL3x_b, genL4x_b, genL5x_b, genL6x_b, genL7x_b, 5.0*genL8x_b), axis=1)
        #VUNet-360 end

        #GONet
        xgonet = F.concat((genL1x_f, genL2x_f, genL3x_f, genL4x_f, genL5x_f, genL6x_f, genL7x_f, genL8x_f), axis=0)
        with chainer.using_config('train', False):
            img_gen = gen(invg(xgonet))
            dis_real = dis(xgonet)
            dis_gen = dis(img_gen)
            output = fl(xgonet-img_gen, dis_real-dis_gen, dis_real)

        xrefy = F.concat((Variable(cur_img), Variable(goal_img)), axis=3)
        xgonetx = F.concat((genL1x_f, genL2x_f, genL3x_f, genL4x_f, genL5x_f, genL6x_f, genL7x_f, genL8x_f), axis=3)
        xgonety = F.concat((genL1x_b, genL2x_b, genL3x_b, genL4x_b, genL5x_b, genL6x_b, genL7x_b, genL8x_b), axis=3)

        msg_pub = Twist()
        
        #velocity cap and publish
        vt = cuda.to_cpu(vresg.data[0][0])
        wt = cuda.to_cpu(wresg.data[0][0])
        if np.absolute(vt) < 0.2:
            msg_pub.linear.x = vt
            msg_pub.linear.y = 0.0
            msg_pub.linear.z = 0.0
            msg_pub.angular.x = 0.0
            msg_pub.angular.y = 0.0
            msg_pub.angular.z = wt
        else:
            if np.absolute(wt) < 0.001:
                msg_pub.linear.x = 0.2*np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                msg_pub.linear.x = 0.2*np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.2*np.sign(vt)/rd                

        #front predicted image
        out_imgc = cuda.to_cpu(xgonetx.data)
        imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
        imgc = np.reshape(imgb, (3, 128, 128*8))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge)
        image_genf.publish(imgm)

        #back predicted image
        out_imgc = cuda.to_cpu(xgonety.data)
        imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
        imgc = np.reshape(imgb, (3, 128, 128*8))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge)
        image_genb.publish(imgm)

        #current and goal image
        out_imgc = cuda.to_cpu(xrefy.data)
        imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
        imgc = np.reshape(imgb, (3, 128, 256*2))
        imgd = imgc.transpose(1, 2, 0)
        imge = imgd.astype(np.uint8)
        imgm = bridge.cv2_to_imgmsg(imge)
        image_ref.publish(imgm)

        #velocities
        msg_out.publish(msg_pub)
        j = 0

def callback_ref(msg):
    global goal_img
    goal_img = preprocess_image(msg) #subgoal image

xp = cuda.cupy
cuda.get_device(0).use()

#model definition
dvmpc = Vmpc(in_ch=12)
enc = Encoder(in_ch=6)
dec = Decoder(out_ch=96)

gen = Generator()
dis = Discriminator()
invg = invG()
fl = FL()


serializers.load_hdf5(model_file_invg, invg)
serializers.load_hdf5(model_file_gen, gen)
serializers.load_hdf5(model_file_dis, dis)
serializers.load_hdf5(model_file_fl, fl)

serializers.load_hdf5(model_file_dvmpc, dvmpc)
serializers.load_hdf5(model_file_dec, dec)
serializers.load_hdf5(model_file_enc, enc)

gen.to_gpu()
dis.to_gpu()
invg.to_gpu()
fl.to_gpu()

dvmpc.to_gpu()
enc.to_gpu()
dec.to_gpu()

batchsize = 1
goal_img = np.zeros((1,3,128,256), dtype=np.float32)

countm = 0
for it in range(128):
    for jt in range(256):
        if mask_brr[0][0][it][jt] > 0.5:
            countm += 1

print countm
mask_c = np.concatenate((mask_brr, mask_brr, mask_brr), axis=1)


# main function
if __name__ == '__main__':

    #initialize node
    rospy.init_node('sync_topic', anonymous=True)

    print 'sleeping 10s'
    rospy.sleep(10.0)
    #subscribe of topics
    msg1_sub = rospy.Subscriber('/cv_camera_node1/image', Image, callback)
    msg2_sub = rospy.Subscriber('/cv_camera_node2/image_ref', Image, callback_ref)

    #publisher of topics
    msg_out = rospy.Publisher('/cmd_vel_mux/input/ref_GONetpp', Twist,queue_size=10) #velocities for the robot control
    image_genf = rospy.Publisher('img_genf',Image,queue_size=10) #front predicted images
    image_genb = rospy.Publisher('img_genb',Image,queue_size=10) #back predicted images
    image_ref = rospy.Publisher('img_ref',Image,queue_size=10)   #current and reference images

    bridge = CvBridge()
    # waiting callback

    print 'waiting message .....'
    rospy.spin()
