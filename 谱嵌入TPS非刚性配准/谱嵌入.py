#!/usr/bin/python
# -*- coding: utf-8 -*-

print(__doc__)

from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state

# Next line to silence pyflakes.
Axes3D


#定义变量
# Variables for manifold learning.
n_neighbors = 10
n_samples = 1000

modelName = 'FeMale'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/模型/data_v(2).obj'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/参考代码/3D-Smoothing-meshes/example_meshes/bunny.obj'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/curcature/guassCurvature'+modelName+'.txt'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/小论文/师弟截图/sample_filter_MeshedReconstruction1.obj'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/sample_filter_大字型左右转left.obj'
inputFile = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/降采样/sample_filter_抬手2.obj'
# inputFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样滤波曲率特征/MeshedReconstruction1SampleFilterresultCurvature.obj'
# inputFile = '球.obj'


# spectralFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/spectral/spectral'+modelName+'0test.txt'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/sample_filter_大字型左右转right.txt'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样滤波曲率特征/spectralMeshedReconstruction1SampleFilterresultCurvature.obj'

x = []
y = []
z = []
i = 0
curcatureList = []

#旋转矩阵，可选
import math
angle = 0
rotaX = np.array([[1,0,0],
         [0,math.cos(angle),math.sin(angle)],
         [0,-math.sin(angle),math.cos(angle)]])
angle = 200
rotaY = np.array([[math.cos(angle),0,-math.sin(angle)],
         [0,1,0],
         [math.sin(angle),0,math.cos(angle)]])
angle = 0
rotaZ = np.array([[math.cos(angle),math.sin(angle),0],
         [-math.sin(angle),math.cos(angle),0],
         [0,0,1]])

# outputFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/curcature/guassCurvature'+modelName+'Filter100.obj'
#读入数据
with open(inputFile, "r") as f:
    for line in f.readlines():
        if (line[0] == 'f'):
            continue
        if(line[0]=='v'):
            line = line.replace('v ', '')
        #     print(line)
        line = line.replace('	',' ').split(' ')

        # if(len(line)>3 and abs(float(line[3]))>100 ):
        if (len(line) >= 3 ):
            x.append(float(line[0]))
            y.append(float(line[1]))
            z.append(float(line[2]))
            # curcatureList.append(float(line[3]))
            i = i + 1
x, y, z = (np.asarray(x), \
    np.asarray(y), \
    np.asarray(z))

rotation = np.dot(rotaZ ,np.dot(rotaY , np.dot(rotaX ,np.array([x,y,z]))))
x, y, z = rotation[0],rotation[1],rotation[2]
# print(x,y,z)
n_samples = i
# print(i)
# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi
# Sever the poles from the sphere.
indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
colors = p[indices]

X = []
sphere_data = np.array([x, y, z]).T
print(sphere_data.shape)

#谱嵌入

t0 = time()
se = manifold.SpectralEmbedding(n_components=2,
                                n_neighbors=n_neighbors)
trans_data = se.fit_transform(sphere_data).T
t1 = time()
print("Spectral Embedding: %.2g sec" % (t1 - t0))

#可视化谱嵌入结果
# Plot our dataset.
fig = plt.figure(figsize=(110, 3))
plt.suptitle(u"标准模型1谱嵌入结果", fontsize=14)
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x, y, z, cmap=plt.cm.rainbow,s=1)
ax.view_init(40, -10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.title("原始模型" )

from numpy import *
# trans_data_ = trans_data.T
print(sphere_data.shape)
ax = fig.add_subplot(122)
ax.scatter(trans_data[0], trans_data[1], cmap=plt.cm.rainbow,s=1)
plt.title("谱嵌入")

plt.show()

#谱嵌入文件保存
# import os
# if(os.path.exists(spectralFile)):
#     os.remove(spectralFile)
# fwrite = open(spectralFile, "w")
# for k in range(len(trans_data[0])):
#     fwrite.write(str(trans_data[0][k])+'\t'+str(trans_data[1][k])+'\t'+str(curcatureList[k])+'\n')
# fwrite.close()
