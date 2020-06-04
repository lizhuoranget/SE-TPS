import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

#TPS方法
def makeT(cp):
    # cp: [K x 3] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K+4, K+4))
    T[:K, 0] = 1
    T[:K, 1:4] = cp
    T[K, 4:] = 1
    T[K+1:, 4:] = cp.T
    # compute every point pair of points
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1 # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 4:] = R
    return T

def liftPts(p, cp):
    # p: [N x 3], input points
    # cp: [K x 3], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K+4))
    pLift[:,0] = 1
    pLift[:,1:4] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:, 4:] = R
    return pLift

def tps_transform(gallery, probe):
    """
    Compute the new points coordination with Thin-Plate-Spline algorithm
    """
    src_pt_xs = probe[:, 0]
    src_pt_ys = probe[:, 1]
    src_pt_zs = probe[:, 2]
    cps = np.vstack([src_pt_xs, src_pt_ys, src_pt_zs]).T
    # construct T
    T = makeT(cps)
    # solve cx, cy (coefficients for x and y)
    tar_pt_xt = gallery[:, 0]
    tar_pt_yt = gallery[:, 1]
    tar_pt_zt = gallery[:, 2]

    xtAug = np.concatenate([tar_pt_xt, np.zeros(4)])
    ytAug = np.concatenate([tar_pt_yt, np.zeros(4)])
    ztAug = np.concatenate([tar_pt_zt, np.zeros(4)])

    cx = np.linalg.solve(T, xtAug)  # [K+3]
    cy = np.linalg.solve(T, ytAug)
    cz = np.linalg.solve(T, ztAug)

    return cx, cy, cz

#点分类，分为重合点和非重合点
def classPoint(cloud1,cloud2):

    return


#定义变量
modelName = 'HelloFemale'
tagetName = 'Female'
spectralFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/curcature/guassCurvature' + modelName + '.txt'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样后文件/MeshedReconstructionFilterresult_.pcd'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样滤波曲率特征/spectralMeshedReconstructionSampleFilterresultCurvature.obj'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/小论文/师弟截图/sample_filter_MeshedReconstruction1.obj'
# spectralFile = '/Users/lizhuoran/Desktop/FXQ/小论文/师弟截图/sample_filter_MeshedReconstruction1.obj'
spectralFile = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/sample_filter_大字型左右转left.obj'

# spectralFile2 = '/Users/lizhuoran/Desktop/FXQ/实验数据/curcature/guassCurvature' + tagetName + '.txt'
# spectralFile2 = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样后文件/MeshedReconstruction1Filterresult_.pcd'
# spectralFile2 = '/Users/lizhuoran/Desktop/FXQ/实验数据/pcd降采样滤波曲率特征/spectralMeshedReconstruction1SampleFilterresultCurvature.obj'
spectralFile2 = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/sample_filter_大字型左右转right.obj'

#读入数据
def readData(inputFile):
    x,y,z = [],[],[]
    xc,yc,zc = [],[],[]
    i = 0
    with open(inputFile, "r") as f:
        for line in f.readlines():
            if(line[0]=='f'):
                continue
            if(line[0]=='v'):
                line = line.replace('v ', '')
            line = line.replace('	', ' ').split(' ')
            # if(len(line)>3 and abs(float(line[3]))>100 ):
            if (len(line) >= 3):
                x.append(float(line[0]))
                y.append(float(line[1]))
                z.append(float(line[2]))
                i += 1
                # 控制点
                if (True):
                    # if(abs(float(line[3]))>0):
                    xc.append(float(line[0]))
                    yc.append(float(line[1]))
                    zc.append(float(line[2]))
    allPoint = []
    allPoint.append(x)
    allPoint.append(y)
    allPoint.append(z)
    allPoint = np.array(allPoint)

    controlPoint = []
    controlPoint.append(xc)
    controlPoint.append(yc)
    controlPoint.append(zc)
    controlPoint = np.array(controlPoint)

    return allPoint,controlPoint

probe, probe_c = readData(spectralFile)
gallery, gallery_c = readData(spectralFile2)

#模型点长度对齐
probe_store = probe

if(len(gallery[0])>len(probe[0])):
    gallery = gallery[:,:len(probe[0])]
elif(len(gallery[0])<len(probe[0])):
    probe = probe[:,:len(gallery[0])]

if(len(gallery_c[0])>len(probe_c[0])):
    gallery_c = gallery_c[:,:len(probe_c[0])]
elif(len(gallery_c[0])<len(probe_c[0])):
    probe_c = probe_c[:,:len(gallery_c[0])]

print(probe.shape, gallery.shape)

#特征点求得参数
cx,cy,cz = tps_transform(gallery_c.T , probe_c.T)
para = []
para.append(cx)
para.append(cy)
para.append(cz)
para = np.array(para).T

#驱动所有点转换
transed = liftPts(probe.T, probe_c.T)

print(transed.shape, para.shape)

#转换后结果
result = np.dot(transed, para)

#结果可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

fig = plt.figure(figsize=(110, 3))
plt.suptitle("Manifold Learning with", fontsize=14)

ax = fig.add_subplot(121, projection='3d')
ax.scatter(probe[0], probe[1], probe[2], cmap=plt.cm.rainbow,s=1)
ax.scatter(gallery[0], gallery[1],gallery[2], cmap=plt.cm.rainbow,s=1)
ax.view_init(40, -10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
ax.set_title('original')

ax = fig.add_subplot(122, projection='3d')
result = result.T
ax.scatter(result[0], result[1],result[2], cmap=plt.cm.rainbow,s=1)
ax.scatter(gallery[0], gallery[1],gallery[2], cmap=plt.cm.rainbow,s=1)
ax.view_init(40, -10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
ax.set_title('tps transform')

fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.show()


#输出文件
outputFile = '/Users/lizhuoran/Desktop/FXQ/小论文/师弟截图/TPS结果_所有点.obj'
outputFile = '/Users/lizhuoran/Desktop/FXQ/小论文/截图/左右转模型_TPS结果_分类test_sigma.obj'
f = open(outputFile,'w')
result = result.T
import random
for i in range(len(result)):
    f.write('v '+str(result[i][0]+0.000001 * random.randint(0,9))+' '+str(result[i][1]+0.0001 * random.randint(0,9))+' '+str(result[i][2]+0.0001 * random.randint(0,9))+'\n')
probe_store = probe_store.T
gallery = gallery.T
#计算误差
index = len(gallery)-1
delta_x = gallery[index][0]-probe_store[index][0]
delta_y = gallery[index][1]-probe_store[index][1]
delta_z = gallery[index][2]-probe_store[index][2]
delta_x = 0
# delta_y = 0
# delta_z = 0

for i in range(len(gallery),len(probe_store)):
    # print(i)
    f.write('v '+str(probe_store[i][0]+delta_x)+' '+str(probe_store[i][1]+delta_y)+' '+str(probe_store[i][2]+delta_z)+'\n')
    probe_store[i][0] += delta_x
    probe_store[i][1] += delta_y
    probe_store[i][2] += delta_z
for i in range(len(probe_store)):
    if(i%3==0):
        f.write('f '+'%s/%s'%(i+1,i+1)+' %s/%s'%(i+2,i+2)+' %s/%s'%(i+3,i+3)+'\n')
f.close()

probe_store = probe_store.T
fig = plt.figure(figsize=(110, 3))
plt.suptitle("Manifold Learning with", fontsize=14)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(probe_store[0], probe_store[1], probe_store[2], cmap=plt.cm.rainbow,s=1)
ax.view_init(40, -10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
ax.set_title('original')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.show()