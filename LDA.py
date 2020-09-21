import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as ax

def GetData():
    file = open("logReg_data.txt","r")

    data = []
    target = []
    c1_size = 0
    c2_size = 0
    #target = []
    for line in file:
        *X, y = (map(float,line.split()))

        if y==0:
            data.insert(c1_size, list(X))
            target.insert(c1_size, 0)
            c1_size += 1
        else:
            data.append(list(X))
            target.append(1)
            c2_size += 1

    data = np.array(data)

    #Centering Data
    mean = np.mean(data, axis=0)
    data -= mean


    return data,target,c1_size,c2_size


def Ploting(tit = "default"):
    # add grid
    plt.grid(True,linestyle='--')

    # add title
    plt.title(tit)
    plt.tight_layout()

    # add x,y axes labels
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')

def LDA(Data,Y,c1_size,c2_size):
    data_1 = Data[:c1_size]
    data_2 = Data[c1_size:]

    # mean of classes
    mean_1 = np.mean(data_1, axis=0)
    mean_2 = np.mean(data_2, axis=0)

    # between-class scatter matrix
    dif = mean_1 - mean_2
    Sb = np.outer(dif,dif)

    #  within-class scatter matrix
    dif = data_1 - mean_1
    Sw = np.dot(dif.T,dif)
    dif = data_2 - mean_2
    Sw += np.dot(dif.T,dif)

    #main eigenvector
    Result = np.dot(np.linalg.pinv(Sw),Sb)

    u, s, vh = np.linalg.svd(Result, full_matrices=True)
    s = np.diag(s)

    print("U:\n", u)
    print("S:\n", s, "\n")
    #print("Vt:\n", vh, "\n")

    eigenv = np.dot(u, s)[:,0].reshape(2,1)

    print("eigenv:\n", eigenv, "\n")

    ###############################
    plt.figure()

    plt.scatter(Data.T[0], Data.T[1], s=35, marker='.', c=Y)

    axis_x = np.linspace(Data[:, 1].min(), Data[:, 1].max(), 100)
    axis_y = (eigenv[1] / eigenv[0]) * axis_x
    plt.plot(axis_x, axis_y, ':r', label='Projection line')

    Ploting("Original Data")
    ###############################

    # project Data
    proj = np.dot(eigenv.T, eigenv)
    proj = np.dot(Data, eigenv) / proj
    proj = np.kron(proj,eigenv.T)

    ###############################
    plt.figure()

    plt.scatter(proj.T[0], proj.T[1], s=35, marker='.', c=Y)

    axis_x = np.linspace(Data[:, 1].min(), Data[:, 1].max(), 100)
    axis_y = (eigenv[1] / eigenv[0]) * axis_x
    plt.plot(axis_x, axis_y, ':g', label='Projection line')

    Ploting("Data projection into line")
    ###############################



if __name__ == '__main__':
    X,Y,c1,c2 = GetData()

    LDA(X,Y,c1,c2)

    plt.show()