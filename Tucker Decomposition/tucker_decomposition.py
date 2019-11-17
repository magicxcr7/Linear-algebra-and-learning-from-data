import numpy as np
import scipy
import scipy.io
import tensorly as tl
from tensorly.decomposition import tucker
from PIL import Image

core_r1=0
core_r2=0
core_r3=0

def read_data(name):
    data = scipy.io.loadmat(name)
    T = data['data']
    return T

def core_size(a,b,c):
    global core_r1, core_r2, core_r3
    core_r1, core_r2, core_r3 = a,b,c
    return np.zeros((core_r1,core_r2,core_r3))

def unfold(S,mode):
    x = S.shape[0]
    y = S.shape[1]
    z = S.shape[2]

    if mode == 0:
        M=np.zeros((x,y*z))
        for i in range(z):
            M[:,y*i:y*(i+1)] = S[:,:,i]
    elif mode == 1:
        M = tl.unfold(np.swapaxes(S,0,2),1)
    elif mode == 2:
        M = tl.unfold(np.swapaxes(S,0,1),2)
    else:
        pass
    return M

def fold(S,mode):
    global core_r1, core_r2, core_r3
    x,y,z = core_r1, core_r2, core_r3
    M=np.zeros((x,y,z))
    if mode == 0:
        for j in range(z):
            for i in range(y):
                M[:,i,j] = S[:,y*j + i]
    elif mode == 1:
        for j in range(z):
            for i in range(x):
                M[i,:,j]    = S[:,x*j + i]
    elif mode == 2:
        for j in range(y):
            for i in range(x):
                M[i,j,:]    = S[:,x*j + i]
    else:
        pass
    return M

def init_randn(T):
    A = np.random.randn(T.shape[0],core_r1)
    B = np.random.randn(T.shape[1],core_r2)
    C = np.random.randn(T.shape[2],core_r3)
    return A,B,C

def init_svd(T):

    A_svd=scipy.linalg.svd(unfold(T,0))
    A = A_svd[0][:,:core_r1]

    B_svd=scipy.linalg.svd(unfold(T,1))
    B = B_svd[0][:,:core_r2]

    C_svd=scipy.linalg.svd(unfold(T,2))
    C = C_svd[0][:,:core_r3]
    return A,B,C

def show_matrix_norm(A,B,C):
    print('matrix norm:    '
            'A:',np.linalg.norm(A),
            '    B',np.linalg.norm(B),
            '    C',np.linalg.norm(C))

def optimize_core(T,A,B,C,mode):
    if mode == 0:
        g = np.linalg.lstsq( A , np.dot(unfold(T,0),np.linalg.pinv(np.kron(C,B).T)) ,rcond=None)[0]
    elif mode == 1:
        g = np.linalg.lstsq( B , np.dot(unfold(T,1),np.linalg.pinv(np.kron(C,A).T)) ,rcond=None)[0]
    elif mode == 2:
        g = np.linalg.lstsq( C , np.dot(unfold(T,2),np.linalg.pinv(np.kron(B,A).T)) ,rcond=None)[0]
    else:
        pass
    return fold(g,mode)

def mode_test(T,A,B,C):
    for i in range(3): #test different 3 mode result , and it is the same
        mode = i
        g = optimize_core(T,A,B,C,mode)
        print('g-mode({}) '.format(i),np.linalg.norm(g))

def calculate_error(T,g,A,B,C,mode):


    if mode == 0:
        error = unfold(T,0) - (np.dot(A,unfold(g,0)).dot(np.kron(C,B).T))
    elif mode == 1:
        pass
    elif mode == 2:
        pass
    else:
        pass

    return np.linalg.norm(error)

def ordinary_tk(T,r1,r2,r3):
    for i in range(2):
        core, factors = tucker(T, rank=[r1,r2,r3])
        print('--------------')
        print(  i,'compare test:    '
            'factors[0]:',np.linalg.norm(factors[0]),
            '    factors[1]',np.linalg.norm(factors[1]),
            '    factors[2]',np.linalg.norm(factors[2]),
            '    core',np.linalg.norm(core))
        print('element sum'
            '               ',factors[0].sum(),
            '               ',factors[1].sum(),
            '            ',factors[2].sum(),
            '       ',core.sum())

        print('calculate||T - X|| :',calculate_error(T,core,factors[0],factors[1],factors[2],0))