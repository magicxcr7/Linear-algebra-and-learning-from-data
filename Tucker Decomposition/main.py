import tucker_decomposition as t
import numpy as np
from tensorly.decomposition import tucker

if __name__ == '__main__':
    T = t.read_data('data.mat')     #size P,Q,R
    r1,r2,r3 = 20,10,5
    g = t.core_size(r1,r2,r3)    #define core size

    A,B,C = t.init_randn(T) #size A[P,r1] B[Q,r2] C[R,r3]
    t.show_matrix_norm(A,B,C)

    # mode
    # 0:aremin(g)|| T - Ag(CxB)^t||
    # 1:aremin(g)|| T - Bg(CxA)^t||
    # 2:aremin(g)|| T - Cg(BxA)^t||
    
    #t.mode_test(T,A,B,C)      #test different 3 mode result , and that is the same

    mode = 0
    g = t.optimize_core(T,A,B,C,mode)
    print('g ',np.linalg.norm(g))
    print('calculate||T - X|| :',t.calculate_error(T,g,A,B,C,mode))    #calculate||T - X||



    print('\n\n----for svd init method:')
    A,B,C = t.init_svd(T)   #size A[P,r1] B[Q,r2] C[R,r3]
    t.show_matrix_norm(A,B,C)
    g = t.optimize_core(T,A,B,C,mode)
    print('g ',np.linalg.norm(g))
    print('calculate||T - X|| :',t.calculate_error(T,g,A,B,C,mode))    #calculate||T - X||

    

    print('\n\n----compare to ordinary tucker decomposition:')
    t.ordinary_tk(T,r1,r2,r3)      #compare Q1



        