import math 
import time
import numpy as np
import numba as nb



fp = open("outputNUMBA.txt", "a")
for run in range(1,6) :
    fp.write('Run: '+str(run)+'\n')
    for n in range(1000, 4200, 200) :
        fp.write('Data size: '+str(n)+'\n')
        #==============================
		#Matrix computations NUMBA
		#==============================
        fp.write('Matrix computations NUMBA. \n')
        A = np.zeros((n,n))
        n2 = n*n
        Af = np.zeros((1,n2))


        # 1: Linear Congruential Generator
        # https://en.wikipedia.org/wiki/Linear_congruential_generator
        Af[0,0] = 3.141592653589793 
        a = 1331 
        c = 2.718281828459045 
        m = 34564564 

        # https://stackoverflow.com/questions/46147670/best-way-to-iterate-over-an-array-that-uses-its-own-output
        tic = time.time()

        # numba compilation (v 0.48) doesn't work on lists    -> 
        #          https://stackoverflow.com/questions/53314071/turn-off-list-reflection-in-numba
        #Aflist = Af.tolist()[0] ;
        #@nb.jit # (nopython=True)
        #def func(Aflist):
        #    for ii in range(1,n2):
        #        Aflist[ii] = np.mod( (a * Aflist[ii-1] + c) , m )
        #    return Aflist


        # direct reference to an element of an array
        @nb.jit(nopython=True)
        def func(Af):
            for ii in range(1,n2):
                Af[0,ii] = np.mod( (a * Af[0,ii-1] + c) , m )
            return Af

        Af = func(Af)
        matrixGeneration_NUMBA = time.time() - tic; fp.write('matrixGeneration_NUMBA = '+str(matrixGeneration_NUMBA)+'\n')


        # element of an array used in the next computation copied to a variable
        tic = time.time()

        @nb.jit(nopython=True)
        def func(Af):
            last = Af[0,0]
            for ii in range(1,n2):
                Af[0,ii] = last = np.mod( (a * Af[0,ii-1] + c) , m )
            return Af

        Af = func(Af)
        matrixGeneration2_NUMBA = time.time() - tic; fp.write('matrixGeneration2_NUMBA = '+str(matrixGeneration2_NUMBA)+'\n')


        # 2: Matrices
        A = Af.reshape(A.shape)
        x = np.float64(np.transpose(np.arange(1,n+1)))
        b = np.transpose(np.zeros((n)))


        # matrix times vector
        tic = time.time()

        @nb.jit("float64[:](float64[:,:],float64[:],float64[:])", nopython=True)
        def func2(A, x, b):
            b = A@x
            return b 
            
        b = func2(A, x, b)
        matrixTimesVectorTime_NUMBA = time.time() - tic; fp.write('matrixTimesVectorTime_NUMBA = '+str(matrixTimesVectorTime_NUMBA)+'\n')


        # system of linear equations
        tic = time.time()

        @nb.jit("float64[:](float64[:,:],float64[:],float64[:])", nopython=True)
        def func3(A, b, x):
            x = np.linalg.solve(A,b)
            return x
            
        x = func3(A, b, x)
        sysOfLinEqSolutionTime_NUMBA = time.time() - tic; fp.write('sysOfLinEqSolutionTime_NUMBA = '+str(sysOfLinEqSolutionTime_NUMBA)+'\n')


        Asqr = np.zeros((n,n))

        # matrix squared
        tic = time.time()

        @nb.jit("float64[:,:](float64[:,:],float64[:,:])", nopython=True)
        def func4(A, Asqr):
            Asqr = A@A
            return Asqr
            
        Asqr = func4(A, Asqr)
        matrixSquareTime_NUMBA = time.time() - tic; fp.write('matrixSquareTime_NUMBA = '+str(matrixSquareTime_NUMBA)+'\n')


        B = np.zeros((n,n))
        C = np.zeros((n,n))
        D = np.zeros((n,n))

        # multiplication of random matrices generated using build-in tools
        tic = time.time()

        @nb.jit("float64[:,:](float64[:,:],float64[:,:],float64[:,:])", nopython=True)
        def func5(B, C, D):
            B = np.random.rand(n,n)
            C = np.random.rand(n,n)
            D = np.dot(B,C)
            return D

        D = func5(B, C, D)
        matrixRandMulTime_NUMBA = time.time() - tic; fp.write('matrixRandMulTime_NUMBA = '+str(matrixRandMulTime_NUMBA)+'\n')


        E = np.random.rand(n,n,3)

        # copying parts of 3D matrix (loop)
        tic = time.time()
        @nb.jit("void(float64[:,:,:])", nopython=True)
        def func6(E):
            for i in range(n):
                for j in range(n):
                    E[i,j,0] = E[i,j,1]
                    E[i,j,2] = E[i,j,0]
                    E[i,j,1] = E[i,j,2]
        E = func6(E)
        matrix3DCopyLoop_NUMBA = time.time() - tic; fp.write('matrix3DCopyLoop_NUMBA = '+str(matrix3DCopyLoop_NUMBA)+'\n')


        E = np.random.rand(n,n,3)

        # copying parts of 3D matrix (vectorised)
        tic = time.time()
        @nb.jit("void(float64[:,:,:])", nopython=True)
        def func7(E):
            E[:,:,0] = E[:,:,1]
            E[:,:,2] = E[:,:,0]
            E[:,:,1] = E[:,:,2]
        E = func7(E)
        matrix3DCopyVect_NUMBA = time.time() - tic; fp.write('matrix3DCopyVect_NUMBA = '+str(matrix3DCopyVect_NUMBA)+'\n')

        #==============================
		#Numerical computations NUMBA
		#==============================
        fp.write('Numerical computations NUMBA. \n')
        ret = np.zeros((n,1))

        # 1: Trigonometric functions
        x = np.zeros((n,1))
        temp = -57

        val = math.pi / 180

        for i in np.arange(1,n) :
            x[i] =  temp
            temp += 1
            if temp == 58 :
                temp = -57

        tic = time.time()

        @nb.jit("float64[:,:](float64[:,:],float64,float64[:,:])", nopython=True)
        def func8(x, val, ret):
            ret = np.sin(x*val)
            ret += np.arcsin(x*val)
            ret += np.cos(x*val)
            ret += np.arccos(x*val)
            ret += np.tan(x*val)
            ret += np.arctan(x*val)
            return ret
            
        ret = func8(x, val, ret)
        numTrig_NUMBA = time.time() - tic; fp.write('numTrig_NUMBA = '+str(numTrig_NUMBA)+'\n')
        print(str(ret)+'\n')
            

        # 2: Fibonacci number - iterative
        tic = time.time()

        fPrev = 0
        f = 1

        @nb.jit("int64(float64,float64)", nopython=True)
        def func9(fPrev, f):
            for num in range(1, int(n/100)):
                fPrev, f = f, fPrev + f
            return f
            
        f = func9(fPrev, f)
        fibIterative_NUMBA = time.time() - tic; fp.write('fibIterative_NUMBA = '+str(fibIterative_NUMBA)+'\n')
        print(str(f)+'\n')


        # 3: Fibonacci number - recursive
        tic = time.time()

        @nb.jit("int64(float64)", nopython=True)
        def fibRecur(n):
            if n <= 2:
                return 1
            else:
                return fibRecur(n-1) + fibRecur(n-2)
            
        f = fibRecur(int(n/100))
        fibRecursive_NUMBA = time.time() - tic; fp.write('fibRecursive_NUMBA = '+str(fibRecursive_NUMBA)+'\n')
        print(str(f)+'\n')

    for n in range(1000, 10600, 600) :
        fp.write('Data size: '+str(n)+'\n')
        #==============================
		#GARCH log-likelihood NUMBA
		#==============================
        fp.write('GARCH log-likelihood NUMBA. \n')
        y = np.loadtxt('data.dat', delimiter = ',')
        y = y[:n]

        N = len(y)
        o, a, b = 0.001, 0.85, 0.01
        y2 = np.square(y)

        @nb.jit("float64(float64,float64,float64,float64,float64[:],float64)", nopython=True)
        def fun(hh,o,a,b,y2,N):
            lik = 0.0
            h = hh
            for i in range(1,N):
                h=o+a*y2[i-1]+b*h
                lik += np.log(h) + y2[i]/h
            return(lik)
            
        tic = time.time()
        for i in range(0, 100):
            lik = fun(np.var(y), o, a, b, y2, N)
        numGARCH_NUMBA = time.time() - tic; fp.write('numGARCH = '+str(numGARCH_NUMBA)+'\n')
        print(str(lik)+'\n')
fp.close()
