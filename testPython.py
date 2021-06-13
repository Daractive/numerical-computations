import math 
import time
import numpy as np


        
fp = open("outputPY.txt", "a")
for run in range(1,6) :
    fp.write('Run: '+str(run)+'\n')
    for n in range(1000, 4200, 200) :
        fp.write('Data size: '+str(n)+'\n')
        #==============================
		#Matrix computations
		#==============================
        fp.write('Matrix computations. \n')
        A = np.zeros((n,n))
        n2 = n*n
        Af = np.zeros((1,n2))


        # 1: Linear Congruential Generator
        # https://en.wikipedia.org/wiki/Linear_congruential_generator
        Af[0,0] = 3.141592653589793
        a = 1331
        c = 2.718281828459045
        m = 34564564


        # direct reference to an element of an array
        tic = time.time()
        for ii in np.arange(1,n2):
            Af[0,ii] =  np.mod( (a * Af[0,ii-1] + c) , m ) 
        matrixGeneration1 = time.time() - tic; fp.write('matrixGeneration1 = '+str(matrixGeneration1)+'\n') 


        # element of an array used in the next computation copied to a variable
        tic = time.time()
        last = Af[0,0]
        for ii in np.arange(1,n2):
            Af[0,ii] = last = np.mod( (a * last + c) , m ) 
        matrixGeneration2 = time.time() - tic; fp.write('matrixGeneration2 = '+str(matrixGeneration2)+'\n') 


        # very bad approach
        # using concatenation
        # y = np.empty(Af.size)
        # y = np.concatenate(([n2], y[:-1])) * 2 
        # tic = time.time()
        # for ii in np.arange(1,n2):
        #     y = np.concatenate(([Af[0,0]], np.mod( (a * y[:-1] + c) , m )  ))
        # matrixGeneration3 = time.time() - tic; f.write('matrixGeneration3 = '+str(matrixGeneration3)+'\n')


        # using python lists + direct reference to an element of an array
        # https://stackoverflow.com/questions/46147670/best-way-to-iterate-over-an-array-that-uses-its-own-output
        tic = time.time()
        Aflist = Af.tolist()
        for ii in range(1, len(Aflist[0])):
            Aflist[0][ii] =  np.mod( (a * Aflist[0][ii-1] + c) , m ) 
        Af = np.asarray(Aflist[0]) 
        matrixGeneration4 = time.time() - tic; fp.write('matrixGeneration4 = '+str(matrixGeneration4)+'\n')


        # using python lists + direct iteration over the list
        # https://stackoverflow.com/questions/46147670/best-way-to-iterate-over-an-array-that-uses-its-own-output
        tic = time.time()
        Aflist = Af.tolist() 
        for ii, (Aprevii) in enumerate(Aflist[:1], 1):
            Aflist[ii] =  np.mod( (a * Aprevii + c) , m ) 
        Af = np.asarray(Aflist) 
        matrixGeneration5 = time.time() - tic; fp.write('matrixGeneration5 = '+str(matrixGeneration5)+'\n')


        # 2: Matrices
        A = Af.reshape(A.shape)
        x = np.transpose(np.arange(1,n+1))
        b = np.transpose(np.zeros((1,n)))


        # matrix times vector
        tic = time.time()
        b = A@x
        matrixTimesVectorTime = time.time() - tic; fp.write('matrixTimesVectorTime = '+str(matrixTimesVectorTime)+'\n')


        # system of linear equations
        tic = time.time()
        x = np.linalg.solve(A,b)
        sysOfLinEqSolutionTime = time.time() - tic; fp.write('sysOfLinEqSolutionTime = '+str(sysOfLinEqSolutionTime)+'\n')


        Asqr = np.zeros((n,n))

        # matrix squared
        tic = time.time()
        Asqr = A@A
        matrixSquareTime = time.time() - tic; fp.write('matrixSquareTime = '+str(matrixSquareTime)+'\n')


        B = np.zeros((n,n))
        C = np.zeros((n,n))
        D = np.zeros((n,n))

        # multiplication of random matrices generated using build-in tools
        tic = time.time()
        B = np.random.rand(n,n)
        C = np.random.rand(n,n)
        D = np.dot(B,C)
        matrixRandMulTime = time.time() - tic; fp.write('matrixRandMulTime = '+str(matrixRandMulTime)+'\n')


        E = np.random.rand(n,n,3)

        # copying parts of 3D matrix (loop)
        tic = time.time()
        for i in range(n):
            for j in range(n):
                E[i,j,0] = E[i,j,1]
                E[i,j,2] = E[i,j,0]
                E[i,j,1] = E[i,j,2]
        matrix3DCopyLoop = time.time() - tic; fp.write('matrix3DCopyLoop = '+str(matrix3DCopyLoop)+'\n')


        E = np.random.rand(n,n,3)

        # copying parts of 3D matrix (vectorised)
        tic = time.time()
        E[:,:,0] = E[:,:,1]
        E[:,:,2] = E[:,:,0]
        E[:,:,1] = E[:,:,2]
        matrix3DCopyVect = time.time() - tic; fp.write('matrix3DCopyVect = '+str(matrix3DCopyVect)+'\n')

        #==============================
		#Numerical computations
		#==============================
        fp.write('Numerical computations. \n')
        n2 = n*n
        ret = 0

        # 1: Trigonometric functions
        x = np.zeros((n2,1))
        temp = -57

        val = math.pi / 180

        for i in range(0,n2) :
            x[i] =  temp
            temp += 1
            if temp == 58 :
                temp = -57

        tic = time.time()
        ret += np.sin(x*val)
        ret += np.arcsin(x*val)
        ret += np.cos(x*val)
        ret += np.arccos(x*val)
        ret += np.tan(x*val)
        ret += np.arctan(x*val)
        numTrig = time.time() - tic; fp.write('numTrig = '+str(numTrig)+'\n')
        print(str(ret)+'\n')
            

        # 2: Fibonacci number - iterative
        tic = time.time()
        fPrev = 0
        f = 1
        for i in range(1, int(n/100)):
            fPrev, f = f, fPrev + f
        fibIterative = time.time() - tic; fp.write('fibIterative = '+str(fibIterative)+'\n')
        print(str(f)+'\n')


        # 3: Fibonacci number - recursive
        def fibRecur(n):
            if n <= 2:
                return 1
            else:
                return fibRecur(n-1) + fibRecur(n-2)
            
        tic = time.time()
        f = fibRecur(int(n/100))
        fibRecursive = time.time() - tic; fp.write('fibRecursive = '+str(fibRecursive)+'\n')
        print(str(f)+'\n')

    for n in range(1000, 10600, 600) :
        fp.write('Rozmiar danych: '+str(n)+'\n')
        #==============================
		#GARCH log-likelihood
		#==============================
        fp.write('Data size. \n')
        y = np.loadtxt('data.dat', delimiter = ',')
        y = y[:n]

        N = len(y)
        o, a, b = 0.001, 0.85, 0.01
        y2 = np.square(y)

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
        numGARCH = time.time() - tic; fp.write('numGARCH = '+str(numGARCH)+'\n')
        print(str(lik)+'\n')
fp.close()