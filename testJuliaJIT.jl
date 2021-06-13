using StatsBase
using DelimitedFiles
using Statistics


fp = open("outputJlJit.txt", "a")
for run in 1:5
    write(fp, "Seria numer: "*string(run)*"\n")
    for n in 1000:200:4000
        write(fp, "Rozmiar danych: "*string(n)*"\n")
        # ==============================
		#Matrix computations
		# ==============================
        write(fp, "Matrix computations. \n")
        A = zeros(n,n)
        n2 = n*n
        Af = zeros((1,n2)) 

        
        # 1: Linear Congruential Generator
        # https://en.wikipedia.org/wiki/Linear_congruential_generator
        Af[1,1] = 3.141592653589793
        a = 1331
        c = 2.718281828459045
        m = 34564564


        # direct reference to an element of an array 
        function mG1(Af,a,c,m)
            for ii in 2:n2
                Af[1,ii] =  mod( (a * Af[1,ii-1] + c) , m )
            end
            return(Af)
        end
        tic = time()
        Af=mG1(Af,a,c,m)
        matrixGeneration1 = time() - tic; write(fp, "matrixGeneration1 = "*string(matrixGeneration1)*"\n")


        # element of an array used in the next computation copied to a variable
        function mG2(Af,a,c,m)
            for ii in 2:n2
                local lst = Af[1,ii-1]
                Af[1,ii] =  mod( (a * lst + c) , m )
            end
            return(Af)
        end
        tic = time()
        Af=mG2(Af,a,c,m)
        matrixGeneration2 = time() - tic; write(fp,"matrixGeneration2 = "*string(matrixGeneration2)*"\n")


        # 2: Matrices
        A = reshape(Af, n,n) 
        x = 1:1:n
        b = zeros((1,n)) 


        # matrix times vector
        function mTVT(A,x,)
            b = A*x
            return(b)
        end
        tic = time()
        b = mTVT(A,x)
        matrixTimesVectorTime = time() - tic; write(fp, "matrixTimesVectorTime = "*string(matrixTimesVectorTime)*"\n")


        # system of linear equations
        function sOLEST(A,b)
            x = A\b
            return(x)
        end
        tic = time()
        x = sOLEST(A,b)
        sysOfLinEqSolutionTime = time() - tic; write(fp,"sysOfLinEqSolutionTime = "*string(sysOfLinEqSolutionTime)*"\n")


        Asqr = zeros((n,n))

        # matrix squared
        function mST(A)
            Asqr = A*A 
            return(Asqr)
        end
        tic = time()
        Asqr=mST(A)
        matrixSquareTime = time() - tic; write(fp, "matrixSquareTime = "*string(matrixSquareTime)*"\n")


        B = zeros(n,n)
        C = zeros(n,n)
        D = zeros(n,n)

        # multiplication of random matrices generated using build-in tools
        function mRMT()
            B = randn(n,n)
            C = randn(n,n)
            D = B*C
            return(D)
        end
        tic = time()
        D=mRMT()
        matrixRandMulTime = time() - tic; write(fp, "matrixRandMulTime = "*string(matrixRandMulTime)*"\n")


        E = randn(n,n,3)

        # copying parts of 3D matrix (loop)
        function m3DCL(E)
            for j = 1:n, i = 1:n
                E[i,j,1] = E[i,j,2]
                E[i,j,3] = E[i,j,1]
                E[i,j,2] = E[i,j,3]
            end
            return(E)
        end
        tic = time()
        E=m3DCL(E)
        matrix3DCopyLoop = time() - tic; write(fp,"matrix3DCopyLoop = "*string(matrix3DCopyLoop)*"\n")


        # copying parts of 3D matrix (vectorised)
        E = randn(n,n,3)
        function m3DCV(E)
            E[:,:,1] = E[:,:,2]
            E[:,:,3] = E[:,:,1]
            E[:,:,2] = E[:,:,3]
            return(E)
        end
        tic = time()
        E=m3DCV(E)
        matrix3DCopyVect = time() - tic; write(fp, "matrix3DCopyVect = "*string(matrix3DCopyVect)*"\n")

        # ==============================
		#Numerical computations
		# ==============================
        write(fp, "Numerical computations. \n")
        ret = zeros(n,1)
 
        # 1: Trigonometric functions
        x = zeros(n,1)
        temp = -57

        val = pi / 180

        for i=1:n
            x[i] =  temp
            temp += 1
            if temp == 58
                temp = -57
            end
        end
        function nT(x,val)
            ret = sin.(x*val)
            ret += asin.(x*val)
            ret += cos.(x*val)
            ret += acos.(x*val)
            ret += tan.(x*val)
            ret += atan.(x*val)
            return(ret)
        end
        tic = time()
        ret=nT(x,val)
        numTrig = time() - tic; write(fp, "numTrig = "*string(numTrig)*"\n")
        print(ret[1]," ",string(run),"\n")


        # 2: Fibonacci number - iterative
        function fI()
            fPrev,f = (0,1)
            for i = 2:n/100 
                fPrev,f = (f, fPrev+f) 
            end
            return(f)
        end
        tic = time()
        f=fI()
        fibIterative = time() - tic; write(fp, "fibIterative = "*string(fibIterative)*"\n")
        print(f," ",string(run),"\n")
    end
end
close(fp)       
