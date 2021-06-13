using StatsBase
using DelimitedFiles
using Statistics


fp = open("output.txt", "a")
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
        tic = time()
        for ii in 2:n2
            Af[1,ii] =  mod( (a * Af[1,ii-1] + c) , m )
        end
        matrixGeneration1 = time() - tic; write(fp, "matrixGeneration1 = "*string(matrixGeneration1)*"\n")


        # element of an array used in the next computation copied to a variable
        tic = time()
        for ii in 2:n2
            last = Af[1,ii-1]
            Af[1,ii] =  mod( (a * last + c) , m )
        end
        matrixGeneration2 = time() - tic; write(fp,"matrixGeneration2 = "*string(matrixGeneration2)*"\n")


        # 2: Matrices
        A = reshape(Af, n,n) 
        x = 1:1:n
        b = zeros((1,n)) 


        # matrix times vector
        tic = time()
        b = A*x 
        matrixTimesVectorTime = time() - tic; write(fp, "matrixTimesVectorTime = "*string(matrixTimesVectorTime)*"\n")


        # system of linear equations
        tic = time()
        x = A\b
        sysOfLinEqSolutionTime = time() - tic; write(fp,"sysOfLinEqSolutionTime = "*string(sysOfLinEqSolutionTime)*"\n")


        Asqr = zeros((n,n))

        # matrix squared
        tic = time()
        Asqr = A*A 
        matrixSquareTime = time() - tic; write(fp, "matrixSquareTime = "*string(matrixSquareTime)*"\n")


        B = zeros(n,n)
        C = zeros(n,n)
        D = zeros(n,n)

        # multiplication of random matrices generated using build-in tools
        tic = time()
        B = randn(n,n)
        C = randn(n,n)
        D = B*C
        matrixRandMulTime = time() - tic; write(fp, "matrixRandMulTime = "*string(matrixRandMulTime)*"\n")


        E = randn(n,n,3)

        # copying parts of 3D matrix (loop)
        tic = time()
        for j = 1:n, i = 1:n
			E[i,j,1] = E[i,j,2]
			E[i,j,3] = E[i,j,1]
			E[i,j,2] = E[i,j,3]
        end
        matrix3DCopyLoop = time() - tic; write(fp,"matrix3DCopyLoop = "*string(matrix3DCopyLoop)*"\n")


        # copying parts of 3D matrix (vectorised)
        E = randn(n,n,3)

        tic = time()
        E[:,:,1] = E[:,:,2]
        E[:,:,3] = E[:,:,1]
        E[:,:,2] = E[:,:,3]
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

        tic = time()
        ret = sin.(x*val)
        ret += asin.(x*val)
        ret += cos.(x*val)
        ret += acos.(x*val)
        ret += tan.(x*val)
        ret += atan.(x*val)
        numTrig = time() - tic; write(fp, "numTrig = "*string(numTrig)*"\n")
        print(ret[1]," ",string(run),"\n")


        # 2: Fibonacci number - iterative
        tic = time()
        fPrev,f = (0,1)
        for i = 2:n/100 
            fPrev,f = (f, fPrev+f) 
        end
        fibIterative = time() - tic; write(fp, "fibIterative = "*string(fibIterative)*"\n")
        print(f," ",string(run),"\n")

        # 3: Fibonacci number - recursive 
        function fibRecur(n)
            if n <= 2
                1
            else
                fibRecur(n-1)+fibRecur(n-2)
            end
        end

        tic = time()
        f = fibRecur(n/100)
        fibRecursive = time() - tic; write(fp, "fibRecursive = "*string(fibRecursive)*"\n")
        print(f," ",string(run),"\n")
    end

    for n in 1000:600:10000
        write(fp, "Rozmiar danych: "*string(n)*"\n")
        # ==============================
		#GARCH log-likelihood
		# ==============================
        write(fp, "GARCH log-likelihood. \n")
        y = readdlm("data.dat")
        y=y[1:n,1]

        N=length(y)
        o,a,b=[0.001,0.85,0.01]
        y2=y.^2
        v=var(y)

        function fun(o, a, b, h, y2, N)
            local lik = 0.0
            for i in 2:N
                h = o+a*y2[i-1]+b*h
                lik += log(h)+y2[i]/h
            end
            return(lik)
        end


        tic = time()
        for n in 1:100
            lik=fun(o,a,b,v,y2,N)
        end
        numGARCH = time() - tic; write(fp, "numGARCH = "*string(numGARCH)*"\n")
    end
end
close(fp)       
