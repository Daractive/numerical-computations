close all
clc

fp = fopen('output.txt', 'a');
for run = 1:5
		fprintf(fp, "Seria numer: %d\n", run);
		for n = 1000:200:4000
			fprintf(fp, "Rozmiar danych: %d\n", n);
				%==============================
				%Matrix computations
				%==============================
				fprintf(fp, "Matrix computations using a single-block memory allocation.\n");
				A = zeros(n) ;
                n2 = n*n ;
                
                
                % 1: Linear Congruential Generator
                % https://en.wikipedia.org/wiki/Linear_congruential_generator
                A(1) = 3.141592653589793 ;
                a = 1331 ;
                c = 2.718281828459045 ;
                m = 34564564 ;
                
                
                % direct reference to an element of an array
                tic
                for ii=2:n2
                   A(ii) =  mod( (a * A(ii-1) + c) , m ) ;
                end
                matrixGeneration1 = toc;
                fprintf(fp, "matrixGeneration1 = %f s\n", matrixGeneration1);
                
                
                % element of an array used in the next computation copied to a variable
                tic
                for ii=2:n2
                   last = A(ii-1) ;
                   A(ii) =  mod( (a * last + c) , m ) ;
                end
                matrixGeneration2 = toc;
                fprintf(fp, "matrixGeneration2 = %f s\n", matrixGeneration2);
                
                
                % array to vector and then back
                tic
                [p,q] = size(A) ;
                A = A(:) ;
                for ii=2:n2
				   last = A(ii-1) ;
                   A(ii) =  mod( (a * last + c) , m ) ;
                end
                A = reshape(A,p,q) ;
                matrixGeneration3 = toc;
                fprintf(fp, "matrixGeneration3 = %f s\n", matrixGeneration3);
                
                
                % 2: Matrices
                x = [1:n]' ;
                b = zeros(1,n) ;
                
                
                % matrix times vector
                tic
                b = A*x ;
                matrixTimesVectorTime = toc;
                fprintf(fp, "matrixTimesVectorTime = %f s\n", matrixTimesVectorTime);
                
                
                % system of linear equations
                tic
                x = A\b ;
                sysOfLinEqSolutionTime = toc;
                fprintf(fp, "sysOfLinEqSolutionTime = %f s\n", sysOfLinEqSolutionTime);
                
                
                Asqr = zeros(n) ;
                
                % matrix squared
                tic
                Asqr = A*A ;
                matrixSquareTime = toc;
                fprintf(fp, "matrixSquareTime = %f s\n", matrixSquareTime);
                
                
                B = zeros(n) ;
                C = zeros(n) ;
                D = zeros(n) ;
                
                % multiplication of random matrices generated using build-in tools
                tic
                B = rand (n);
                C = rand (n);
                D = B * C;
                matrixRandMulTime = toc;
                fprintf(fp, "matrixRandMulTime = %f s\n", matrixRandMulTime);
                
                
                E = rand (n,n,3);
                
                % copying parts of 3D matrix (loop)
                tic
                for j = 1:n
                    for i = 1:n
                        E(i,j,1) = E(i,j,2);
                        E(i,j,3) = E(i,j,1);
                        E(i,j,2) = E(i,j,3);
                    end
                end
                matrix3DCopyLoop = toc;
                fprintf(fp, "matrix3DCopyLoop = %f s\n", matrix3DCopyLoop);
                
                 
                E = rand (n,n,3);
                
                % copying parts of 3D matrix (vectorised)
                tic
                E(:,:,1) = E(:,:,2);
                E(:,:,3) = E(:,:,1);
                E(:,:,2) = E(:,:,3);
                matrix3DCopyVect = toc;
                fprintf(fp, "matrix3DCopyVect = %f s\n", matrix3DCopyVect);
                
                %==============================
				%Numerical computations
				%==============================
				ret = 0;


                % 1: Trigonometric functions
                x = zeros(n,1);
                temp = -57;
                
                val = pi / 180;
                
                for i=1:n
                   x(i) =  temp;
                   temp = temp + 1;
                       if temp == 58
                        temp = -57;
                       end
                end
                
                tic
                ret = sin(x*val);
                ret = ret + asin(x*val);
                ret = ret + cos(x*val);
                ret = ret + acos(x*val);
                ret = ret + tan(x*val);
                ret = ret + atan(x*val);
                numTrig = toc;
                fprintf(fp, "numTrig = %f s\n", numTrig);
                
                % 2: Fibonacci number - iterative
                tic
                fn = [1 0];
                f = 0;
                 
                for i = (1:n/100)
                    fn(2) = f;
                    f = sum(fn);
                    fn(1) = fn(2);
                end
                fibIterative = toc;
                fprintf(fp, "fibIterative = %f s\n", fibIterative);
                
                % 3: Fibonacci number - recursive
                
                tic
                  f=fibRecur(n/100);
                fibRecursive = toc;
                fprintf(fp, "fibRecursive = %f s\n", fibRecursive);
		end
        
		for n = 1000:600:10000
			fprintf(fp, "Rozmiar danych: %d\n", n);
			    %==============================
				%GARCH log-likelihood
				%==============================
                load('data.dat');
                y=data;
                y=y(1:n);
                
                N=length(y);
                o=0.001;
                a=0.85;
                b=0.01;
                
                lik=0;
                y2=y.^2;
                v=var(y);
                
                tic
                for temp = 1:100
                    lik=likelihood(o,a,b,v,y2,N);
                end
                numGARCH=toc;
                fprintf(fp, "numGARCH = %f s\n", numGARCH);
		end
	fprintf(fp, "\n \n"); 
end
fclose(fp);