subs = [1,1,1;1,2,1;3,4,2; 1,3,2];
vals = [1; 2; 3;7];
X = sptensor(subs,vals);
M = cp_als(X, 3, 'tol',1e-6,'maxiters',1000);
full(X);
full(M);