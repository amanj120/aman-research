%idxs = [1 1 1; 2 2 2; 1 3 4; 2 2 3];
%vals = [2; 6; 1; 4];
%X = sptensor(idxs,vals);
%X(4,5,6) = 11;
rng(3);
indices = int32(csvread("indexes.csv"));
values = csvread("values.csv");
%disp(length(indices));
%disp(length(values));
len = length(indices);
T1 = sptensor((indices + 1),values);
%disp(T1.size);
%[M1,guess1] = cp_als(T1, 100, 'tol',1e-6,'maxiters',1000);
M1alt = cp_als(T1, 100, 'tol',1e-6,'maxiters',1000);
fiber = a{:} * 1000; %the document fiber, shoudl contain clusterings
lambdas = M1alt.lambda;
csvwrite('fiber.csv',fiber);
csvwrite('lambdas.csv',lambdas);
