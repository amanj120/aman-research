import numpy
import tensorflow

nd = 3 #number of dimensions
dimensions = (3,3,3) #has to be a tuple
# according to kolda-bader the typical rank of this should be 4

#help(numpy.random.random_sample) #get information about creating a random ndarray
#numpy.random.randint(low, high, size) #random sample of integers

numpy.random.randint(10,99,dimensions)

def base(v1,v2,v3):
	ret = numpy.zeros((len(v1),len(v2),len(v3)))
	return ret;

def expand(v1,v2,v3):
	b = base(v1,v2,v3)
	for i in range(len(v1)):
		for j in range(len(v2)):
			for k in range(len(v3)):
				b[i][j][k] = v1[i] * v2[j] * v3[k]
	return b

# l = ([1,2],[0,1,1,0],[2,3,4])
# arr = expand(l)
# cp = tensorly.decomposition.parafac(arr)
# cp.factors = [array([[ -7.61577311],[-15.23154621]]), array([[0.], [0.70710678],[0.70710678],[0.]]), array([[-0.37139068],[-0.55708601],[-0.74278135]])]

def cp_als(X, r):
	factors = []
	for i in range(nd):
		factors[i] = numpy.ones(r, dimensions[i])
	for repeats in range(10):
		for n in range(nd):
			#go through all dimensions

'''
array([[[  0.,   0.,   0.],
        [  0.,   0.,   0.]],

       [[ 16.,  12.,  10.],
        [ 64.,  48.,  40.]],

       [[ 24.,  18.,  15.],
        [ 96.,  72.,  60.]],

       [[ 56.,  42.,  35.],
        [224., 168., 140.]]]) 
'''

indices = [[1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2],[2,0,0],[2,0,1],[2,0,2],[2,1,0],[2,1,1],[2,1,2],[3,0,0],[3,0,1],[3,0,2],[3,1,0],[3,1,1],[3,1,2]]
values = [16.0, 12.0, 10.0, 64.0, 48.0, 40.0, 24.0, 18.0, 15.0, 96.0, 72.0, 60.0, 56.0, 42.0, 35.0, 224.0, 168.0, 140.0]
shape = [4,2,3]

st = tensorflow.SparseTensor(indices=indices, values=values, dense_shape=shape)



