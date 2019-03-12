import numpy as np
###### euclid distance only for OpenFace
###### cosine_similarity only for VGGFace

#calculate euclid distance between two vectors
#important to use numpy array as vector
#source: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy/21986532
def euclid_distance(a,b):
    distance = np.linalg.norm(a-b)

    return distance

#source: dotproduct divided by product of each norm of two vectors
# https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
def cosine_similarity(a,b):
    cos_sim = np.dot(a, b)/( np.linalg.norm(a) * np.linalg.norm(b))

    return cos_sim


#create test vectors with numpy
#it should be find with higher dimensional vectors, just need to created in numpy
a = np.array([1,0,1])
b = np.array([0,1,1])

#result must be sqrt(2), which is correct
print(euclid_distance(a,b))


#result should be 0.5 but 0.49999999 should be fine because of numeric approximation
print(cosine_similarity(a,b))
