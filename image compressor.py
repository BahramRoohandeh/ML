import numpy as np
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros(X.shape[0] , dtype = int)
    for i in range(m):
        distance = []
        for j in range(K):
            d = np.sum((X[i]-centroids[j])**2)
            distance.append(d)
        idx[i] = np.argmin(distance)
        
    return idx
            
            
def compute_centroids(X, idx, K):
    m_k = np.zeros((K , X.shape[1]))
    for i in range(K):
        c_i = X[idx == i]
        m_k[i]  = np.mean(c_i, axis=0)

        
    return m_k
    
    
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    K = initial_centroids.shape[0]
    centroids = initial_centroids

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx
    

def kMeans_init_centroids(X, K):

    randidx = np.random.permutation(X.shape[0])

    centroids = X[randidx[:K]]
    
    return centroids    
    
image_adress = input('please enter impage path.name')
original_img = plt.imread(image_adress)
plt.imshow(original_img)


X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
K = input('please enter K')
K = int(K)
max_iters = 10
initial_centroids = kMeans_init_centroids(X_img, K)
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
idx = find_closest_centroids(X_img, centroids)

for i in range(X_img.shape[0]):
    X_img[i] = centroids[idx[i]]
    
X_recovered = np.reshape( X_img, original_img.shape) 

plt.imshow(X_recovered)
