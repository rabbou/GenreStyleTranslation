# source https://github.com/baolingfeng/PSC2CODE/blob/02e40d33ad4c9b92d7772977bad3ad78aec08966/python/baseline_scripts/clusterImagesPCA.py

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import numpy as np
import cv2, random
from PIL import Image
from collections import defaultdict

def combine_images_into_tensor(img_fnames, size=256):
    # Initialize the tensor
    tensor = np.zeros((len(img_fnames), size * size))
    for i, fname in enumerate(img_fnames):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        tensor[i] = img.reshape(size * size)
    return tensor

def get_pca_reducer_incremental(tr_tensor, n_comp=10, bs=10):
    # Apply Incremental PCA on the training images
    pca = IncrementalPCA(n_components=n_comp, batch_size=bs)
    for i in range(0, len(tr_tensor), bs):
        print(f"fitting {i//bs} th batch")
        pca.partial_fit(tr_tensor[i:i+bs, :])
    return pca

def cluster_kmeans(all_img_fnames, num_clusters=4, bs=10):
    # Select images at random for PCA
    random.shuffle(all_img_fnames)
    tr_img_fnames = all_img_fnames[:400]

    # Flatten and combine the images
    tr_tensor = combine_images_into_tensor(tr_img_fnames)

    # Perform PCA
    print("Learning PCA...")
    n_comp = 10
    pca = get_pca_reducer_incremental(tr_tensor, n_comp, bs)

    # Transform images in batches
    print("applying PCA transformation")
    points = np.zeros((len(all_img_fnames), n_comp))
    batch_size = 50
    for i in range(0, len(all_img_fnames), batch_size):
        print(f"Transforming {i//25} th batch")
        batch_fnames = all_img_fnames[i:i+batch_size]
        all_tensor = combine_images_into_tensor(batch_fnames)
        points[i:i+batch_size] = pca.transform(all_tensor)

    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)

    # Organize image filenames based on the obtained clusters
    cluster_fnames = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        cluster_fnames[label].append(all_img_fnames[i])

    return cluster_fnames
