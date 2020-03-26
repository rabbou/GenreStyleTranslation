# kmeans clustering using PCA inspired by https://github.com/baolingfeng/PSC2CODE/blob/02e40d33ad4c9b92d7772977bad3ad78aec08966/python/baseline_scripts/clusterImagesPCA.py

import argparse, json, csv, random, os, shutil, re, glob, sys
import urllib.request as req
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument('--genres', nargs='+', default=['Metal', 'Dance', 'Classic'],
                    help='Genres as a list')
parser.add_argument('--labels_path', default='albums/MuMu_dataset_multi-label.csv',
                    help="File with MuMu album genres")

def create_dir(dir_name):
    """Create directory with check not to overwrite existing directory"""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print("Warning: dir {} already exists".format(dir_name))

def create_with_overwrite(dir_name):
    """Create new directory or overwrite existing dir if it exists."""
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def has_face(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    return len(faces) > 0

def make_data(labels, labels_path):
    paths = ['albums/' + label + '/' for label in labels]
    for path in paths:
        create_with_overwrite(path)
    with open(labels_path, 'r', newline='') as csvf:
        readc = csv.reader(csvf, delimiter=',')
        for album in readc:
            path = "albums/MUMU/" + album[0] + ".jpg"
            if os.path.isfile(path):
                img = cv2.imread(path)
                if not has_face(img):
                    # leave out album covers with faces on them
                    img = Image.fromarray(img).resize((256,256))
                    for i in range(len(labels)):
                        if labels[i] in album[5]:
                            img.save(paths[i] + album[0] + ".jpg")

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

def test_train_clustering(labels, method):
    dirs = ['albums' + label + '/' for label in labels]
    for i in range(len(labels)):
        train_pth = 'albums/'+ labels[i] +'/train/'
        test_pth = 'albums/'+ labels[i] +'/test/'
        create_dir(train_pth)
        create_dir(test_pth)
        print(labels[i])
        all_img_fnames = list(glob.glob(f"{dirs[i]}/*.jpg"))
        if method == 'kmeans':
            clustered_fnames = cluster_kmeans(all_img_fnames, num_clusters=5, bs=10)
        elif method == 'em':
            clustered_fnames = cluster_kmeans(all_img_fnames, num_clusters=5, bs=10)
        mode = np.argmax([len(clustered_fnames[i]) for i in range(len(clustered_fnames))])
        print('Cluster size:', len(clustered_fnames[mode]))
        train, test = train_test_split(clustered_fnames[mode], test_size=0.2, random_state=42)
        for album in train:
            shutil.copy(album, train_pth)
        for album in test:
            shutil.copy(album, test_pth)

if __name__ == '__main__':
    args = parser.parse_args()
    make_data(args.genres, args.labels_path)
    test_train_clustering(['Metal', 'Dance', 'Classic'], cl='kmeans')
    build_gan_ds('Metal', 'Classic', 'metal2classic')
