
import argparse, json, csv, random, os, shutil, re, glob, sys
import urllib.request as req
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from clustering.k_means import *

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

def train_test_clustering_split(labels, method='kmeans'):
    dirs = ['albums/' + label + '/' for label in labels]
    for i in range(len(labels)):
        train_pth = 'albums/'+ labels[i] +'/train/'
        test_pth = 'albums/'+ labels[i] +'/test/'
        create_dir(train_pth)
        create_dir(test_pth)
        print(labels[i])
        all_img_fnames = list(glob.glob(f"{dirs[i]}/*.jpg"))
        # clustering
        if method == 'kmeans':
            clustered_fnames = cluster_kmeans(all_img_fnames, num_clusters=5, bs=10)
        mode = np.argmin([len(clustered_fnames[i]) for i in range(len(clustered_fnames))])
        print('Cluster size:', len(clustered_fnames[mode]))

        # split into 90/10 train_test
        train, test = train_test_split(clustered_fnames[mode], test_size=0.1, random_state=42)
        for album in train:
            shutil.copy(album, train_pth)
        for album in test:
            shutil.copy(album, test_pth)

if __name__ == '__main__':
    args = parser.parse_args()
    genres = args.genres
    make_data(genres, args.labels_path)
    train_test_clustering_split(genres)
