
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
parser.add_argument('--genres', nargs='+', default=['Metal', 'Electronic', 'Classic'],
                    help='Genres as a list')
parser.add_argument('--labels_path', default='albums/MuMu_dataset_multi-label.csv',
                    help="File with MuMu album genres")
parser.add_argument('--album_path', default='albums/', help='where to save albums')
parser.add_argument('--data_dir', default='albums/MUMU/',
                    help="Directory with MUMU album images")
parser.add_argument('--clustering_method', default='kmeans',
                    help="which clustering method to use")

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

def get_album_genres(f_genres):
    """Retrieve genres for each album in the MuMu dataset"""
    album_genres = dict()
    with open(f_genres) as f:
        data = csv.reader(f)
        next(data) # skip header
        for row in data:
            img_id = row[0] # amazon_id
            genres = list(row[5].split(','))
            album_genres[img_id] = genres
    return album_genres

def has_face(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    return len(faces) > 0

def make_data(labels, data_dir, labels_path, album_path):
    album_genres = get_album_genres(labels_path)
    print('Iterating over all albums in original data:')
    for album in tqdm(album_genres.keys()):
        present_labels = [label for label in labels if label in album_genres[album]]
        # check if any of the labels are part of the target genres
        if present_labels:
            path = data_dir + album + ".jpg"
            if os.path.isfile(path):
                img = plt.imread(path)
                if not has_face(img):
                    # leave out album covers with faces on them
                    img = Image.fromarray(img).resize((256,256))
                    for i in range(len(present_labels)):
                        img.save(album_path + present_labels[i] + '/' + album + ".jpg")

def plot_cluster(files, label, method, nrows=10, ncols=10):
    plt.figure(figsize=(ncols, nrows))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(cv2.imread(files[i]))
        plt.axis('off')
    plt.axis('off')
    create_dir('../report_files')
    plt.savefig('report_files/' + label+'_'+method)

def train_test_clustering_split(labels, album_path, method='kmeans', display=True):
    dirs = [album_path + label + '/' for label in labels]
    for i in range(len(labels)):
        train_pth = album_path + labels[i] +'/train_' + method + '/'
        test_pth = album_path + labels[i] +'/test_' + method + '/'
        create_with_overwrite(train_pth)
        create_with_overwrite(test_pth)
        print(labels[i])
        all_img_fnames = list(glob.glob(f"{dirs[i]}/*.jpg"))
        # clustering
        n = 10
        if method == 'kmeans':
            clustered_fnames = cluster_kmeans(all_img_fnames, num_clusters=n, bs=25)
        # choice = np.random.randint(0,n-1)
        choice = np.argsort([len(clustered_fnames[i]) for i in range(len(clustered_fnames))])[n//2]
        print('Cluster size:', len(clustered_fnames[choice]))
        if display:
            plot_cluster(clustered_fnames[choice], labels[i], method)

        # split into 90/10 train_test
        train, test = train_test_split(clustered_fnames[choice], test_size=0.1, random_state=42)
        for album in train:
            shutil.copy(album, train_pth)
        for album in test:
            shutil.copy(album, test_pth)

if __name__ == '__main__':
    args = parser.parse_args()
    genres = args.genres
    paths = [args.album_path + label + '/' for label in genres]
    for path in paths:
        create_with_overwrite(path)
    make_data(genres, args.data_dir, args.labels_path, args.album_path)
    train_test_clustering_split(genres, args.album_path, args.clustering_method)
