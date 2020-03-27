
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
parser.add_argument('--album_path', default='albums/', help='where to save albums')
parser.add_argument('--data_dir', default='albums/MUMU/',
                    help="Directory with MUMU album images")

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
                img = cv2.imread(path)
                if not has_face(img):
                    # leave out album covers with faces on them
                    img = Image.fromarray(img).resize((256,256))
                    for i in range(len(present_labels)):
                        img.save(album_path + present_labels[i] + '/' + album + ".jpg")

    # with open(labels_path, 'r', newline='') as csvf:
    #     reader = csv.reader(csvf, delimiter=',')
    #     print('Iterating over all albums in original data:')
    #     for album in tqdm(reader, total=147296):
    #         # album = reader[i]
    #         path = "albums/MUMU/" + album[0] + ".jpg"
    #         if os.path.isfile(path):
    #             img = cv2.imread(path)
    #             if not has_face(img):
    #                 # leave out album covers with faces on them
    #                 img = Image.fromarray(img).resize((256,256))
    #                 for i in range(len(labels)):
    #                     if labels[i] in album[5]:
    #                         img.save(paths[i] + album[0] + ".jpg")

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
    paths = [args.album_path + label + '/' for label in genres]
    for path in paths:
        create_with_overwrite(path)
    make_data(genres, args.data_dir, args.labels_path, args.album_path)
    # train_test_clustering_split(genres)
