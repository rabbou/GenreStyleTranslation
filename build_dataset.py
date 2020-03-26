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

parser.add_argument('--genres', nargs='+', default=['Metal', 'Dance', 'Classic'],
                    help='Genres as a list')

def has_face(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    return len(faces) > 0

def make_data(labels):
    paths = ['/Users/ruben/Box/CV Project/pytorch-CycleGAN-and-pix2pix/datasets/' + label + '/' for label in labels]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    with open('albums/MuMu_dataset_multi-label.csv', 'r', newline='') as csvf:
        readc = csv.reader(csvf, delimiter=',')
        for album in readc:
            path = "albums/MUMU/" + album[0] + ".jpg"
            if os.path.isfile(path):
                img = cv2.imread(path)
                if not has_face(img):
                    # leave out album covers with faces on them
                    img = misc.imresize(image, (256,256))
                    for i in range(len(labels)):
                        if labels[i] in album[5]:
                            shutil.copy(img, paths[i] + album[0] + ".jpg")

if __name__ == '__main__':
