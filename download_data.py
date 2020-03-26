# data scraping source: https://github.com/koenig125/album-artwork-classification/blob/master/build_dataset.py

import argparse, json, csv, random, os, shutil, re, glob, sys
import urllib.request as req
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='albums/MUMU',
                    help="Directory with MUMU album images")
parser.add_argument('--mumu_metadata', default='albums/amazon_metadata_MuMu.json',
                    help="File with MuMu metadata")

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

def get_image_urls(mumu_metadata):
    """Retrieve all urls for album cover images in the MuMu dataset"""
    with open(mumu_metadata) as f:
        data = json.load(f)
    img_urls = dict()
    for entry in data:
        img_url = entry['imUrl']
        img_id = entry['amazon_id']
        img_urls[img_id] = img_url
    return img_urls

def download_images(img_urls, data_dir):
    """Download album cover images from the MuMu dataset to `data_dir`"""
    create_dir(data_dir)
    for img_id, img_url in tqdm(img_urls.items()):
        img_format = img_url[-4:]
        img_file = os.path.join(data_dir, img_id + img_format)
        req.urlretrieve(img_url, img_file)

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

def save(filenames, output_dir):
    """Save the images contained in `filenames` to the `output_dir`"""
    for filename in tqdm(filenames):
        image = Image.open(filename)
        image.save(os.path.join(output_dir, filename.split('/')[-1]))

if __name__ == '__main__':
    args = parser.parse_args()

    # Scrape album artwork if not already present
    if not os.path.isdir(args.data_dir):
        print("Dataset at {} not found. Scraping images now.".format(args.data_dir))
        img_urls = get_image_urls(args.mumu_metadata)
        download_images(img_urls, args.data_dir)

    # Get the filenames in the data directory
    # filenames = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.jpg')]

    print("Done building dataset")
