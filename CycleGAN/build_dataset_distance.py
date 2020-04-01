
import shutil
from skimage import measure
import cv2
import numpy as np
import scipy.cluster.hierarchy as sci
import os
import random
import matplotlib.pyplot as plt


def dist_mat(directory):
    try:
        names = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory, i))]
        num = len(names)
        distances = np.zeros((num, num))
        dims = (300, 300)
        for i in range(0, num):
            im1 = cv2.imread(directory + names[i], 0)
            if im1.shape != dims:
                im1 = cv2.resize(im1, dims)
            for j in range(i, num):
                im2 = cv2.imread(directory + names[j], 0)
                if im2.shape != dims:
                    im2 = cv2.resize(im2, dims)
                s = measure.compare_ssim(im1, im2)
                distances[i, j] = 1-s #### switching to distance metric for upgam
        return distances, names
    except Exception as e:
        print(e)

def distances_ssim(directory, target, si, distance_flag):
    try:
        names = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory, i))]
        num = len(names)
        distances = []
        im1 = cv2.imread(target, 0)
        dims = im1.shape
        for i in range(0, num):
            im2 = cv2.imread(directory + names[i], 0)
            if im2.shape != dims:
                im2 = cv2.resize(im2, dims)
            s = measure.compare_ssim(im1, im2)
            if distance_flag is True:
                distances.append([names[i], 1-s])
            else:
                distances.append([names[i], s])
        distances.sort(key=lambda distances: distances[1])
        return distances[num - si:]
    except Exception as e:
        print(e)

def move_subsample(distances, names, directory):
    try:
        l = len(names)
        sums = np.zeros(l)
        total_ssi = 0
        total_pics = 0
        files = []
        num = 0

        for i in range(0, l):
            sums[i] = np.sum(distances[:, i]) + np.sum(distances[i, :])

        maxes = np.argsort(sums)[-15:]
        subpop = []
        temp = np.zeros(l)

        for j in maxes:
            temp[:j] = distances[:j, j]
            temp[j:] = distances[j, j:]
            sub_max = np.argsort(temp)[-10:]
            subpop.extend(sub_max)
            for x in sub_max:
                total_ssi += temp[x]
                total_pics +=1

        subpop = list(dict.fromkeys(subpop))
        train_size = len(subpop)//4 * 3
        random.shuffle(subpop)

        if not os.path.isfile(directory + 'train_distance'):
            os.mkdir(directory + 'train_distance')
        for i in subpop[:train_size]:
            shutil.copy(directory + 'albums/' + names[i], directory + 'train_distance/' + names[i])
            if num < 100:
                files.append(directory + 'albums/' + names[i])
                num +=1

        if not os.path.isfile(directory + 'test_distance'):
            os.mkdir(directory + 'test_distance')
        for y in subpop[train_size:]:
            shutil.copy(directory + 'albums/' +names[y], directory + 'test_distance/' + names[y])
            if num < 100:
                files.append(directory + 'albums/' + names[y])
                num +=1

        print(total_ssi/total_pics)
        return files
    except Exception as e:
        print(e)

def move_sample(distances, directory, new_name):
    try:
        files = []
        num = 0
        for i, _ in distances:
            if os.path.isfile(directory + i):
                shutil.copy(directory + i, new_name + i)
                # python train.py --dataroot ./datasets/techno2jazz --name testTech2Jazz --model cycle_gan --gpu_ids -1
                if num < 100:
                    files.append(directory + i)
        return files
    except Exception as e:
        print(e)

def plot_cluster(files, label, method, nrows=10, ncols=10):
    plt.figure(figsize=(ncols, nrows))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(cv2.imread(files[i]))
        plt.axis('off')
    plt.axis('off')
    plt.show()
    #plt.savefig('report_files/' + label+'_'+method)

if __name__ == '__main__':
    try:

        genre = 'metal'
        if os.path.isfile('./' + genre + '/distances.npy'):
            dataset = np.load('./' + genre + '/distances.npy')
            names = np.load('./' + genre + '/order.npy', encoding='bytes')
            if isinstance(type(names[0]), str):
                names = [x.decode("utf-8") for x in names]
            files = move_subsample(dataset, names, './'+genre+ '/')
            plot_cluster(files, 'metal', 'SSIM Distance')
        else:
            print('Distance matrix does not exit, generating new one')
            print('This is going to take a really long time.')
            dist, nams = dist_mat('./' + genre + '/albums/')
            move_subsample(dist, nams, './'+genre + '/')

    except Exception as e:
        print(e)