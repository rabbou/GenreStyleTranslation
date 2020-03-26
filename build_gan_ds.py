import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--genres', nargs=2,
                    help='Genres as a list')

def build_gan_ds(labelA, labelB, foldername):
    path = 'datasets/'
    if not os.path.exists(path+foldername):
        os.mkdir(path+foldername)
        os.mkdir(path+'trainA')
        os.mkdir(path+'trainB')
        os.mkdir(path+'testA')
        os.mkdir(path+'testB')
    shutil.copytree(path+labelA+'/train', path+foldername+'/trainA')
    shutil.copy(path+labelA+'/test', path+foldername+'/testA')
    shutil.copytree(path+labelB+'/train', path+foldername+'/trainB')
    shutil.copy(path+labelB+'/test', path+foldername+'/testB')

if __name__ == '__main__':
    args = parser.parse_args()
    genres = args.genres
    build_gan_ds(genres[0], genres[1], lower(genres[0])+'2'+lower(genres[1]))
