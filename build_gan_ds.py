import os, shutil, argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument('--genreA', help='Genre A')
parser.add_argument('--genreB', help='Genre B')
parser.add_argument('--new_path', default='datasets/', help='Genres as a list')
parser.add_argument('--album_path', default='albums/', help='directory where albums are')
parser.add_argument('--clustering_method', default='kmeans', help='method chosen')

def create_with_overwrite(dir_name):
    """Create new directory or overwrite existing dir if it exists."""
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def build_gan_ds(new_path, album_path, labelA, labelB, foldername, method):
    create_with_overwrite(new_path+foldername)
    shutil.copytree(album_path+labelA+'/train'_+method, new_path+foldername+'/trainA')
    shutil.copytree(album_path+labelA+'/test_'+method, new_path+foldername+'/testA')
    shutil.copytree(album_path+labelB+'/train_'+method, new_path+foldername+'/trainB')
    shutil.copytree(album_path+labelB+'/test_'+method, new_path+foldername+'/testB')
    print('Size of train A:', len(list(glob.glob(f"{new_path+foldername+'/trainA'}/*.jpg"))))
    print('Size of train B:', len(list(glob.glob(f"{new_path+foldername+'/trainB'}/*.jpg"))))
    print('Size of test A:', len(list(glob.glob(f"{new_path+foldername+'/testA'}/*.jpg"))))
    print('Size of test B:', len(list(glob.glob(f"{new_path+foldername+'/testB'}/*.jpg"))))

if __name__ == '__main__':
    args = parser.parse_args()
    genres = args.genres
    build_gan_ds(args.new_path, args.album_path, genres[0], genres[1],
                 genres[0].lower()+'2'+genres[1].lower(), args.clustering_method)
