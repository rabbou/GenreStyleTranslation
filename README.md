# Genre to Album Translation for Artwork Creation Using Cycle-Consistent Adversarial Network

*This project will be completed by April 1st*

## Data

To download album covers from amazon to the `albums` folder, run

    python3 download_data.py

To make a folder for each wanted genre, run

    python3 build_genre_data.py --genres Genre1 Genre2 Genre3 ...

This will also run k-means on each genre to create a batch of more similar
artwork within each genre, and we create a test/train (90%/10%) split of the data.
The resulting folders are located in `albums/Genre/train` and `albums/Genre/test`.

Finally, we need to create the cycleGAN datasets, which will be called `classic2metal`
for instance if we want to translate artwork from Classic to Metal. Each such folder,
located in the `dataset` directory contains `trainA, trainB, testA, testB` subfolders
corresponding to each genres.

    python3 build_gan_ds.py --genreA GenreA --genreB GenreB
