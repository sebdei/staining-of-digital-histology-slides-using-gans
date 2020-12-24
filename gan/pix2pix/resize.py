from utils.resize import align_sizes
import glob


image_paths = glob.glob("data/train/*")
image_paths = sorted(image_paths)

assert len(image_paths) % 2 == 0

for i in range(0, int(len(image_paths) / 2)):
    path1 = image_paths[i * 2]
    path2 = image_paths[i * 2 + 1]

    align_sizes(path1, path2)
