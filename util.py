import os


def mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, (list, tuple)):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
