import os


def get_folder_name(path, prefix='model'):
    folders = [f for f in os.listdir(path)]
    if not folders:
        path = os.path.join(path, '{}_0'.format(prefix))
    else:
        n = sorted([int(f.split('_')[-1]) for f in folders])[-1]
        path = os.path.join(path, '{}_{}'.format(prefix, n+1))
    return path
