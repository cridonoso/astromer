import os

def set_working_directory_up_two_levels():
    current_dir = os.getcwd()
    if current_dir.split('/')[-1] == 'astromer':
        return
    target_dir = os.path.abspath(os.path.join(current_dir, '../../'))  # Go up two levels
    os.chdir(target_dir)