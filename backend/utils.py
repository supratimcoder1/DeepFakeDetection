import os

def cleanup_temp(path):
    if os.path.exists(path):
        os.remove(path)