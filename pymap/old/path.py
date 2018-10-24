import os.path

class Path:
    """ Class to handle cross-plattform paths by storing them as lists."""

    def __init__(self, path):
        """ Initializes the path instance.
        
        Parameters:
        -----------
        path : string or list
            The path either as absolute string or its components as list."""

        if type(path) == str:
            self.path = os.path.normpath(path).split(os.path.sep)
        elif type(path) == list:
            self.path = path
        else:
            raise RuntimeError(f'Unkown path class type {type(path)}')

    def __str__(self):
        return os.path.sep.join(self.path)

    def __getitem__(self, key):
        if not type(key) == int:
            raise RuntimeError(f'Invalid key for path instances {key}. Only integers are allowed!')
        return self.path[key]

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        return PathIterator(self)

class PathIterator:
    """ Iterator class for the path instance """
    def __init__(self, path):
        self.path = path
        self.current = 0

    def __next__(self):
        if self.current >= len(self.path):
            raise StopIteration
        self.current += 1
        return self.path[self.current - 1]


ROOT = "tools/"
THIS = "../"

def rootpath(p):
    """ Extends a path relative to the tools directory to be relative to root directory"""
    return ROOT + os.path.relpath(p).replace("\\", "/")

def path(p, from_root):
    return rootpath(p) if from_root else os.path.relpath(p)