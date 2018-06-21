import pickle
import os

def load_tag_fies(index_file):
    tag_files = []
    with open(index_file) as f:
        for line in f:
            tag_files.append(line.replace('\n',''))
    return tag_files

def load(filename):
    if os.path.isfile(filename):
        print 'Load from %s' % filename
        with open(filename) as f:
            [data,] = pickle.load(f)
        return data
    else: return None

def save(data, filename):
    print 'Saving to %s' % filename
    with open(filename, 'w') as f:
        pickle.dump([data,], f)

