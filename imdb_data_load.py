"""
    Download from https://ai.stanford.edu/~amaas/data/sentiment/
"""

import os
import pickle 

def load_from (folder):
    print(f'load_from: {folder}')
    tlist = []
    txtfiles = os.listdir(folder)
    for file in txtfiles:
        filepath = os.path.join (folder, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            s = f.readlines()[0]
            # print(type(s), s)
            # break
             # very simple text maipulation, and convert to list
            tlist.append(s.replace('\n', ' ').split())
    return tlist

def load_texts(folder):
    pos = load_from(folder + '/pos')
    neg = load_from(folder + '/neg')
    print (len(pos), len(neg))
    return pos, neg

def load_data(folder):
    pos, neg = load_texts(folder)
    xtrain = pos + neg 
    ytrain = [1]*len(pos) + [0]*len(neg)
    return xtrain, ytrain

def load_imdb(folder, loadpkl=True):
    pklfilename = os.path.join(folder, 'my_imdb.pkl')
    print('load_imdb(): ', pklfilename)
    if loadpkl:
        with open(pklfilename, 'rb') as f:
            xtr, ytr, xte, yte = pickle.load(f)
    else:
        xtr, ytr = load_data(os.path.join(folder, 'train'))
        xte, yte = load_data(os.path.join(folder, 'test'))
        with open(pklfilename, 'wb') as f:
            pickle.dump([xtr,ytr,xte,yte], f)
    #
    return xtr, ytr, xte, yte
#

if __name__ == "__main__" :
    print(os.listdir('data/aclImdb'))
    xtrain, ytrain, xtest, ytest = load_imdb ('.data/aclImdb', loadpkl=False)
    print(len(ytrain))
    print(len(ytest))
    print('Finished.')