import subprocess
import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle
import glob

def label_stream():
    for l in sys.stdin:
        l = l.rstrip().split()
        name = l[0]
        pdfs = np.array(map(int,l[1:]),dtype=np.int32)
        yield name,pdfs

if __name__ == "__main__":
    output_file = sys.argv[1]
    labels = label_stream()
    with gzip.open(output_file,'wb') as f:
        count = 0
        for name,lbls in labels:
            pickle.dump((name,lbls),f,protocol=2)
            count += 1
            if count % 100 == 0:
                print "Wrote %d utterances to %s"%(count,output_file)
    print "Wrote %d utterances to %s"%(count,output_file)


