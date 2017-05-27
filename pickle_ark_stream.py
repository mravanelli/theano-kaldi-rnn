import subprocess
import numpy as np
import os
from itertools import izip
import sys
import gzip
import cPickle as pickle

import ark_io

def ark_stream():
    return ark_io.parse_binary(sys.stdin)

if __name__ == "__main__":
    output_file = sys.argv[1]
    features = ark_stream()
    with gzip.open(output_file,'wb') as f:
        count = 0
        for name,features in features:
            pickle.dump((name,features),f,protocol=2)
            count += 1
            if count % 100 == 0:
                print "Wrote %d utterances to %s"%(count,output_file)
    print "Wrote %d utterances to %s"%(count,output_file)

