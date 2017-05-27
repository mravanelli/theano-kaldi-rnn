import numpy as np
import struct
def parse_matrix(stream):
    result = []
    line = stream.next().strip()
    while not line.endswith(']'):
        result.append(map(float,line.split()))
        line = stream.next().strip()
    result.append(map(float,line.split()[:-1]))
    return np.array(result,dtype=np.float32)
    
def parse(stream):
    for line in stream:
        line = line.strip()
        if line.endswith('['):
            name = line.strip().split()[0]
            yield name,parse_matrix(stream)

def print_ark(name,array):
    print name,"["
    for i,row in enumerate(array):
        print " ",
        for cell in row:
            print "%0.6f"%cell,
        if i == array.shape[0]-1:
            print "]"
        else:
            print

def print_ark_binary(buffer,name,array):
    activations = np.asarray(array, dtype='float32')
    rows, cols = array.shape
    buffer.write(struct.pack('<%ds' % (len(name)), name))
    buffer.write(struct.pack('<cxcccc', ' ', 'B', 'F', 'M', ' '))
    buffer.write(struct.pack('<bi', 4, rows))
    buffer.write(struct.pack('<bi', 4, cols))
    buffer.write(array)

def parse_binary(buffer):
    try:
        while True:
            name = read_uttid(buffer)
            if name == '' : return
            data = read_kaldi_matrix(buffer)
            yield name,data
    except EOFError:
        pass


def read_kaldi_matrix(buffer, skip_binary_preamble=False):
    """
    Kaldi binary reader thanks to Pawel
    https://github.com/pswietojanski/pylearn2speech/blob/master/pylearn2/datasets/speech_utils/kaldi_providers.py#L25
    """

    if skip_binary_preamble:
        descr = struct.unpack('<ccc', buffer.read(5))  # read 0B{F,D}{V,M,C}[space], function tested for 0BFM types only
        repr_type = descr[0]
        cont_type = descr[1]
    else:
        descr = struct.unpack('<xcccc', buffer.read(5))  # read 0B{F,D}{V,M,C}[space], function tested for 0BFM types only
        binary_mode = descr[0]
        repr_type = descr[1]
        cont_type = descr[2]
        assert binary_mode == "B"

    if (repr_type == 'F'):
        dtype = np.dtype(np.float32)
    elif (repr_type == 'D'):
        dtype = np.dtype(np.float64)
    else:
        raise ValueError('Wrong representation type in Kaldi header (is feats '
                        'compression enabled? - this is not supported in the '
                        'current version. Feel free to add this functionality.): %c' % (repr_type))

    rows, cols = 1, 1
    if (cont_type == 'M'):
        p1, rows = struct.unpack('<bi', buffer.read(5))  # bytes from 5 to 10
        p2, cols = struct.unpack('<bi', buffer.read(5))  # bytes from 10 to 15
        assert p1 == 4 and p2 == 4  # Number of bytes dimensionality is stored?
    elif (cont_type == 'V'):
        p1, rows = struct.unpack('<bi', buffer.read(5))  # bytes from 5 to 10
        assert p1 == 4
    else:
        raise Exception('Wrong container type in Kaldi header: %c' % (cont_type))

    assert rows > 0 and cols > 0  # just a range sanity checks
    assert rows < 360000 and cols < 30000  # just a sensible range sanity checks

    result = np.frombuffer(buffer.read(rows * cols * dtype.itemsize), dtype=dtype)
    if (cont_type == 'M'):
        result = np.reshape(result, (rows, cols))

    return result


def read_uttid(buffer):
    uttid = ''
    c = buffer.read(1)
    while c != ' ' and c != '':
        uttid += c
        c = buffer.read(1)
    return uttid

