import gzip
import six.moves.cPickle as pickle
import numpy as np
import sys
import struct
import ConfigParser
from optparse import OptionParser

def parse_option():
 # Parsing command line
 parser=OptionParser()
 parser.add_option("--cfg") # Mandatory

 # To do options
 parser.add_option("--do_training")
 parser.add_option("--do_eval")
 parser.add_option("--do_forward")

 # Data options
 parser.add_option("--out_file")
 parser.add_option("--tr_files")
 parser.add_option("--tr_labels")
 parser.add_option("--dev_files")
 parser.add_option("--dev_labels")
 parser.add_option("--te_files")
 parser.add_option("--cw_left")
 parser.add_option("--cw_right")
 parser.add_option("--pt_file")


 # Architecture options
 parser.add_option("--NN_type")
 parser.add_option("--N_fea")
 parser.add_option("--N_lay")
 parser.add_option("--N_hid")
 parser.add_option("--N_out")
 parser.add_option("--seed")

 # Optimization options
 parser.add_option("--batch_size")
 parser.add_option("--learning_rate")
 parser.add_option("--dropout_factor")
 parser.add_option("--alpha")
 parser.add_option("--alpha_mem")
 parser.add_option("--epsilon")

 # Forward options
 parser.add_option("--count_file")
 parser.add_option("--best_model")


 # Parsing Options
 (options,args)=parser.parse_args()

 # Reading the Config File
 cfg_file=options.cfg
 Config = ConfigParser.ConfigParser()
 Config.read(cfg_file)

 # To do options
 if options.do_training==None:
  options.do_training=Config.get('todo', 'do_training')
  
 if options.do_eval==None:
  options.do_eval=Config.get('todo', 'do_eval')
 
 if options.do_forward==None:
  options.do_forward=Config.get('todo', 'do_forward')
 
 
 # Data options
 if options.out_file==None:
  options.out_file=Config.get('data', 'out_file')
 
 if options.tr_files==None:
  options.tr_files=Config.get('data', 'tr_files')
 
 if options.tr_labels==None:
  options.tr_labels=Config.get('data', 'tr_labels')

 if options.dev_files==None:
  options.dev_files=Config.get('data', 'dev_files')

 if options.dev_labels==None:
  options.dev_labels=Config.get('data', 'dev_labels')
 
 if options.te_files==None:
  options.te_files=Config.get('data', 'te_files')

 
 if options.pt_file==None:
  options.pt_file=Config.get('data', 'pt_file')
 
 
 # Architecture options

 if options.cw_left==None:
  options.cw_left=Config.get('architecture', 'cw_left')
 
 if options.cw_right==None:
  options.cw_right=Config.get('architecture', 'cw_right')
 
 if options.N_fea==None:
  options.N_fea=Config.get('architecture', 'N_fea')

 if options.NN_type==None:
  options.NN_type=Config.get('architecture', 'NN_type')
 
 if options.N_lay==None:
  options.N_lay=Config.get('architecture', 'N_lay')
 
 if options.N_hid==None:
  options.N_hid=Config.get('architecture', 'N_hid')
 
 if options.N_out==None:
  options.N_out=Config.get('architecture', 'N_out')
 
 if options.seed==None:
  options.seed=Config.get('architecture', 'seed')

 # Optimization options
 if options.batch_size==None:
  options.batch_size=Config.get('optimization', 'batch_size')

 if options.learning_rate==None:
  options.learning_rate=Config.get('optimization', 'learning_rate')

 if options.dropout_factor==None:
  options.dropout_factor=Config.get('optimization', 'dropout_factor')
 
 if options.alpha==None:
  options.alpha=Config.get('optimization', 'alpha')
 
 if options.alpha_mem==None:
  options.alpha_mem=Config.get('optimization', 'alpha_mem')
 
 if options.epsilon==None:
  options.epsilon=Config.get('optimization', 'epsilon')

 # Forward options
 if options.count_file==None:
  options.count_file=Config.get('forward', 'count_file')
 
 if options.best_model==None:
  options.best_model=Config.get('forward', 'best_model')
 
 return options   

def store_options(options,out_file):
 f = open(out_file, 'w')
 
 f.write('[todo]\n')
 f.write('do_training=%s\n' %(options.do_training))
 f.write('do_training=%s\n' %(options.do_eval))
 f.write('do_forward=%s\n' %(options.do_forward)) 

 f.write('[data]\n')
 f.write('out_file=%s\n' %(options.out_file))
 f.write('tr_files=%s\n' %(options.tr_files))
 f.write('tr_labels=%s\n' %(options.tr_labels))
 f.write('dev_files=%s\n' %(options.dev_files))
 f.write('dev_labels=%s\n' %(options.dev_labels))
 f.write('te_files=%s\n' %(options.te_files))
 f.write('pt_file=%s\n\n' %(options.pt_file))

 f.write('[architecture]\n')
 f.write('cw_left=%s\n' %(options.cw_left))
 f.write('cw_right=%s\n' %(options.cw_right))
 f.write('N_fea=%s\n' %(options.N_fea))
 f.write('NN_type=%s\n' %(options.NN_type))
 f.write('N_lay=%s\n' %(options.N_lay))
 f.write('N_hid=%s\n' %(options.N_hid))
 f.write('N_out=%s\n' %(options.N_out))
 f.write('seed=%s\n' %(options.seed))

 f.write('[optimization]\n')
 f.write('batch_size=%s\n' %(options.batch_size))
 f.write('learning_rate=%s\n' %(options.learning_rate))
 f.write('dropout_factor=%s\n' %(options.dropout_factor))
 f.write('alpha=%s\n' %(options.alpha))
 f.write('alpha_mem=%s\n' %(options.alpha_mem))
 f.write('epsilon=%s\n\n' %(options.epsilon))

 f.write('[forward]\n')
 f.write('count_file=%s\n' %(options.count_file))
 f.write('best_model=%s\n\n' %(options.best_model))

 f.closed
  
def load_dataset(data_file):

 name = []
 end_index=[]
 
 f = gzip.open(data_file, 'rb')
 # load the first two objects
 [name_new,data]=pickle.load(f)
 name.append(name_new)
 end_index.append(data.shape[0]-1)
 [name_new,data_new]=pickle.load(f)
 data=np.concatenate((data,data_new),axis=0)
 name.append(name_new)
 end_index.append(data.shape[0]-1)

 # get the remaining pickled items
 while True:
  try:
   [name_new,data_new]=pickle.load(f)
   data=np.concatenate((data,data_new),axis=0)
   name.append(name_new)
   end_index.append(data.shape[0]-1)
  except EOFError:
   break

 return [name,data,end_index]


def context_window(fea,left,right):
 
 if len(fea.shape)==2:
  N_row=fea.shape[0]
  N_fea=fea.shape[1]
 else:
  N_row=fea.shape[0]
  N_fea=1
 
 frames = np.empty((N_row-left-right, N_fea*(left+right+1)))
 
 for frame_index in range(left,N_row-right):
  right_context=fea[frame_index+1:frame_index+right+1].flatten() # right context
  left_context=fea[frame_index-left:frame_index].flatten() # left context
   
  current_frame=np.concatenate([left_context,fea[frame_index].flatten(),right_context])

  frames[frame_index-left]=current_frame

 return frames


def print_ark_binary(buffer,name,array):
    activations = np.asarray(array, dtype='float32')
    rows, cols = array.shape
    buffer.write(struct.pack('<%ds' % (len(name)), name))
    buffer.write(struct.pack('<cxcccc', ' ', 'B', 'F', 'M', ' '))
    buffer.write(struct.pack('<bi', 4, rows))
    buffer.write(struct.pack('<bi', 4, cols))
    buffer.write(array)
    
def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = f.next().strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts

  
def load_chunk(tr_file,tr_label,left,right):
 # open the file
 [data_name,data_set,end_index]=load_dataset(tr_file)
 [data_name,data_lab,end_index_lab]=load_dataset(tr_label)

 # Context window
 data_set=context_window(data_set,left,right)
 left_lab=0
 right_lab=0
 data_lab=context_window(data_lab,left_lab,right_lab)

 # mean and variance normalization
 data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)

 fea_dim=data_set.shape[1]

 # Label processing
 data_lab=data_lab-data_lab.min()

 # Zero padding
 data_set=np.concatenate((np.zeros([left,data_set.shape[1]]), data_set, np.zeros([right,data_set.shape[1]])))
 data_lab=np.concatenate((np.zeros([left_lab,data_lab.shape[1]]), data_lab, np.zeros([right_lab,data_lab.shape[1]])))

 # list conversion
 beg_snt=0
 data_set_list=[]
 data_lab_list=[]
 snt_len_list=[]

 for end_snt in end_index:
    data_set_list.append(data_set[beg_snt:end_snt+1])
    data_lab_list.append(data_lab[beg_snt:end_snt+1])
    snt_len_list.append(data_set[beg_snt:end_snt+1].shape[0])
    beg_snt=end_snt+1

 del data_set
 del data_lab

 snt_len_list_ord,range_ord,data_set_list_ord,data_lab_list_ord=zip(*sorted(zip(snt_len_list,range(len(snt_len_list)),data_set_list,data_lab_list)))

    
 del data_set_list
 del data_lab_list
 del snt_len_list
 
 return data_set_list_ord,data_lab_list_ord,snt_len_list_ord



def load_chunk_nolab(data_file,left,right,norm_input_flag,shuffle_seed):

  # Open the file
  [data_name,data_set,end_index]=load_dataset(data_file)

  # Context window
  data_set=context_window(data_set,left,right)

  # Zero padding
  data_set=np.concatenate((np.zeros([left,data_set.shape[1]]), data_set, np.zeros([right,data_set.shape[1]])))

  # Mean and Variance Normalization
  if norm_input_flag=='yes':
   data_set=(data_set-np.mean(data_set,axis=0))/np.std(data_set,axis=0)


  # shuffle (only for test data)
  if shuffle_seed>0:
   np.random.seed(shuffle_seed)
   np.random.shuffle(data_set)

  return [data_name,data_set,end_index]


