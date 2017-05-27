# Theano-Kaldi RNNs
# Mirco Ravanelli - Fondazione Bruno Kessler
# May 2017

# Description: 
# this code implements bidirectional RNNs (reluRNN,LSTM,GRU,reluGRU,M_GRU,M_reluGRU) in theano.
# Batch normalization (applied to feed-forward connection) and recurrent dropout are also included
# The optimization is based on RMSE prop algorithm

import numpy as np

import theano

import theano.tensor as T

from theano import function

from theano import shared

import time

import six.moves.cPickle as pickle

from theano import function

import os

import timeit

import sys

import re

from data_io import load_dataset,context_window,load_chunk,load_chunk_nolab,load_counts,print_ark_binary,parse_option,store_options

import random 

from recurrent_neural_networks import init

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import ConfigParser

from shutil import copyfile

from optparse import OptionParser


left_lab=0
right_lab=0

#  Parsing options from command line and config file
options=parse_option()

#[to do]
do_training=options.do_training
do_eval=options.do_eval
do_forward=options.do_forward 
 

#[data]
out_file=options.out_file
tr_files=options.tr_files.split(',')
tr_labels=options.tr_labels.split(',')
dev_files=options.dev_files.split(',')
dev_labels=options.dev_labels.split(',')
te_files=options.te_files.split(',')
pt_file=options.pt_file


#[architecture]
cw_left=int(options.cw_left)
cw_right=int(options.cw_right)
N_fea=int(options.N_fea)
NN_type=options.NN_type
N_lay=int(options.N_lay)
N_hid=int(options.N_hid)
N_out=int(options.N_out)
seed=int(options.seed)

#[optimization]
batch_size=int(options.batch_size)
learning_rate=float(options.learning_rate)
options.dropout_factor=1-float(options.dropout_factor)
dropout_factor=options.dropout_factor
alpha_tr=float(options.alpha)
alpha_mem=float(options.alpha_mem)
epsilon=float(options.epsilon)

#[forward]
count_file=options.count_file
best_model=options.best_model


if NN_type=='GRU':
 from recurrent_neural_networks import GRU as RNN_arch
 
if NN_type=='reluGRU':
 from recurrent_neural_networks import reluGRU as RNN_arch
 
if NN_type=='M_GRU':
 from recurrent_neural_networks import M_GRU as RNN_arch

if NN_type=='M_reluGRU':
 from recurrent_neural_networks import M_reluGRU as RNN_arch
 
if NN_type=='LSTM':
 from recurrent_neural_networks import LSTM as RNN_arch

if NN_type=='reluRNN':
 from recurrent_neural_networks import reluRNN as RNN_arch
 

# input parameters of the computational graph
index=T.lscalar('index')
alpha=T.scalar('alpha')
test_flag=T.scalar('test_flag')
lr=T.scalar('lr')
batch_size_gh = T.iscalar('batch_size_gh')

RNN=RNN_arch(index=index,alpha=alpha,test_flag=test_flag,lr=lr,batch_size_gh=batch_size_gh,options=options)


# Do pre-training/load a saved model if necessary
if pt_file != 'none':
 open_param=pickle.load(open(pt_file))
 
 for param_index in range(len(RNN.param_save)):
  RNN.param_save[param_index].set_value(open_param[param_index])

   
if do_training=='yes':
  
 # Compilation of the computational graph
 train = theano.function(inputs=[lr,batch_size_gh,test_flag,alpha], outputs=[RNN.cost,RNN.err],updates=RNN.updates)


 # -------------TRAINING LOOP---------------------
   
 N_chunks=len(tr_files)
 
 # ---------------Training Phase----------------#
  
 start_epoch=timeit.default_timer()
 
 N_tr_snt_tot=0
 
 for chunk in range(0,N_chunks):
   
  # Load Training Data
  [data_set_list_ord,data_lab_list_ord,snt_len_list_ord]=load_chunk(tr_files[chunk],tr_labels[chunk],cw_left,cw_right)
  N_tr_snt=len(data_set_list_ord)
  N_tr_snt_tot=N_tr_snt_tot+N_tr_snt
  n_train_batches=N_tr_snt // batch_size
  beg_minibatch_index=0

  for i in range(n_train_batches):
    # Forming the mini-batches
    end_minibatch_index=beg_minibatch_index+batch_size
    max_len=max(snt_len_list_ord[beg_minibatch_index:end_minibatch_index])
    minibatch_tensor_fea=np.zeros([max_len,batch_size,N_fea],dtype=theano.config.floatX) # frame_index,fea_index, batch_index
    minibatch_tensor_lab=np.zeros([max_len,batch_size,1],dtype='int32')

    for k in range(batch_size):
      beg_index_fr=max_len-snt_len_list_ord[beg_minibatch_index+k]
      end_index_fr=max_len
      minibatch_tensor_fea[beg_index_fr:end_index_fr,k,:]=data_set_list_ord[beg_minibatch_index+k]
      minibatch_tensor_lab[beg_index_fr:end_index_fr,k,:]=data_lab_list_ord[beg_minibatch_index+k]
    
  
    # processing minibatches
    RNN.x.set_value(minibatch_tensor_fea)
    RNN.y_lab.set_value(minibatch_tensor_lab)


    [tr_cost,tr_err]=train(learning_rate,batch_size,0,alpha_tr)

    if i==0:
     avg_tr_cost=tr_cost
     avg_tr_err=tr_err
    else:
     avg_tr_cost=avg_tr_cost+tr_cost
     avg_tr_err=avg_tr_err+tr_err

  
    beg_minibatch_index=end_minibatch_index
 
 end_epoch=timeit.default_timer()
  
  
 # ---------------Saving model----------------#
 current_param=[]
  
 for p in RNN.param_save: 
    current_param.append(p.get_value())
  
 with open(out_file, 'wb') as f:
    pickle.dump(current_param,f)
    
  
 # Save training info
 info_file=out_file.replace(".pkl","-train.info")
 store_options(options,info_file)
 
 with open(info_file, "a") as inf:
    inf.write("[TRAINING_RESULTS]\n")
    inf.write("tr_cost=%f\n" %(avg_tr_cost/n_train_batches))
    inf.write("tr_err=%f\n" %(avg_tr_err/n_train_batches))
    inf.write("tr_N_snt=%i\n" %(N_tr_snt_tot))
    inf.write("tr_time=%f\n" %(end_epoch-start_epoch))
    
# ---------------Evaluation Phase----------------#
if do_eval=='yes':
  
 # Compilation of dev function
 dev=function([batch_size_gh,test_flag],[RNN.cost,RNN.err])
 
 start_epoch=timeit.default_timer()
 N_chunks_dev=len(dev_files)
 n_dev_snt_tot=0
 for chunk_dev in range(0,N_chunks_dev):
    
  # Open Dev data
  [dev_set_list_ord,dev_lab_list_ord,snt_len_list_ord_dev]=load_chunk(dev_files[chunk_dev],dev_labels[chunk_dev],cw_left,cw_right)
    
  n_dev_snt=len(dev_set_list_ord)  
  n_dev_snt_tot=n_dev_snt_tot+n_dev_snt
    
  for i in range(n_dev_snt):

    snt_length=snt_len_list_ord_dev[i]
      
    minibatch_tensor_fea=np.zeros([snt_length,1,N_fea],dtype=theano.config.floatX) # frame_index,fea_index, batch_index
    minibatch_tensor_lab=np.zeros([snt_length,1,1],dtype='int32')
      
    minibatch_tensor_fea[:,0,:]=dev_set_list_ord[i]
    minibatch_tensor_lab[:,0,:]=dev_lab_list_ord[i]
      
    RNN.x.set_value(minibatch_tensor_fea)
    RNN.y_lab.set_value(minibatch_tensor_lab)
      
    [dev_cost,dev_err]=dev(1,1)
  
    if i==0 and chunk_dev==0:
      avg_dev_cost=dev_cost
      avg_dev_err=dev_err
    else:
      avg_dev_cost=avg_dev_cost+dev_cost
      avg_dev_err=avg_dev_err+dev_err
 
 end_epoch=timeit.default_timer()
 
 info_file=out_file.replace(".pkl","-dev.info")
 store_options(options,info_file)
 # Printing output
 with open(info_file, "a") as inf:
   inf.write("[DEV_RESULTS]\n")
   inf.write("dev_cost=%f\n" %(avg_dev_cost/n_dev_snt))
   inf.write("dev_err=%f\n" %(avg_dev_err/n_dev_snt))
   inf.write("dev_N_snt=%i\n" %(n_dev_snt_tot))
   inf.write("dev_time=%f\n" %(end_epoch-start_epoch))

      
if do_forward=='yes':
 
 # Load count file
 counts = load_counts(count_file)
 
 # Forward function compilation
 forward=function([batch_size_gh,test_flag],[T.log(RNN.p_y_given_x)-T.log(counts/T.sum(counts))])
 
 start_epoch=timeit.default_timer()
 
 # Load best_model
 open_param=pickle.load(open(best_model))
 
 for param_index in range(len(RNN.param_save)):
  RNN.param_save[param_index].set_value(open_param[param_index])
  

  
 N_chunks_te=len(te_files)

 n_test_snt_tot=0
 
 for chunk_te in range(0,N_chunks_te):  
   
  # Load test data
  [test_name,test_set,end_index_test]=load_chunk_nolab(te_files[chunk_te],cw_left,cw_right,'yes',-1)
     
  out_ark=out_file.replace(".pkl", "_"+os.path.splitext(os.path.basename(te_files[chunk_te]))[0]+".ark")

  out_w = open(out_ark, 'w')
  
  n_test_snt=len(test_name)
  n_test_snt_tot=n_test_snt_tot+n_test_snt
  beg_minibatch_test=0
    
  for i in range(n_test_snt):
    end_minibatch_test=end_index_test[i]

    snt_len=(end_minibatch_test-beg_minibatch_test)+1
    minibatch_tensor_fea=np.zeros([snt_len,1,N_fea],dtype=theano.config.floatX)
    minibatch_tensor_fea[:,0,:]=test_set[beg_minibatch_test:end_minibatch_test+1,0:N_fea] 
    
    RNN.x.set_value(minibatch_tensor_fea)
    beg_minibatch_test=end_minibatch_test+1
    
    [prob_out]=forward(1,1)
    print_ark_binary(out_w,test_name[i],prob_out)
  out_w.close()
  
 end_epoch=timeit.default_timer() 
 
 info_file=out_file.replace(".pkl","-forward.info")
 store_options(options,info_file)
 # Printing output
 with open(info_file, "a") as inf:
   inf.write("[FORWARD_RESULTS]\n")
   inf.write("test_N_snt=%i\n" %(n_test_snt_tot))
   inf.write("test_time=%f\n" %(end_epoch-start_epoch))

