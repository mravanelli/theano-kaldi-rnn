import numpy as np

import theano

import theano.tensor as T

from theano import function

from theano import shared

import sys

import re

import random 

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def init(init_type,size,seed):
    
    rng = np.random.RandomState(seed)
    
    if init_type=="sigmoid":
      n_in=size[0]
      n_out=size[1]
      init_arr = np.asarray(rng.uniform(low=-4.*np.sqrt(6. / (n_in + n_out)),high=4.*np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
   
    if init_type=="tanh" or init_type=="glorot" or init_type=="relu":
      n_in=size[0]
      n_out=size[1]
      init_arr = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
    
    if bool(re.search('glorot_red', init_type)):
       red_factor = float(re.search('glorot_red_f_(.+)', init_type).group(1))
       n_in=size[0]
       n_out=size[1]
       init_arr = np.asarray(rng.uniform(low=-np.sqrt(red_factor / (n_in + n_out)),high=np.sqrt(red_factor / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
     
    if bool(re.search('single', init_type)):
       value = float(re.search('single_v_(.+)', init_type).group(1))
       n_in=size[0]
       init_arr=np.ones(n_in,dtype=theano.config.floatX)*value

    if bool(re.search('uniform', init_type)):
       low_val = float(re.search('uniform_l_(.+)_h', init_type).group(1))
       high_val = float(re.search('uniform_l_(.+)_h_(.+)', init_type).group(2))
       n_in=size[0]
       n_out=size[1]
       init_arr = np.asarray(rng.uniform(low=low_val,high=high_val,size=(n_in, n_out)),dtype=theano.config.floatX)
       
    return init_arr
  
# Step functon for GRU time-unroling    
def step_GRU(at_h,at_z,at_r, h_tm1, U_h,U_z,U_r,drop_mask):
    z_t=T.nnet.sigmoid(at_z+T.dot(h_tm1,U_z)) #update gate
    r_t=T.nnet.sigmoid(at_r+T.dot(h_tm1,U_r)) #reset gate
    h_t=(1-z_t)*(T.tanh(at_h+T.dot(h_tm1*r_t,U_h))*drop_mask)+z_t*h_tm1
    return h_t
  
def step_reluGRU(at_h,at_z,at_r, h_tm1, U_h,U_z,U_r,drop_mask):
    z_t=T.nnet.sigmoid(at_z+T.dot(h_tm1,U_z)) #update gate
    r_t=T.nnet.sigmoid(at_r+T.dot(h_tm1,U_r)) #reset gate
    h_t=(1-z_t)*(T.nnet.relu(at_h+T.dot(h_tm1*r_t,U_h))*drop_mask)+z_t*h_tm1
    return h_t

def step_M_GRU(at_h,at_z, h_tm1, U_h,U_z,drop_mask):
    z_t=T.nnet.sigmoid(at_z+T.dot(h_tm1,U_z)) #update gate
    h_t=(1-z_t)*(T.tanh(at_h+T.dot(h_tm1,U_h))*drop_mask)+z_t*h_tm1
    return h_t
  
def step_M_reluGRU(at_h,at_z, h_tm1, U_h,U_z,drop_mask):
    z_t=T.nnet.sigmoid(at_z+T.dot(h_tm1,U_z)) #update gate
    h_t=(1-z_t)*(T.nnet.relu(at_h+T.dot(h_tm1,U_h))*drop_mask)+z_t*h_tm1
    return h_t
  
def step_LSTM(at_f,at_i,at_o, at_c,h_tm1,c_tm1 ,U_f,U_i,U_o,drop_mask):
    f_t=T.nnet.sigmoid(at_f+T.dot(c_tm1,U_f)) #forgot gate
    i_t=T.nnet.sigmoid(at_i+T.dot(c_tm1,U_i)) #input gate
    o_t=T.nnet.sigmoid(at_o+T.dot(c_tm1,U_o)) #output gate
    c_t=f_t*c_tm1+i_t*T.tanh(at_c)*drop_mask
    h_t=o_t*T.tanh(c_t)
    return h_t,c_t

def step_reluRNN(at, h_tm1, W,drop_mask):
    h_t = T.nnet.relu(at + T.dot(h_tm1, W))*drop_mask
    return h_t
  
class GRU(object):
  
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)

  # input features and labels     
  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')

  # parameters list definition
  W_h=[]
  W_z=[]
  W_r=[]
  
  running_avg_h=[]
  running_std_h=[]
  running_avg_z=[]
  running_std_z=[]
  running_avg_r=[]
  running_std_r=[]

  gamma_h=[]
  beta_h=[]
  gamma_z=[]
  beta_z=[]
  gamma_r=[]
  beta_r=[]

  U_h=[]
  U_z=[]
  U_r=[]

  # Parameter initialization
  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid
   
   # Feed Forward Connections
   W_h.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_h',borrow=True))
   W_z.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+1),name='W_z',borrow=True))
   W_r.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+2),name='W_r',borrow=True))


   running_avg_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_h'))
   running_std_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_h'))
   running_avg_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_z'))
   running_std_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_z'))
   running_avg_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_r'))
   running_std_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_r'))

   gamma_h.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_h'))
   beta_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_h'))
   gamma_z.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_z'))
   beta_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_z'))
   gamma_r.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_r'))
   beta_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_r'))

   # Recurrent Connections
   U_h_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed)
   U_z_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+1)
   U_r_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+2)

   Q, R = np.linalg.qr(U_h_init)
   Q = Q * np.sign(np.diag(R))
   U_h.append(shared(Q,name='U_h',borrow=True))

   Q, R = np.linalg.qr(U_z_init)
   Q = Q * np.sign(np.diag(R))
   U_z.append(shared(Q,name='U_z',borrow=True))

   Q, R = np.linalg.qr(U_r_init)
   Q = Q * np.sign(np.diag(R))
   U_r.append(shared(Q,name='U_r',borrow=True))

  # output layer
  b_o=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')
  W_o=shared(init('glorot_red_f_0.0',(2*N_hid,N_out),seed),name='W_o',borrow=True)

  # Parameter for updating
  param=W_h+W_z+W_r+U_h+U_z+U_r+[W_o]+[b_o]+gamma_h+beta_h+gamma_z+beta_z+gamma_r+beta_r
  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  # Parameter to save
  param_save=param+running_avg_h+running_std_h+running_avg_z+running_std_z+running_avg_r+running_std_r+memory_grad

  # Update list initialization
  updates_running_avg_h=[]
  updates_running_std_h=[]
  updates_running_avg_z=[]
  updates_running_std_z=[]
  updates_running_avg_r=[]
  updates_running_std_r=[]

  # Definition of the computations for each layer
  current_input=x
  for i in range(N_lay):
   W_conc=T.concatenate([W_h[i],W_z[i],W_r[i]],axis=1) # concatenation of W matrices
   at_conc=T.dot(current_input, W_conc)                # It is convenient to perform these dot products in parallel for all the time steps outside the scan.
   at_h=at_conc[:,:,0:N_hid] # de-concatenation
   at_z=at_conc[:,:,N_hid:N_hid*2]  #de-concatenation
   at_r=at_conc[:,:,N_hid*2:N_hid*3]  #de-concatenation

   # Means and stds computations for batch norm 
   mean_h=T.mean(at_h,axis=0)
   std_h=T.std(at_h,axis=0)

   mean_h=(1-test_flag)*T.mean(mean_h,axis=0)+test_flag*running_avg_h[i]
   std_h=(1-test_flag)*T.mean(std_h,axis=0)+test_flag*running_std_h[i]

   mean_z=T.mean(at_z,axis=0)
   std_z=T.std(at_z,axis=0)

   mean_z=(1-test_flag)*T.mean(mean_z,axis=0)+test_flag*running_avg_z[i]
   std_z=(1-test_flag)*T.mean(std_z,axis=0)+test_flag*running_std_z[i]

   mean_r=T.mean(at_r,axis=0)
   std_r=T.std(at_r,axis=0)

   mean_r=(1-test_flag)*T.mean(mean_r,axis=0)+test_flag*running_avg_r[i]
   std_r=(1-test_flag)*T.mean(std_r,axis=0)+test_flag*running_std_r[i]

   # batch norm computation
   at_h=T.nnet.bn.batch_normalization(at_h, gamma_h[i], beta_h[i], mean_h, std_h, mode='low_mem')
   at_z=T.nnet.bn.batch_normalization(at_z, gamma_z[i], beta_z[i], mean_z, std_z, mode='low_mem')
   at_r=T.nnet.bn.batch_normalization(at_r, gamma_r[i], beta_r[i], mean_r, std_r, mode='low_mem')

   # Concatenation for Bidirectional processing
   at_h=T.concatenate([at_h,at_h[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_z=T.concatenate([at_z,at_z[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_r=T.concatenate([at_r,at_r[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 

   # Estimating dropout mask (same mask for each time-step)
   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor 

   h, _  = theano.scan(step_GRU,
                              sequences=[at_h,at_z,at_r],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[U_h[i],U_z[i],U_r[i],drop_mask],
                              truncate_gradient=-1)

   # Dividing h_back and h_forward and concatenation
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation
   current_input=h
 
   # Updated definitions
   updates_running_avg_h.append((running_avg_h[i], (1-alpha)*running_avg_h[i]+alpha*mean_h))
   updates_running_std_h.append((running_std_h[i], (1-alpha)*running_std_h[i]+alpha*std_h))
   updates_running_avg_z.append((running_avg_z[i], (1-alpha)*running_avg_z[i]+alpha*mean_z))
   updates_running_std_z.append((running_std_z[i], (1-alpha)*running_std_z[i]+alpha*std_z))
   updates_running_avg_r.append((running_avg_r[i], (1-alpha)*running_avg_r[i]+alpha*mean_r))
   updates_running_std_r.append((running_std_r[i], (1-alpha)*running_std_r[i]+alpha*std_r))

  # Output layer definition

  # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
  # I have to reshape both labels and p_y_given_x to fullfill this requirement.
  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))

  p_y_given_x = T.nnet.softmax(T.dot(h, W_o)+b_o)

  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)

  err=T.mean(T.neq(y_pred,y_lab_flat))

  # Gradient Computation
  gparams = T.grad(cost, param)

  # Memory memorization for RMSProp
  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(p, p - lr/(T.sqrt(memory)+epsilon) * gparam) for p, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg_h+updates_running_std_h+updates_running_avg_z+updates_running_std_z+updates_running_avg_r+updates_running_std_r+updates_mem
  
  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x
  
  
class reluGRU(object):
  
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)

  # input features and labels     
  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')

  # parameters list definition
  W_h=[]
  W_z=[]
  W_r=[]
  
  running_avg_h=[]
  running_std_h=[]
  running_avg_z=[]
  running_std_z=[]
  running_avg_r=[]
  running_std_r=[]

  gamma_h=[]
  beta_h=[]
  gamma_z=[]
  beta_z=[]
  gamma_r=[]
  beta_r=[]

  U_h=[]
  U_z=[]
  U_r=[]

  # Parameter initialization
  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid
   
   # Feed Forward Connections
   W_h.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_h',borrow=True))
   W_z.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+1),name='W_z',borrow=True))
   W_r.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+2),name='W_r',borrow=True))


   running_avg_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_h'))
   running_std_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_h'))
   running_avg_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_z'))
   running_std_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_z'))
   running_avg_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_r'))
   running_std_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_r'))

   gamma_h.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_h'))
   beta_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_h'))
   gamma_z.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_z'))
   beta_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_z'))
   gamma_r.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_r'))
   beta_r.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_r'))

   # Recurrent Connections
   U_h_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed)
   U_z_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+1)
   U_r_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+2)

   Q, R = np.linalg.qr(U_h_init)
   Q = Q * np.sign(np.diag(R))
   U_h.append(shared(Q,name='U_h',borrow=True))

   Q, R = np.linalg.qr(U_z_init)
   Q = Q * np.sign(np.diag(R))
   U_z.append(shared(Q,name='U_z',borrow=True))

   Q, R = np.linalg.qr(U_r_init)
   Q = Q * np.sign(np.diag(R))
   U_r.append(shared(Q,name='U_r',borrow=True))

  # output layer
  b_o=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')
  W_o=shared(init('glorot_red_f_0.0',(2*N_hid,N_out),seed),name='W_o',borrow=True)

  # Parameter for updating
  param=W_h+W_z+W_r+U_h+U_z+U_r+[W_o]+[b_o]+gamma_h+beta_h+gamma_z+beta_z+gamma_r+beta_r
  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  # Parameter to save
  param_save=param+running_avg_h+running_std_h+running_avg_z+running_std_z+running_avg_r+running_std_r+memory_grad

  # Update list initialization
  updates_running_avg_h=[]
  updates_running_std_h=[]
  updates_running_avg_z=[]
  updates_running_std_z=[]
  updates_running_avg_r=[]
  updates_running_std_r=[]

  # Definition of the computations for each layer
  current_input=x
  for i in range(N_lay):
   W_conc=T.concatenate([W_h[i],W_z[i],W_r[i]],axis=1) # concatenation of W matrices
   at_conc=T.dot(current_input, W_conc)                # It is convenient to perform these dot products in parallel for all the time steps outside the scan.
   at_h=at_conc[:,:,0:N_hid] # de-concatenation
   at_z=at_conc[:,:,N_hid:N_hid*2]  #de-concatenation
   at_r=at_conc[:,:,N_hid*2:N_hid*3]  #de-concatenation

   # Means and stds computations for batch norm 
   mean_h=T.mean(at_h,axis=0)
   std_h=T.std(at_h,axis=0)

   mean_h=(1-test_flag)*T.mean(mean_h,axis=0)+test_flag*running_avg_h[i]
   std_h=(1-test_flag)*T.mean(std_h,axis=0)+test_flag*running_std_h[i]

   mean_z=T.mean(at_z,axis=0)
   std_z=T.std(at_z,axis=0)

   mean_z=(1-test_flag)*T.mean(mean_z,axis=0)+test_flag*running_avg_z[i]
   std_z=(1-test_flag)*T.mean(std_z,axis=0)+test_flag*running_std_z[i]

   mean_r=T.mean(at_r,axis=0)
   std_r=T.std(at_r,axis=0)

   mean_r=(1-test_flag)*T.mean(mean_r,axis=0)+test_flag*running_avg_r[i]
   std_r=(1-test_flag)*T.mean(std_r,axis=0)+test_flag*running_std_r[i]

   # batch norm computation
   at_h=T.nnet.bn.batch_normalization(at_h, gamma_h[i], beta_h[i], mean_h, std_h, mode='low_mem')
   at_z=T.nnet.bn.batch_normalization(at_z, gamma_z[i], beta_z[i], mean_z, std_z, mode='low_mem')
   at_r=T.nnet.bn.batch_normalization(at_r, gamma_r[i], beta_r[i], mean_r, std_r, mode='low_mem')

   # Concatenation for Bidirectional processing
   at_h=T.concatenate([at_h,at_h[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_z=T.concatenate([at_z,at_z[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_r=T.concatenate([at_r,at_r[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 

   # Estimating dropout mask (same mask for each time-step)
   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor 

   h, _  = theano.scan(step_reluGRU,
                              sequences=[at_h,at_z,at_r],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[U_h[i],U_z[i],U_r[i],drop_mask],
                              truncate_gradient=-1)

   # Dividing h_back and h_forward and concatenation
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation
   current_input=h
 
   # Updated definitions
   updates_running_avg_h.append((running_avg_h[i], (1-alpha)*running_avg_h[i]+alpha*mean_h))
   updates_running_std_h.append((running_std_h[i], (1-alpha)*running_std_h[i]+alpha*std_h))
   updates_running_avg_z.append((running_avg_z[i], (1-alpha)*running_avg_z[i]+alpha*mean_z))
   updates_running_std_z.append((running_std_z[i], (1-alpha)*running_std_z[i]+alpha*std_z))
   updates_running_avg_r.append((running_avg_r[i], (1-alpha)*running_avg_r[i]+alpha*mean_r))
   updates_running_std_r.append((running_std_r[i], (1-alpha)*running_std_r[i]+alpha*std_r))

  # Output layer definition

  # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
  # I have to reshape both labels and p_y_given_x to fullfill this requirement.
  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))

  p_y_given_x = T.nnet.softmax(T.dot(h, W_o)+b_o)

  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)

  err=T.mean(T.neq(y_pred,y_lab_flat))

  # Gradient Computation
  gparams = T.grad(cost, param)

  # Memory memorization for RMSProp
  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(p, p - lr/(T.sqrt(memory)+epsilon) * gparam) for p, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg_h+updates_running_std_h+updates_running_avg_z+updates_running_std_z+updates_running_avg_r+updates_running_std_r+updates_mem
  
  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x
  
  
class M_GRU(object):
  
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)

  # input features and labels     
  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')

  # parameters list definition
  W_h=[]
  W_z=[]
 
  
  running_avg_h=[]
  running_std_h=[]
  running_avg_z=[]
  running_std_z=[]


  gamma_h=[]
  beta_h=[]
  gamma_z=[]
  beta_z=[]


  U_h=[]
  U_z=[]


  # Parameter initialization
  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid
   
   # Feed Forward Connections
   W_h.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_h',borrow=True))
   W_z.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+1),name='W_z',borrow=True))



   running_avg_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_h'))
   running_std_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_h'))
   running_avg_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_z'))
   running_std_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_z'))


   gamma_h.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_h'))
   beta_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_h'))
   gamma_z.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_z'))
   beta_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_z'))


   # Recurrent Connections
   U_h_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed)
   U_z_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+1)


   Q, R = np.linalg.qr(U_h_init)
   Q = Q * np.sign(np.diag(R))
   U_h.append(shared(Q,name='U_h',borrow=True))

   Q, R = np.linalg.qr(U_z_init)
   Q = Q * np.sign(np.diag(R))
   U_z.append(shared(Q,name='U_z',borrow=True))


  # output layer
  b_o=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')
  W_o=shared(init('glorot_red_f_0.0',(2*N_hid,N_out),seed),name='W_o',borrow=True)

  # Parameter for updating
  param=W_h+W_z+U_h+U_z+[W_o]+[b_o]+gamma_h+beta_h+gamma_z+beta_z
  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  # Parameter to save
  param_save=param+running_avg_h+running_std_h+running_avg_z+running_std_z+memory_grad

  # Update list initialization
  updates_running_avg_h=[]
  updates_running_std_h=[]
  updates_running_avg_z=[]
  updates_running_std_z=[]


  # Definition of the computations for each layer
  current_input=x
  for i in range(N_lay):
   W_conc=T.concatenate([W_h[i],W_z[i]],axis=1) # concatenation of W matrices
   at_conc=T.dot(current_input, W_conc)                # It is convenient to perform these dot products in parallel for all the time steps outside the scan.
   at_h=at_conc[:,:,0:N_hid] # de-concatenation
   at_z=at_conc[:,:,N_hid:N_hid*2]  #de-concatenation
  

   # Means and stds computations for batch norm 
   mean_h=T.mean(at_h,axis=0)
   std_h=T.std(at_h,axis=0)

   mean_h=(1-test_flag)*T.mean(mean_h,axis=0)+test_flag*running_avg_h[i]
   std_h=(1-test_flag)*T.mean(std_h,axis=0)+test_flag*running_std_h[i]

   mean_z=T.mean(at_z,axis=0)
   std_z=T.std(at_z,axis=0)

   mean_z=(1-test_flag)*T.mean(mean_z,axis=0)+test_flag*running_avg_z[i]
   std_z=(1-test_flag)*T.mean(std_z,axis=0)+test_flag*running_std_z[i]


   # batch norm computation
   at_h=T.nnet.bn.batch_normalization(at_h, gamma_h[i], beta_h[i], mean_h, std_h, mode='low_mem')
   at_z=T.nnet.bn.batch_normalization(at_z, gamma_z[i], beta_z[i], mean_z, std_z, mode='low_mem')
   

   # Concatenation for Bidirectional processing
   at_h=T.concatenate([at_h,at_h[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_z=T.concatenate([at_z,at_z[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 


   # Estimating dropout mask (same mask for each time-step)
   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor 

   h, _  = theano.scan(step_M_GRU,
                              sequences=[at_h,at_z],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[U_h[i],U_z[i],drop_mask],
                              truncate_gradient=-1)

   # Dividing h_back and h_forward and concatenation
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation
   current_input=h
 
   # Updated definitions
   updates_running_avg_h.append((running_avg_h[i], (1-alpha)*running_avg_h[i]+alpha*mean_h))
   updates_running_std_h.append((running_std_h[i], (1-alpha)*running_std_h[i]+alpha*std_h))
   updates_running_avg_z.append((running_avg_z[i], (1-alpha)*running_avg_z[i]+alpha*mean_z))
   updates_running_std_z.append((running_std_z[i], (1-alpha)*running_std_z[i]+alpha*std_z))


  # Output layer definition

  # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
  # I have to reshape both labels and p_y_given_x to fullfill this requirement.
  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))

  p_y_given_x = T.nnet.softmax(T.dot(h, W_o)+b_o)

  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)

  err=T.mean(T.neq(y_pred,y_lab_flat))

  # Gradient Computation
  gparams = T.grad(cost, param)

  # Memory memorization for RMSProp
  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(p, p - lr/(T.sqrt(memory)+epsilon) * gparam) for p, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg_h+updates_running_std_h+updates_running_avg_z+updates_running_std_z+updates_mem
  
  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x
  
  
class M_reluGRU(object):
  
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)

  # input features and labels     
  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')

  # parameters list definition
  W_h=[]
  W_z=[]
 
  
  running_avg_h=[]
  running_std_h=[]
  running_avg_z=[]
  running_std_z=[]


  gamma_h=[]
  beta_h=[]
  gamma_z=[]
  beta_z=[]


  U_h=[]
  U_z=[]


  # Parameter initialization
  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid
   
   # Feed Forward Connections
   W_h.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_h',borrow=True))
   W_z.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+1),name='W_z',borrow=True))


   running_avg_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_h'))
   running_std_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_h'))
   running_avg_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_z'))
   running_std_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_z'))

   gamma_h.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_h'))
   beta_h.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_h'))
   gamma_z.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_z'))
   beta_z.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_z'))


   # Recurrent Connections
   U_h_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed)
   U_z_init=init('glorot_red_f_0.01',(N_hid,N_hid),seed+1)


   Q, R = np.linalg.qr(U_h_init)
   Q = Q * np.sign(np.diag(R))
   U_h.append(shared(Q,name='U_h',borrow=True))

   Q, R = np.linalg.qr(U_z_init)
   Q = Q * np.sign(np.diag(R))
   U_z.append(shared(Q,name='U_z',borrow=True))


  # output layer
  b_o=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')
  W_o=shared(init('glorot_red_f_0.0',(2*N_hid,N_out),seed),name='W_o',borrow=True)

  # Parameter for updating
  param=W_h+W_z+U_h+U_z+[W_o]+[b_o]+gamma_h+beta_h+gamma_z+beta_z
  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  # Parameter to save
  param_save=param+running_avg_h+running_std_h+running_avg_z+running_std_z+memory_grad

  # Update list initialization
  updates_running_avg_h=[]
  updates_running_std_h=[]
  updates_running_avg_z=[]
  updates_running_std_z=[]


  # Definition of the computations for each layer
  current_input=x
  for i in range(N_lay):
   W_conc=T.concatenate([W_h[i],W_z[i]],axis=1) # concatenation of W matrices
   at_conc=T.dot(current_input, W_conc)                # It is convenient to perform these dot products in parallel for all the time steps outside the scan.
   at_h=at_conc[:,:,0:N_hid] # de-concatenation
   at_z=at_conc[:,:,N_hid:N_hid*2]  #de-concatenation
  

   # Means and stds computations for batch norm 
   mean_h=T.mean(at_h,axis=0)
   std_h=T.std(at_h,axis=0)

   mean_h=(1-test_flag)*T.mean(mean_h,axis=0)+test_flag*running_avg_h[i]
   std_h=(1-test_flag)*T.mean(std_h,axis=0)+test_flag*running_std_h[i]

   mean_z=T.mean(at_z,axis=0)
   std_z=T.std(at_z,axis=0)

   mean_z=(1-test_flag)*T.mean(mean_z,axis=0)+test_flag*running_avg_z[i]
   std_z=(1-test_flag)*T.mean(std_z,axis=0)+test_flag*running_std_z[i]


   # batch norm computation
   at_h=T.nnet.bn.batch_normalization(at_h, gamma_h[i], beta_h[i], mean_h, std_h, mode='low_mem')
   at_z=T.nnet.bn.batch_normalization(at_z, gamma_z[i], beta_z[i], mean_z, std_z, mode='low_mem')
   

   # Concatenation for Bidirectional processing
   at_h=T.concatenate([at_h,at_h[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_z=T.concatenate([at_z,at_z[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 


   # Estimating dropout mask (same mask for each time-step)
   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor 

   h, _  = theano.scan(step_M_reluGRU,
                              sequences=[at_h,at_z],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[U_h[i],U_z[i],drop_mask],
                              truncate_gradient=-1)

   # Dividing h_back and h_forward and concatenation
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation
   current_input=h
 
   # Updated definitions
   updates_running_avg_h.append((running_avg_h[i], (1-alpha)*running_avg_h[i]+alpha*mean_h))
   updates_running_std_h.append((running_std_h[i], (1-alpha)*running_std_h[i]+alpha*std_h))
   updates_running_avg_z.append((running_avg_z[i], (1-alpha)*running_avg_z[i]+alpha*mean_z))
   updates_running_std_z.append((running_std_z[i], (1-alpha)*running_std_z[i]+alpha*std_z))


  # Output layer definition

  # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
  # I have to reshape both labels and p_y_given_x to fullfill this requirement.
  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))

  p_y_given_x = T.nnet.softmax(T.dot(h, W_o)+b_o)

  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)

  err=T.mean(T.neq(y_pred,y_lab_flat))

  # Gradient Computation
  gparams = T.grad(cost, param)

  # Memory memorization for RMSProp
  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(p, p - lr/(T.sqrt(memory)+epsilon) * gparam) for p, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg_h+updates_running_std_h+updates_running_avg_z+updates_running_std_z+updates_mem
  
  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x
  

class LSTM(object):
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)
  
  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')

  W_f=[]
  W_i=[]
  W_o=[]
  W_c=[]

  running_avg_f=[]
  running_std_f=[]
  running_avg_i=[]
  running_std_i=[]
  running_avg_o=[]
  running_std_o=[]
  running_avg_c=[]
  running_std_c=[]

  gamma_f=[]
  beta_f=[]
  gamma_i=[]
  beta_i=[]
  gamma_o=[]
  beta_o=[]
  gamma_c=[]
  beta_c=[]

  U_f=[]
  U_i=[]
  U_o=[]

  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid

   # Feed Forward Connections
   W_f.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_f',borrow=True))
   W_i.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+1),name='W_i',borrow=True))
   W_o.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+2),name='W_o',borrow=True))
   W_c.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed+3),name='W_c',borrow=True))

   running_avg_f.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_f'))
   running_std_f.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_f'))
   running_avg_i.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_i'))
   running_std_i.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_i'))
   running_avg_o.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_o'))
   running_std_o.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_o'))
   running_avg_c.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg_c'))
   running_std_c.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std_c'))

   gamma_f.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_f'))
   beta_f.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_f'))

   gamma_i.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_i'))
   beta_i.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_i'))

   gamma_o.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_o'))
   beta_o.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_o'))

   gamma_c.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma_c'))
   beta_c.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta_c'))

   # Recurrent Connections
   U_f.append(shared(init('glorot_red_f_0.01',(N_hid,N_hid),seed),name='U_f',borrow=True))
   U_i.append(shared(init('glorot_red_f_0.01',(N_hid,N_hid),seed+1),name='U_i',borrow=True))
   U_o.append(shared(init('glorot_red_f_0.01',(N_hid,N_hid),seed+2),name='U_o',borrow=True))

  # output layer
  b_out=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')
  W_out=shared(init('glorot_red_f_0.0',(2*N_hid,N_out),seed),name='W_o',borrow=True)

  param=W_f+W_i+W_o+W_c+U_f+U_i+U_o+[W_out]+[b_out]+gamma_f+gamma_i+gamma_o+gamma_c+beta_f+beta_i+beta_o+beta_c
  param_save=param+running_avg_f+running_std_f+running_avg_i+running_std_i+running_avg_o+running_std_o+running_avg_c+running_std_c

  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  current_input=x

  updates_running_avg_f=[]
  updates_running_std_f=[]
  updates_running_avg_i=[]
  updates_running_std_i=[]
  updates_running_avg_o=[]
  updates_running_std_o=[]
  updates_running_avg_c=[]
  updates_running_std_c=[]

  for i in range(N_lay):

   W_conc=T.concatenate([W_f[i],W_i[i],W_o[i],W_c[i]],axis=1) # concatenation

   at_conc=T.dot(current_input, W_conc)  # I can perform these dot products outside the scan and all in parallel!!
   at_f=at_conc[:,:,0:N_hid] # de-concatenation
   at_i=at_conc[:,:,N_hid:N_hid*2]
   at_o=at_conc[:,:,N_hid*2:N_hid*3]
   at_c=at_conc[:,:,N_hid*3:N_hid*4]
 
   mean_f=T.mean(at_f,axis=0)
   std_f=T.std(at_f,axis=0)
   mean_f=(1-test_flag)*T.mean(mean_f,axis=0)+test_flag*running_avg_f[i]
   std_f=(1-test_flag)*T.mean(std_f,axis=0)+test_flag*running_std_f[i]

   mean_i=T.mean(at_i,axis=0)
   std_i=T.std(at_i,axis=0)
   mean_i=(1-test_flag)*T.mean(mean_i,axis=0)+test_flag*running_avg_i[i]
   std_i=(1-test_flag)*T.mean(std_i,axis=0)+test_flag*running_std_i[i]

   mean_o=T.mean(at_o,axis=0)
   std_o=T.std(at_o,axis=0)
   mean_o=(1-test_flag)*T.mean(mean_o,axis=0)+test_flag*running_avg_o[i]
   std_o=(1-test_flag)*T.mean(std_o,axis=0)+test_flag*running_std_o[i]

   mean_c=T.mean(at_c,axis=0)
   std_c=T.std(at_c,axis=0)
   mean_c=(1-test_flag)*T.mean(mean_c,axis=0)+test_flag*running_avg_c[i]
   std_c=(1-test_flag)*T.mean(std_c,axis=0)+test_flag*running_std_c[i]

   at_f=T.nnet.bn.batch_normalization(at_f, gamma_f[i], beta_f[i], mean_f, std_f, mode='low_mem')
   at_i=T.nnet.bn.batch_normalization(at_i, gamma_i[i], beta_i[i], mean_i, std_i, mode='low_mem')
   at_o=T.nnet.bn.batch_normalization(at_o, gamma_o[i], beta_o[i], mean_o, std_o, mode='low_mem')
   at_c=T.nnet.bn.batch_normalization(at_c, gamma_c[i], beta_c[i], mean_c, std_c, mode='low_mem')

   # Concatenation for Bidirectional processing
   at_f=T.concatenate([at_f,at_f[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_i=T.concatenate([at_i,at_i[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_o=T.concatenate([at_o,at_o[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 
   at_c=T.concatenate([at_c,at_c[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 

   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor

   [h,c], _  = theano.scan(step_LSTM,
                              sequences=[at_f,at_i,at_o,at_c],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid]),T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[U_f[i],U_i[i],U_o[i],drop_mask],
                              truncate_gradient=-1)

   #cost=T.nnet.categorical_crossentropy(p_y_given_x,y_lab)
   # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
   # I have to reshape both labels and p_y_given_x to fullfill this requirement.

   # Dividing h_back and h_forward and concatenation
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation
   current_input=h
 
   updates_running_avg_f.append((running_avg_f[i], (1-alpha)*running_avg_f[i]+alpha*mean_f))
   updates_running_std_f.append((running_std_f[i], (1-alpha)*running_std_f[i]+alpha*std_f)) 
   updates_running_avg_i.append((running_avg_i[i], (1-alpha)*running_avg_i[i]+alpha*mean_i))
   updates_running_std_i.append((running_std_i[i], (1-alpha)*running_std_i[i]+alpha*std_i)) 
   updates_running_avg_o.append((running_avg_o[i], (1-alpha)*running_avg_o[i]+alpha*mean_o))
   updates_running_std_o.append((running_std_o[i], (1-alpha)*running_std_o[i]+alpha*std_o))
   updates_running_avg_c.append((running_avg_c[i], (1-alpha)*running_avg_c[i]+alpha*mean_c))
   updates_running_std_c.append((running_std_c[i], (1-alpha)*running_std_c[i]+alpha*std_c))

 
 
  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))

  p_y_given_x = T.nnet.softmax(T.dot(h, W_out)+b_out)

  #p_y_m = T.reshape(p_y_given_x, (p_y_given_x.shape[0] * p_y_given_x.shape[1], -1))
  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)
  err=T.mean(T.neq(y_pred,y_lab_flat))


  gparams = T.grad(cost, param)

  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(param, param - lr/(T.sqrt(memory)+epsilon) * gparam) for param, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg_f+updates_running_std_f+updates_running_avg_i+updates_running_std_i+updates_running_avg_o+updates_running_std_o+updates_running_avg_c+updates_running_std_c+updates_mem

  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x
  
class reluRNN(object):
 def __init__(self, index,alpha,test_flag,lr,batch_size_gh,options):
   
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
  current_lr=float(options.learning_rate)
  dropout_factor=float(options.dropout_factor)
  alpha_tr=float(options.alpha)
  alpha_mem=float(options.alpha_mem)
  epsilon=float(options.epsilon)
  
  
  # Parameter initialization
  W_in=[]
  W=[]
  gamma=[]
  beta=[]
  running_avg=[]
  running_std=[]

  for i in range(N_lay): 
   if i==0:
     n_in=N_fea
   else:
     n_in=2*N_hid
 
   W_in.append(shared(init('glorot_red_f_0.01',(n_in,N_hid),seed),name='W_in',borrow=True))
   W.append(shared(init('glorot_red_f_0.01',(N_hid,N_hid),seed),name='W',borrow=True))
   gamma.append(shared((np.asarray(np.zeros([N_hid])+1.,dtype=theano.config.floatX)),name='gamma'))
   beta.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='beta'))
   running_avg.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_avg'))
   running_std.append(shared((np.asarray(np.zeros([N_hid]),dtype=theano.config.floatX)),name='running_std'))

  W_out=shared(init('glorot_red_f_0.01',(2*N_hid,N_out),seed),name='W_out',borrow=True)
  b_o=shared((np.asarray(np.zeros(N_out,)+0.0,dtype=theano.config.floatX)),name='b_o')

  param= W_in+W+gamma+beta+[W_out]+[b_o]
  param_save=param+running_avg+running_std


  x=shared(np.zeros([600,batch_size,N_fea],dtype=theano.config.floatX),name='x')
  y_lab=shared(np.zeros([600,batch_size,1],dtype='int32'),name='y_lab')


  # Memory for the gradient
  memory_grad = [theano.shared(np.zeros_like(p.get_value()))for p in param]

  updates_running_avg=[]
  updates_running_std=[]
  current_input=x

  for i in range(N_lay): 
   at=T.dot(current_input, W_in[i])
   mean=T.mean(at,axis=0)
   std=T.std(at,axis=0)

   mean=(1-test_flag)*T.mean(mean,axis=0)+test_flag*running_avg[i]
   std=(1-test_flag)*T.mean(std,axis=0)+test_flag*running_std[i]

   at=T.nnet.bn.batch_normalization(at, gamma[i], beta[i], mean, std, mode='low_mem')
   at=T.concatenate([at,at[::-1]],axis=1) #reversing time-steps for bi-RNN and concatenation 

   # dropout RNN
   srng = RandomStreams(seed=seed)
   drop_mask = (1-test_flag)*T.switch(srng.binomial(size=(2*batch_size_gh,N_hid),p=dropout_factor),1,0)+ test_flag*dropout_factor 

   updates_running_avg.append((running_avg[i], (1-alpha)*running_avg[i]+alpha*mean))
   updates_running_std.append((running_std[i], (1-alpha)*running_std[i]+alpha*std))
 
   h, _  = theano.scan(step_reluRNN,
                              sequences=[at],
                              outputs_info=[T.zeros([2*batch_size_gh,N_hid])],
                              non_sequences=[W[i],drop_mask],
                              truncate_gradient=-1)
 

   #cost=T.nnet.categorical_crossentropy(p_y_given_x,y_lab)
   # The categorical crossentropy is defined over between matices (2D tensors) and 1-d vectors.
   # I have to reshape both labels and p_y_given_x to fullfill this requirement.
   h_forward=h[:,0:batch_size_gh,:]
   h_back=h[:,batch_size_gh:,:]
   h_back=h_back[::-1] # Reverting time steps
   h=T.concatenate([h_forward,h_back],axis=2) # concatenation

   current_input=h

  h=T.reshape(h, (h.shape[0] * h.shape[1], -1))
  p_y_given_x = T.nnet.softmax(T.dot(h, W_out)+b_o)
  #p_y_m = T.reshape(p_y_given_x, (p_y_given_x.shape[0] * p_y_given_x.shape[1], -1))
  y_lab_flat = y_lab.flatten(ndim=1)

  cost=T.mean(T.nnet.categorical_crossentropy(p_y_given_x,y_lab_flat))

  y_pred=T.argmax(p_y_given_x,axis=1)
  err=T.mean(T.neq(y_pred,y_lab_flat))

  gparams = T.grad(cost, param)

  memory_new = [alpha_mem*memory + (1-alpha_mem)*gparam ** 2 for gparam,memory in zip(gparams,memory_grad)]

  updates = [(param, param - lr/(T.sqrt(memory)+epsilon) * gparam) for param, gparam,memory in zip(param, gparams,memory_new)]

  updates_mem = [(memory, memory_new_par) for memory,memory_new_par in zip(memory_grad,memory_new)]

  updates=updates+updates_running_avg+updates_running_std+updates_mem

  self.cost=cost
  self.err=err
  self.updates=updates
  self.param_save=param_save
  self.x=x 
  self.y_lab=y_lab
  self.p_y_given_x=p_y_given_x