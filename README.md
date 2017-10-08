## Introduction:

THEANO-KALDI-RNNs is a software which offers the possibility to use various Recurrent Neural Networks (RNNs) in the context of a Kaldi-based hybrid HMM/RNN speech recognizer.
 
The current version supports the following standard architectures:
- reluRNN
- LSTM
- GRU
 
The code also considers some architectural variations:
- ReluGRU
- Minimal Gated Recurrent Units (M_GRU)
- M-reluGRU (also known as Light-GRU)
 
The latter architectures have been explored in [1] (see reference). 
Please cite this paper if you use this toolkit or a part of it.
 
All the RNNs are based on a state-of-the-art technology which includes:

- Bidirectional architectures
- Bach Normalization (applied to feed-forward connections)
- Recurrent Dropout
- RMSE prop optimization
 
## Prerequisites:

- If not already done, install KALDI (http://kaldi-asr.org/) and make sure that your KALDI installation is working. 

- Run the original  TIMIT kaldi recipe in *egs/timit/s5/run.sh* and check whether everything is properly working. This step is necessary to compute features and labels that will be inherited in the theano/python part of this code. 

- Install THEANO (http://deeplearning.net/software/theano/install.html) and make sure your installation is working. Try for instance to  type *import theano* in the python environment and check whether everything works fine. 
 
The code has been tested with:
- Python  2.7 
- Ubuntu 14 and RedHat (6,7)
- Theano 0.8.1 
 
## How to run an experiment:

#### 1. Run the Kaldi s5 baseline of TIMIT.  
This step is necessary to  derive the labels later used to train the RNN.  In particular: 
- go to *$KALDI_ROOT/egs/timit/s5*.
- run the script *run.sh*. Make sure everything (especially the *tri3-ali* part) works fine. Note that the s5 recipe computes tri3-ali for training data only. Please, computed them for test and dev data as well with the following commands:
``` 
steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 24 --cmd "$train_cmd" \
 data/test data/lang exp/tri3 exp/tri3_ali_test
```
            
 
#### 2. Convert kaldi features and labels into pkl format. 
- Set your own paths in  *compute_features.sh* (ali_dir,ali_dir_dev,ali_dir_test,data_dir,...)
- Run *compute_features.sh*.
 
#### 3. Write the Config file. 
- Open the file *TIMIT_GRU.cfg* and modify it according to your paths.  Feel free to modify the DNN architecture and the other optimization parameters according to your needs. See the comments in the  TIMIT_GRU.cfg file for a brief description of each parameter. The number of outputs *N_out* can be found with the following kaldi command (see number of pdfs):
``` 
am-info exp/tri3/final.mdl
``` 
- The required *count_file* in the config file (used to normalize the DNN posteriors before feeding the decoder) corresponds to the following file:

*/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts*

 
#### 4. Run the experiment. 
- Open the file *run_exp.sh*
- Set cfg_file, graph_dir, data_dir, ali_dir  according to your specific paths
- To replicate GRU experiments of the paper [1], set *cfg_file=TIMIT_GRU.cfg* in *run_exp.sh*
- To replicate M_reluGRU experiments (improved architecture) of the paper [1], set *cfg_file=TIMIT_M_reluGRU.cfg* in *run_exp.sh*

#### 5. Check the results.
- After training, forward and decoding phases are finished, you can go into the *kaldi_decoding_scripts* foder and run *./RESULT* to check the system performance.  
 
- Note that the performance obtained can be slightly  different from that reported in the paper due, for instance, to the randomness introduced by different initializations. To mitigate this source of randomness and perform a fair comparison across the various architectures, in [1] we ran  more experiments with different seeds (i.e., setting a different seed in the cfg_file) and we averaged the obtained error rates. 
 
 
Please, note that this is an ongoing project. It would be helpful to report us any issue!
 
## Reference:
*[1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio, "Batch-normalized joint training for DNN-based distant speech recognition", in Proceedings of Interspeech 2017*

https://arxiv.org/abs/1710.00641
 
