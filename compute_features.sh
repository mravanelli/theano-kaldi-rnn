#!/bin/bash

# Compute features

#Mirco Ravanelli – April 2017 (mravanelli@fbk.eu)

#This script converts both features and labels from the kaldi format (ark) to the pkl used in Theano.
#Part of the codes reported here are insipred by the theano-kaldi project (https://github.com/shawntan/theano-kaldi)
#Make sure you already run the kaldi recipe for the TIMIT dataset.  Make also sure to have the aligments (label) for training, dev and test data. Alignements of training data are already generated withing “run.sh” of the kaldi-trunk/egs/timit/s5. To generate aligments of dev and test sentences you have to add in run.sh the following lines:

#steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri3/graph data/dev exp/tri3/decode_dev

#steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# exp/tri3/graph data/test exp/tri3/decode_test

# Add kaldi paths
$KALDI_ROOT/egs/timit/s5/path.sh

# Alignment folders
ali_dir=/home/mirco/kaldi-trunk/egs/timit/s5/exp/tri3_ali
ali_dir_dev=/home/mirco/kaldi-trunk/egs/timit/s5/exp/tri3_ali_dev
ali_dir_test=/home/mirco/kaldi-trunk/egs/timit/s5/exp/tri3_ali_test

# Path of data in kaldi format
datadir=/home/mirco/kaldi-trunk/egs/timit/s5/data

# Output folder
out_fold=/home/mirco/theano-kaldi-master/timit_mfcc
mkdir $out_fold

# Initial preprocessing for input feature (CMNV transform)
copy-feats scp:$datadir/train/feats.scp ark:- \
| add-deltas --delta-order=2 ark:- ark:- \
| compute-cmvn-stats ark:- - \
| cmvn-to-nnet --binary=false - $out_fold/feature_transform  || exit 1;

# List of sentences to avoid (there could be some sentences without alignements. We have to avoid them)
cat $ali_dir/log/*align_pass2.*.log | grep 'Did not successfully decode file' | awk '{print $8}' | sed 's/,//' > $out_fold/avoid_list_tr.txt
cat $ali_dir_dev/log/*align_pass2.*.log | grep 'Did not successfully decode file' | awk '{print $8}' | sed 's/,//' > $out_fold/avoid_list_dev.txt
cat $ali_dir_test/log/*align_pass2.*.log | grep 'Did not successfully decode file' | awk '{print $8}' | sed 's/,//' > $out_fold/avoid_list_test.txt

# Set transformation
feat_transform="\
add-deltas --delta-order=2 ark:- ark:- |\
nnet-forward $out_fold/feature_transform ark:- ark:- \
"

# Converting training features
num_split=10
./prepare_pickle.sh $num_split \
    $datadir/train/feats.scp \
    $ali_dir \
    $out_fold/train \
    $out_fold/_log/split \
    $out_fold/avoid_list_tr.txt \
    "$feat_transform" || exit 1;

# Converting dev features
num_split=1
./prepare_pickle.sh $num_split \
    $datadir/dev/feats.scp \
    $ali_dir_dev \
    $out_fold/dev \
    $out_fold/_log/split \
    $out_fold/avoid_list_dev.txt \
    "$feat_transform" || exit 1;

# Converting test features
num_split=1
./prepare_pickle.sh $num_split \
    $datadir/test/feats.scp \
    $ali_dir_test \
    $out_fold/test \
    $out_fold/_log/split \
    $out_fold/avoid_list_test.txt \
    "$feat_transform" || exit 1;






