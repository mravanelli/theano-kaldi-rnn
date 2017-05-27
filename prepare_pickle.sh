#!/bin/bash

$KALDI_ROOT/egs/timit/s5/path.sh

# Arguments
num_split=$1
fea_file=$2
ali_dir=$3
output_prefix=$4
log_dir=$5
avoid_list=$6
feat_transform=$7


# Splitting the lists of feature
tmp_dir=$(mktemp -d)
mkdir -p $log_dir

cat $fea_file | awk '{print $1}' > name_list.txt

total_lines=$(wc -l <name_list.txt)
((lines_per_file = (total_lines + num_split - 1) / num_split))
split -d -a 3 --lines=${lines_per_file} name_list.txt "$tmp_dir/split."


for f in $tmp_dir/split.*; do 
idx=${f##*.}
 # Converting the features
 cat $fea_file | grep -F -f $f | grep -F -v -f $avoid_list | copy-feats scp:- ark:- | eval $feat_transform  | python -u pickle_ark_stream.py  $output_prefix"."$idx".pklgz"

 # Converting the labels
 gunzip -c `ls -v $ali_dir/ali*.gz` | ali-to-pdf $ali_dir'/final.mdl' ark:- ark,t:-  | grep -F -f $f | grep -F -v -f  $avoid_list | python pickle_ali.py $output_prefix"."$idx"_lbl.pklgz"

done


rm -rf $tmp_dir