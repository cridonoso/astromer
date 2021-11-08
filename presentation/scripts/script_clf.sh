#!/bin/bash
NAME=''
for i in "0" "1" "2"
do
  if test $i == 0; 
  then 
      NAME='lstm_att' 
  fi
  if test $i == 1; 
  then 
      NAME='mlp_att' 
  fi
  if test $i == 2; 
  then 
      NAME='lstm' 
  fi

  echo $NAME
  python classification.py --data ../../data/records/alcock --max-obs 200 --w  ../../weights/astromer_10022021/finetuning/alcock --mode $i --take 100 --p ../../exp_clf/alcock/$NAME
done
