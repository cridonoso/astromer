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

  for j in 10
  do
  echo $NAME
  python -m presentation.scripts.classification --data ./data/records/alcock_$j --max-obs 200 --w  ./runs/huge_5/finetuning/alcock_$j --mode $i --take 100 --p ./exp_clf/alcock_$j/$NAME
  done
done