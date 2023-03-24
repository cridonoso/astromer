#!/bin/bash
 

phycores=$(nproc)

echo $phycores  

for ((i=1;i<=phycores;++i))
do
   echo "Core $i"
   python3 mem_time_big.py $i
done
