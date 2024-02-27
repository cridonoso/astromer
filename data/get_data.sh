#!/bin/bash
declare -A dictionary
dictionary["alcock"]="18a4DGPlyJ21DI9HrKS-jJtaA2iIqFDQO"
dictionary["macho"]="1ReqDHn9jKPIS_g8Xm0ThYUDM-XaLcUP3"
dictionary["ogle"]="1L1oiq9pRRGpOVm13b2dyCQhMiVFn3o7x"
dictionary["atlas"]="1pMzeL9BAwMXqra9iFUHLpGyclJzz-VM9"

dictionary["alcock-record"]="1bEETbIgsVjhpkfR0LdYxnYQ8eeaq4wol"
dictionary["macho-record"]="1QLXAsTkaryYUqhjKAM0wh6XKh3tG1M0k"
dictionary["ogle-record"]="1Ei5PZ13LjJ44tA2iBOkDHPueGbLlTl2C"
dictionary["atlas-record"]="1e6dtsaidOBnbVo5gP8IkZXD_8ooiLL6d"

FILEID=${dictionary[$1]}
echo $FILEID

SUB="record"
if [[ "$1" == *"$SUB"* ]];
then
    NAME=${1%-*}
    mkdir -p records/
    DIR=./records/
    OUTFILE=./records/$NAME.zip
    echo $OUTFILE
else
    mkdir -p raw_data/$1
    DIR=./raw_data/$1
    OUTFILE=./raw_data/$1/$1.zip
fi

gdown https://drive.google.com/uc?id=$FILEID -O $OUTFILE -c

unzip $OUTFILE -d $DIR
rm -rf $OUTFILE
