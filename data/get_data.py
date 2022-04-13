#!/bin/bash
declare -A dictionary
dictionary["alcock"]="1ISAlSKVDcULt9TJR3cPYs1sCx0w8KTwB"
dictionary["macho"]="1vWEs_IRGItmxmpWktvCqNx53uzC4o3O3"
dictionary["ogle"]="1BSOA8J78VsNLQ_HZ9wZGlEDC1Rh5BKHt"
dictionary["atlas"]="1ILHb_EMr09jyfWwyyqf0c2qglrnnSz59"

dictionary["alcock-record"]="1YpznRml85u_QSMH75lByMEHmNJQiCdcx"
dictionary["macho-record"]="1ejnuissFNAdczjxSh5IFy6QC9XFnGgAG"
dictionary["ogle-record"]="1pQ88cI74fwxcBnE7TBXACawc7z0wvMpk"
dictionary["atlas-record"]="1lIXWODXob5XwTqJ6rjFDdq5u-pliB4ML"

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

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $OUTFILE && rm -rf /tmp/cookies.txt


unzip $OUTFILE -d $DIR
rm -rf $OUTFILE
