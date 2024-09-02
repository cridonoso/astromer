#!/bin/bash

echo Getting ids...
python -m presentation.pipelines.machofields.get_ids

while IFS=',' read -r col1 field_n idcode
do  
    if [ "$col1" != "" ]; then
        echo Processing $field_n
        python -m presentation.pipelines.machofields.download --id $idcode \
                                                              --field $field_n \
                                                              --target ./data/records/bigmacho
        echo Transforming to records...
        python -m presentation.pipelines.machofields.to_record --config ./data/records/bigmacho/$field_n/config.toml

        echo 'Deleting raw data'
        rm -rf ./data/temp/light_curves
        rm -rf ./data/temp/metadata.parquet                                               
    fi
done < ./data/temp/ids.csv
