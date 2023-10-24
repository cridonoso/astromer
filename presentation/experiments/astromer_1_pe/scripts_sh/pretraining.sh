##################################################################################
########################## WITHOUT POSITIONAL ENCODING ##########################
##################################################################################
# Not pe module
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'not_pe_module' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

##################################################################################
########################## ABSOLUTE POSITIONAL ENCODING ##########################
##################################################################################

# Positional Encoding Base to use ( Me falta el TUPE-A y el Concat ) [First paper]
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_000 --repeat 1 --pe-type 'APE' --pe-func-name 'pe' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'pe_mlp' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'pe_rnn' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'pe_tm' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'pe_gp' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

# Variations in PE connections [Second paper]
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_001 --repeat 1 --pe-type 'APE' --pe-func-name 'pe' --residual-type 'residual_in_all_attblocks' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_002 --repeat 1 --pe-type 'APE' --pe-func-name 'pe' --residual-type 'residual_in_last_attblock' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2


##################################################################################
########################## RELATIVE POSITIONAL ENCODING ##########################
##################################################################################
# To continuous time (my implementation)
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'RPE' --pe-func-name None --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

###########################################################################################
########################## MIXED (Abs + Rel) POSITIONAL ENCODING ##########################
###########################################################################################
# DeBERTa with my implementation (residual_in_last_attblock + pe_relative)
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_003 --repeat 1 --pe-type 'MixPE' --pe-func-name 'pe' --residual-type 'residual_in_last_attblock' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2


###########################################################################################################################################################

###########################################################################
########################## # PE OUT OF Attention ##########################
###########################################################################
# PE out of Attention
python -m presentation.experiments.astromer_1_pe.scripts.pretraining \
    --gpu 0 --pt-folder pretraining/P05R02 --exp-name exp_xxx --repeat 1 --pe-type 'APE' --pe-func-name 'pea' --dropout 0.0 --lr '1e-3' --probed 0.5 --rs 0.2

