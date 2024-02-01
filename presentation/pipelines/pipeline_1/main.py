import sys

import mlflow

from presentation.pipelines.steps import build_tf_data_loader, \
										 load_pt_model, \
										 finetune_model


def pipeline():
 	
 	# Loading pre-trained model
	pt_model, config = load_pt_model(sys.argv[1])
	
    # Update config to include finetune-related information
	config['ft_data'] = sys.argv[2]
	config['pt_path'] = sys.argv[1]


	# Loading labeled data using the same hyperparameters as in the pretraining 
	tf_data = build_tf_data_loader(sys.argv[2], 
								   config, # pretrained model config
								   batch_size=5)

	# Finetune model
	finetune_model(pt_model, tf_data, config)

if __name__ == "__main__":
	pipeline()

