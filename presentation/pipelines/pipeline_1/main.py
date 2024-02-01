import sys

import mlflow

from src.pipelines.steps import build_tf_data_loader, \
								build_model


def pipeline():
 	
	pt_model, config = load_pretrained_model(sys.argv[0])

	print(config)
	# tf_data = build_tf_data_loader(sys.argv[1], 
	# 							   probed, 
	# 						       random_same, 
	# 							   num_cls=None, 
	# 							   batch_size=5)


if __name__ == "__main__":
	pipeline()

