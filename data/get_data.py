from absl import app
from absl import flags
from google_drive_downloader import GoogleDriveDownloader as gdd


dataset_id = {
	'raw':{
		'ztf': '1RY3pwZ5uYJ9HvlNUlivt-DsIbBXHYWx2',
		'wise': '1O-cFXWjuTMNNfWjVQCyEAa2IoAJFSPI6',
		'ogle': '1whdP_2SdMSHcODu8ItTWfM0tI-frGhie',
		'macho': '1zFkrP2TCYoGa3yWkrtymVDmWjLlViIpp',
		'linear': '10E6e4E75TPfgjxYvS6IBb4pveXxB-YNM',
		'gaia': '1VgG2RNi88VcVnpE_MgOYPFjS8A42KXTJ',
		'css': '1IUGvcYQViJxh-Isz9FC6yAdDOuU2mxlf',
		'asas': '1ouQmBedVok5LNH2u2aEQjAvl0KchrmTI',
	},
	'record':{
		'ztf': '',
		'wise': '1zfRvNtzBgdkM1l3_Drhq3BEJoGapp2R6',
		'ogle': '',
		'macho': '1t2M7aqM4JiwapN-CR3Gd36AWwRE1efkB',
		'linear': '1NDZh3bu-6GDfSwpqBNVg-bZk5PAmi2lK',
		'gaia': '1EgLYoqHLlFHxAvXKXM7KsMwvRfMwIQiK',
		'css': '1f-3tJojGQoI6dyq_teCr2OOa19uGmMgO',
		'asas': '1RJjYcMePgVT2mC2x3T7p_B5l23pFEM_C',
		'large_macho':'1Bzi9U30BZVHuwimn2KAATARlRffH8iGP',
	}
}

FLAGS = flags.FLAGS
flags.DEFINE_boolean('record', False, 'Get record if available')
flags.DEFINE_boolean('unzip', True, 'Unzip compressed file')

flags.DEFINE_string("destination", ".", "Folder for saving files")

flags.DEFINE_string("dataset", "macho", "Dataset to be downloaded (macho, linear, asas, wise, gaia, css, ogle)")


def main(argv):
	if FLAGS.record:
		file_id = dataset_id['record'][FLAGS.dataset]
		dest_path='{}/records/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)
	else:
		file_id = dataset_id['raw'][FLAGS.dataset]
		dest_path='{}/raw_data/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)


	gdd.download_file_from_google_drive(file_id=file_id,
										dest_path=dest_path,
										unzip=FLAGS.unzip)


if __name__ == '__main__':
	app.run(main)
