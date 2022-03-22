from absl import app
from absl import flags
from google_drive_downloader import GoogleDriveDownloader as gdd


dataset_id = {
	'raw':{
		'ogle'   : '',
		'macho'  : '',
		'alcock'  : '1JdEPQ3vaTEFscrZUiaKG24HV_9BDx90R',
		'atlas' : '',
	},
	'record':{
		'ogle'   : '1RY3pwZ5uYJ9HvlNUlivt-DsIbBXHYWx2',
		'macho'  : '1O-cFXWjuTMNNfWjVQCyEAa2IoAJFSPI6',
		'alcock'  : '1whdP_2SdMSHcODu8ItTWfM0tI-frGhie',
		'atlas' : '1zFkrP2TCYoGa3yWkrtymVDmWjLlViIpp',
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
		dest_path='./{}/records/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)
	else:
		file_id = dataset_id['raw'][FLAGS.dataset]
		dest_path='./{}/raw_data/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)


	gdd.download_file_from_google_drive(file_id=file_id,
										dest_path=dest_path,
										unzip=FLAGS.unzip)


if __name__ == '__main__':
	app.run(main)
