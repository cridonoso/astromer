import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd

dataset_id = {
	'raw':{
		'ogle'   : '1BSOA8J78VsNLQ_HZ9wZGlEDC1Rh5BKHt',
		'macho'  : '1vWEs_IRGItmxmpWktvCqNx53uzC4o3O3',
		'alcock'  : '1ISAlSKVDcULt9TJR3cPYs1sCx0w8KTwB',
		'atlas' : '1ILHb_EMr09jyfWwyyqf0c2qglrnnSz59',
		'naul':'1-p5XA9ioYqKUhNoIQNEx_ySjE1X3G4AT'
	},
	'record':{
		'ogle'   : '1pQ88cI74fwxcBnE7TBXACawc7z0wvMpk',
		'macho'  : '1ejnuissFNAdczjxSh5IFy6QC9XFnGgAG',
		'alcock' : '1YpznRml85u_QSMH75lByMEHmNJQiCdcx',
		'atlas'  : '1lIXWODXob5XwTqJ6rjFDdq5u-pliB4ML',
	}
}

def run(opt):
	if opt.records:
		file_id = dataset_id['record'][opt.dataset]
		dest_path='./{}/records/{}.zip'.format(opt.p, opt.dataset)
	else:
		file_id = dataset_id['raw'][opt.dataset]
		dest_path='./{}/raw_data/{}.zip'.format(opt.p, opt.dataset)

	gdd.download_file_from_google_drive(file_id=file_id,
										dest_path=dest_path,
										unzip=opt.zip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--dataset', default='macho', type=str,
                        help='Dataset to be downloaded')
    parser.add_argument('--p', default="./data/", type=str,
                        help='Folder to store dataset')
    parser.add_argument('--records', default=False, action='store_true',
                        help='Get record if available')
    parser.add_argument('--zip', default=True, action='store_false',
                        help='Get record if available')

    opt = parser.parse_args()
    run(opt)
