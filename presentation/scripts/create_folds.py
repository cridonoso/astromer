import argparse

from core.data import create_dataset

meta_v2 = meta[meta['Std']>0.1]
meta_v2 = meta_v2[meta_v2['Skewness'].abs()>1]
meta_v2 = meta_v2[meta_v2['Kurtosis']>10]
meta_v2 = meta_v2[meta_v2['N']>20]


for fold_n in range(3):
    test_meta  = pd.concat([frame.sample(n=100) for g, frame in meta.groupby('Class')])
    train_meta = meta[~meta['ID'].isin(test_meta['ID'])]
    print(test_meta.shape, train_meta.shape)

    for nsamples in [0]:
        if nsamples == 0:
            partial_meta = train_meta
            target = './data/records/{}/fold_{}/{}'.format(name, fold_n, name)
        else:
            partial_meta = pd.concat([frame.sample(n=nsamples) for c, frame in train_meta.groupby('Class')])
            target = './data/records/{}/fold_{}/{}_{}'.format(name, fold_n, name, nsamples)

        create_dataset(partial_meta, source, target, max_lcs_per_record=20000,
                       n_jobs=7, subsets_frac=(0.8, 0.2), test_subset=test_meta,
                       names=['mjd', 'mag', 'errmag'],
                       delim_whitespace=True)

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
