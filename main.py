import numpy as np
import h5py
import pandas as pd

rightHandW = pd.read_csv('rightHandW.csv')
backRightPocketW = pd.read_csv('backRightPocketW.csv')
leftJacketW = pd.read_csv('leftJacketW.csv')
rightFrontPocketW = pd.read_csv('rightFrontPocketW.csv')
rightJacketW = pd.read_csv('rightJacketW.csv')

rightHandJ = pd.read_csv('rightHandJ.csv')
backRightPocketJ = pd.read_csv('backRightPocketJ.csv')
leftJacketJ = pd.read_csv('leftJacketPocketJ.csv')
rightFrontPocketJ = pd.read_csv('rightFrontPocketJ.csv')
rightJacketJ = pd.read_csv('rightJacketPocketJ.csv')

with h5py.File('data.h5', 'w') as hdf:
    # # Combined Dataset Creation
    # combined_DataSet = hdf.create_group('/mainDataset')
    # combined_DataSet.create_dataset('mainDataset', data=matrix_2)
    #
    # trainSet = hdf.create_group('/mainDataset/train')
    # trainSet.create_dataset('mainDataset', data=matrix_2)
    #
    # testSet = hdf.create_group('/mainDataset/test')
    # testSet.create_dataset('mainDataset', data=matrix_2)

    # Matt's Dataset Creation
    # Walking
    Matt_Group = hdf.create_group('/Matt_Group')
    Matt_Group.create_dataset('mbrpw', data=backRightPocketW)
    Matt_Group.create_dataset('mrhw', data=rightHandW)
    Matt_Group.create_dataset('mrjw', data=rightJacketW)
    Matt_Group.create_dataset('mrfpw', data=rightFrontPocketW)
    Matt_Group.create_dataset('mljw', data=leftJacketW)
    # Jumping
    Matt_Group.create_dataset('mbrpj', data=backRightPocketJ)
    Matt_Group.create_dataset('mrhj', data=rightHandJ)
    Matt_Group.create_dataset('mrjj', data=rightJacketJ)
    Matt_Group.create_dataset('mrfpj', data=rightFrontPocketJ)
    Matt_Group.create_dataset('mljj', data=leftJacketJ)

    with h5py.File('data.h5', 'r') as hdf:
        items = list(hdf.items())
        print(items)
        Matt_Group = hdf.get('/Matt_Group')
        print(list(Matt_Group.items()))
        d1 = Matt_Group.get('mljj')
        d1 = np.array(d1)
        print(d1.shape)


    # Warren's Dataset Creation
    Warren_Group = hdf.create_group('/Warren_Group')
    Warren_Group.create_dataset('wbrpw', data=backRightPocketW)
    Warren_Group.create_dataset('wrhw', data=rightHandW)
    Warren_Group.create_dataset('wrjw', data=rightJacketW)
    Warren_Group.create_dataset('wrfpw', data=rightFrontPocketW)
    Warren_Group.create_dataset('wljw', data=leftJacketW)
    # Jumping
    Warren_Group.create_dataset('wbrpj', data=backRightPocketJ)
    Warren_Group.create_dataset('wrhj', data=rightHandJ)
    Warren_Group.create_dataset('wrjj', data=rightJacketJ)
    Warren_Group.create_dataset('wrfpj', data=rightFrontPocketJ)
    Warren_Group.create_dataset('wljj', data=leftJacketJ)


    # # Ellen's Dataset Creation
    # Ellen_Group = hdf.create_group('/Ellen_Group')
    # Ellen_Group.create_dataset('dataset_test', data=matrix_1)