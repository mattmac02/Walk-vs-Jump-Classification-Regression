from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import pandas as pd
import os

# Matt's Data
MrightHandW = pd.read_csv('Walking_Data/MrightHandW.csv')
MleftBackPocketW = pd.read_csv('Walking_Data/MleftBackPocketW.csv')
MleftJacketPocketW = pd.read_csv('Walking_Data/MleftJacketW.csv')
MrightFrontPocketW = pd.read_csv('Walking_Data/MrightFrontPocketW.csv')
MrightJacketPocketW = pd.read_csv('Walking_Data/MrightJacketW.csv')

MrightHandJ = pd.read_csv('Jumping_Data/MrightHandJ.csv')
MleftBackPocketJ = pd.read_csv('Jumping_Data/MleftBackPocketJ.csv')
MleftJacketPocketJ = pd.read_csv('Jumping_Data/MleftJacketPocketJ.csv')
MrightFrontPocketJ = pd.read_csv('Jumping_Data/MrightFrontPocketJ.csv')
MrightJacketPocketJ = pd.read_csv('Jumping_Data/MrightJacketPocketJ.csv')

# Warren's Data
WrightHandW = pd.read_csv('Walking_Data/WrightHandW.csv')
WleftBackPocketW = pd.read_csv('Walking_Data/WleftBackPocketW.csv')
WleftJacketPocketW = pd.read_csv('Walking_Data/WleftJacketPocketW.csv')
WrightFrontPocketW = pd.read_csv('Walking_Data/WrightFrontPocketW.csv')
WrightJacketPocketW = pd.read_csv('Walking_Data/WrightJacketPocketW.csv')

WrightHandJ = pd.read_csv('Jumping_Data/WrightHandJ.csv')
WleftBackPocketJ = pd.read_csv('Jumping_Data/WleftBackPocketJ.csv')
WleftJacketPocketJ = pd.read_csv('Jumping_Data/WleftJacketPocketJ.csv')
WrightFrontPocketJ = pd.read_csv('Jumping_Data/WrightFrontPocketJ.csv')
WrightJacketPocketJ = pd.read_csv('Jumping_Data/WrightJacketPocketJ.csv')

# Ellen's Data
ErightHandW = pd.read_csv('Walking_Data/ErightHandW.csv')
EleftBackPocketW = pd.read_csv('Walking_Data/EleftBackPocketW.csv')
EleftJacketPocketW = pd.read_csv('Walking_Data/EleftJacketPocketW.csv')
ErightFrontPocketW = pd.read_csv('Walking_Data/ErightFrontPocketW.csv')
ErightJacketPocketW = pd.read_csv('Walking_Data/ErightJacketPocketW.csv')

ErightHandJ = pd.read_csv('Jumping_Data/ErightHandJ.csv')
EleftBackPocketJ = pd.read_csv('Jumping_Data/EleftBackPocketJ.csv')
EleftJacketPocketJ = pd.read_csv('Jumping_Data/EleftJacketPocketJ.csv')
ErightFrontPocketJ = pd.read_csv('Jumping_Data/ErightFrontPocketJ.csv')
ErightJacketPocketJ = pd.read_csv('Jumping_Data/ErightJacketPocketJ.csv')

Matts_Dataset = pd.concat([MleftBackPocketW, MleftJacketPocketW, MrightFrontPocketW, MrightHandW, MrightJacketPocketW,
                           MleftBackPocketJ, MleftJacketPocketJ, MrightFrontPocketJ, MrightHandJ, MrightJacketPocketJ])

Matts_Dataset.to_csv('Data/Matts_Data.csv')

Warrens_Dataset = pd.concat([WleftBackPocketW, WleftJacketPocketW, WrightFrontPocketW, WrightHandW, WrightJacketPocketW,
                             WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ, WrightJacketPocketJ])

Warrens_Dataset.to_csv('Data/Warrens_Data.csv')

Ellens_Dataset = pd.concat([EleftBackPocketW, EleftJacketPocketW, ErightFrontPocketW, ErightHandW, ErightJacketPocketW,
                            EleftBackPocketJ, EleftJacketPocketJ, ErightFrontPocketJ, ErightHandJ, ErightJacketPocketJ])

Ellens_Dataset.to_csv('Data/Ellens_Data.csv')

combined_data = pd.concat([
    EleftBackPocketW,
    EleftJacketPocketW,
    ErightFrontPocketW,
    ErightHandW,
    ErightJacketPocketW,
    MleftBackPocketW,
    MleftJacketPocketW,
    MrightFrontPocketW,
    MrightHandW,
    MrightJacketPocketW,
    WleftBackPocketW,
    WleftJacketPocketW,
    WrightFrontPocketW,
    WrightHandW,
    WrightJacketPocketW,
    EleftBackPocketJ,
    EleftJacketPocketJ,
    ErightFrontPocketJ,
    ErightHandJ,
    ErightJacketPocketJ,
    MleftBackPocketJ,
    MleftJacketPocketJ,
    MrightFrontPocketJ,
    MrightHandJ,
    MrightJacketPocketJ,
    WleftBackPocketJ,
    WleftJacketPocketJ,
    WrightFrontPocketJ,
    WrightHandJ,
    WrightJacketPocketJ,
])

combined_data.to_csv('Data/Combined_Dataset.csv', index=False)

with h5py.File('data.h5', 'w') as hdf:
    # Combined Dataset Creation
    combined_DataSet = hdf.create_group('/mainDataset')
    combined_DataSet.create_dataset('mainDataset', data=combined_data)

    comb = pd.read_csv('Data/Combined_Dataset.csv')

    window_size = 500
    segments = [comb.iloc[i:i+window_size] for i in range(0, len(comb), window_size)]
    count_segments = int(np.ceil(len(comb) / window_size))

    # Shuffle group elements
    for i in range(count_segments):
        segments[i] = segments[i].sample(frac=1).reset_index(drop=True)

    train_data, test_data = train_test_split(comb, test_size=0.1)
    train_data.to_csv('Data/Training_Data.csv')
    test_data.to_csv('Data/Testing_Data.csv')

    # Training Dataset Creation
    training_Dataset = hdf.create_group('/mainDataset/Training')
    training_Dataset.create_dataset('training_dataset', data=train_data)

    # Testing Dataset Creation
    testing_Dataset = hdf.create_group('/mainDataset/Testing')
    testing_Dataset.create_dataset('testing_dataset', data=test_data)

    # Matt's Dataset Creation
    # Walking
    Matt_Group = hdf.create_group('/Matt_Group')
    Matt_Group.create_dataset('mbrpw', data=MleftBackPocketW)
    Matt_Group.create_dataset('mrhw', data=MrightHandW)
    Matt_Group.create_dataset('mrjw', data=MrightJacketPocketW)
    Matt_Group.create_dataset('mrfpw', data=MrightFrontPocketW)
    Matt_Group.create_dataset('mljw', data=MleftJacketPocketW)
    # Jumping
    Matt_Group.create_dataset('mbrpj', data=MleftBackPocketJ)
    Matt_Group.create_dataset('mrhj', data=MrightHandJ)
    Matt_Group.create_dataset('mrjj', data=MrightJacketPocketJ)
    Matt_Group.create_dataset('mrfpj', data=MrightFrontPocketJ)
    Matt_Group.create_dataset('mljj', data=MleftJacketPocketJ)

    # Warren's Dataset Creation
    Warren_Group = hdf.create_group('/Warren_Group')
    Warren_Group.create_dataset('wbrpw', data=WleftBackPocketW)
    Warren_Group.create_dataset('wrhw', data=WrightHandW)
    Warren_Group.create_dataset('wrjw', data=WrightJacketPocketW)
    Warren_Group.create_dataset('wrfpw', data=WrightFrontPocketW)
    Warren_Group.create_dataset('wljw', data=WleftJacketPocketW)
    # Jumping
    Warren_Group.create_dataset('wbrpj', data=WleftBackPocketJ)
    Warren_Group.create_dataset('wrhj', data=WrightHandJ)
    Warren_Group.create_dataset('wrjj', data=WrightJacketPocketJ)
    Warren_Group.create_dataset('wrfpj', data=WrightFrontPocketJ)
    Warren_Group.create_dataset('wljj', data=WleftJacketPocketJ)

    # Ellen's Dataset Creation
    Ellen_Group = hdf.create_group('/Ellen_Group')
    Ellen_Group.create_dataset('ebrpw', data=EleftBackPocketW)
    Ellen_Group.create_dataset('erhw', data=ErightHandW)
    Ellen_Group.create_dataset('erjw', data=ErightJacketPocketW)
    Ellen_Group.create_dataset('erfpw', data=ErightFrontPocketW)
    Ellen_Group.create_dataset('eljw', data=EleftJacketPocketW)
    # Jumping
    Ellen_Group.create_dataset('ebrpj', data=EleftBackPocketJ)
    Ellen_Group.create_dataset('erhj', data=ErightHandJ)
    Ellen_Group.create_dataset('erjj', data=ErightJacketPocketJ)
    Ellen_Group.create_dataset('erfpj', data=ErightFrontPocketJ)
    Ellen_Group.create_dataset('eljj', data=EleftJacketPocketJ)

 #
 # # Testing HDF5 Output
 #    with h5py.File('data.h5', 'r') as hdf:
 #        items = list(hdf.items())
 #        print(items)
 #        # Matt_Group = hdf.get('/Warren_Group')
 #        print(list(testing_Dataset.items()))
 #        d1 = combined_DataSet.get('testing_dataset')
 #        d1 = np.array(d1)
 #        print('\n')
 #        print(d1.shape)