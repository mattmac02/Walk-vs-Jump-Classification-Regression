import numpy as np
import h5py
import pandas as pd
import os

# define the directories and file names for the data
base_dir = os.getcwd()
walking_dir = os.path.join(base_dir, 'Walking_Data')
jumping_dir = os.path.join(base_dir, 'Jumping_Data')

matt_files = {
    'right_hand_walking': 'MrightHandW.csv',
    'left_back_pocket_walking': 'MleftBackPocketW.csv',
    'left_jacket_walking': 'MleftJacketW.csv',
    'right_front_pocket_walking': 'MrightFrontPocketW.csv',
    'right_jacket_walking': 'MrightJacketW.csv',
    'right_hand_jumping': 'MrightHandJ.csv',
    'left_back_pocket_jumping': 'MleftBackPocketJ.csv',
    'left_jacket_jumping': 'MleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'MrightFrontPocketJ.csv',
    'right_jacket_jumping': 'MrightJacketPocketJ.csv',
}

warren_files = {
    'right_hand_walking': 'WrightHandW.csv',
    'left_back_pocket_walking': 'WleftBackPocketW.csv',
    'left_jacket_walking': 'WleftJacketPocketW.csv',
    'right_front_pocket_walking': 'WrightFrontPocketW.csv',
    'right_jacket_walking': 'WrightJacketPocketW.csv',
    'right_hand_jumping': 'WrightHandJ.csv',
    'left_back_pocket_jumping': 'WleftBackPocketJ.csv',
    'left_jacket_jumping': 'WleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'WrightFrontPocketJ.csv',
    'right_jacket_jumping': 'WrightJacketPocketJ.csv',
}

ellen_files = {
    'right_hand_walking': 'ErightHandW.csv',
    'left_back_pocket_walking': 'EleftBackPocketW.csv',
    'left_jacket_walking': 'EleftJacketPocketW.csv',
    'right_front_pocket_walking': 'ErightFrontPocketW.csv',
    'right_jacket_walking': 'ErightJacketPocketW.csv',
    'right_hand_jumping': 'ErightHandJ.csv',
    'left_back_pocket_jumping': 'EleftBackPocketJ.csv',
    'left_jacket_jumping': 'EleftJacketPocketJ.csv',
    'right_front_pocket_jumping': 'ErightFrontPocketJ.csv',
    'right_jacket_jumping': 'ErightJacketPocketJ.csv',
}

# read in the data files for each subject and store them in a dictionary
matt_data = {}
for key, file in matt_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    matt_data[key] = df

warren_data = {}
for key, file in warren_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    warren_data[key] = df

ellen_data = {}
for key, file in ellen_files.items():
    file_path = os.path.join(walking_dir if 'walking' in key else jumping_dir, file)
    df = pd.read_csv(file_path)
    ellen_data[key] = df

# combine the data into a single dataframe
matt_df = pd.concat(matt_data.values(), axis=1, keys=matt_data.keys())
warren_df = pd.concat(warren_data.values(), axis=1, keys=warren_data.keys())
ellen_df = pd.concat(ellen_data.values(), axis=1, keys=ellen_data.keys())
combined_df = pd.concat([matt_df, warren_df], axis=1, keys=['matt', 'warren'])


# store the combined data in an HDF5 file
with h5py.File('data.h5', 'w') as hdf:
    # create a group for the combined data
    Matt_Group = hdf.create_group('/Matt_Group')
    for key, value in matt_data.items():
        # Create a new dataset in the group for each key-value pair
        Matt_dataset = Matt_Group.create_dataset(key, data=value)

    Warren_Group = hdf.create_group('/Warren_Group')
    for key, value in warren_data.items():
        # Create a new dataset in the group for each key-value pair
        Warren_dataset = Warren_Group.create_dataset(key, data=value)

    Ellen_Group = hdf.create_group('/Ellen_Group')
    for key, value in ellen_data.items():
        # Create a new dataset in the group for each key-value pair
        Ellen_dataset = Ellen_Group.create_dataset(key, data=value)

    # Testing
    # group = hdf.create_group('combined_data')
    # # convert the pandas dataframe to a numpy array
    # data = combined_df.to_numpy()
    # # write the numpy array to the HDF5 file
    # group.create_dataset('data', data=data)
