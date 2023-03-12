# # Matt's Data
# MrightHandW = pd.read_csv('Walking_Data/MrightHandW.csv')
# MbackRightPocketW = pd.read_csv('Walking_Data/MleftBackPocketW.csv')
# MleftJacketW = pd.read_csv('Walking_Data/MleftJacketW.csv')
# MrightFrontPocketW = pd.read_csv('Walking_Data/MrightFrontPocketW.csv')
# MrightJacketW = pd.read_csv('Walking_Data/MrightJacketW.csv')
#
# MrightHandJ = pd.read_csv('Jumping_Data/MrightHandJ.csv')
# MbackRightPocketJ = pd.read_csv('Jumping_Data/MleftBackPocketJ.csv')
# MleftJacketJ = pd.read_csv('Jumping_Data/MleftJacketPocketJ.csv')
# MrightFrontPocketJ = pd.read_csv('Jumping_Data/MrightFrontPocketJ.csv')
# MrightJacketJ = pd.read_csv('Jumping_Data/MrightJacketPocketJ.csv')
#
# # Warren's Data
# WrightHandW = pd.read_csv('Walking_Data/WrightHandW.csv')
# WbackRightPocketW = pd.read_csv('Walking_Data/WleftBackPocketW.csv')
# WleftJacketW = pd.read_csv('Walking_Data/WleftJacketPocketW.csv')
# WrightFrontPocketW = pd.read_csv('Walking_Data/WrightFrontPocketW.csv')
# WrightJacketW = pd.read_csv('Walking_Data/WrightJacketPocketW.csv')
#
# WrightHandJ = pd.read_csv('Jumping_Data/WrightHandJ.csv')
# WbackRightPocketJ = pd.read_csv('Jumping_Data/WleftBackPocketJ.csv')
# WleftJacketJ = pd.read_csv('Jumping_Data/WleftJacketPocketJ.csv')
# WrightFrontPocketJ = pd.read_csv('Jumping_Data/WrightFrontPocketJ.csv')
# WrightJacketJ = pd.read_csv('Jumping_Data/WrightJacketPocketJ.csv')
#
# # Ellen's Data
# ErightHandW = pd.read_csv('Walking_Data/ErightHandW.csv')
# EbackRightPocketW = pd.read_csv('Walking_Data/EleftBackPocketW.csv')
# EleftJacketW = pd.read_csv('Walking_Data/EleftJacketW.csv')
# ErightFrontPocketW = pd.read_csv('Walking_Data/ErightFrontPocketW.csv')
# ErightJacketW = pd.read_csv('Walking_Data/ErightJacketW.csv')
#
# ErightHandJ = pd.read_csv('Jumping_Data/ErightHandJ.csv')
# EbackRightPocketJ = pd.read_csv('Jumping_Data/EbackRightPocketJ.csv')
# EleftJacketJ = pd.read_csv('Jumping_Data/EleftJacketPocketJ.csv')
# ErightFrontPocketJ = pd.read_csv('Jumping_Data/ErightFrontPocketJ.csv')
# ErightJacketJ = pd.read_csv('Jumping_Data/ErightJacketPocketJ.csv')
#
#
# with h5py.File('data.h5', 'w') as hdf:
#     # # Combined Dataset Creation
#     # combined_DataSet = hdf.create_group('/mainDataset')
#     # combined_DataSet.create_dataset('mainDataset', data=matrix_2)
#     #
#     # trainSet = hdf.create_group('/mainDataset/train')
#     # trainSet.create_dataset('mainDataset', data=matrix_2)
#     #
#     # testSet = hdf.create_group('/mainDataset/test')
#     # testSet.create_dataset('mainDataset', data=matrix_2)
#
#     # Matt's Dataset Creation
#     # Walking
#     Matt_Group = hdf.create_group('/Matt_Group')
#     Matt_Group.create_dataset('mbrpw', data=MbackRightPocketW)
#     Matt_Group.create_dataset('mrhw', data=MrightHandW)
#     Matt_Group.create_dataset('mrjw', data=MrightJacketW)
#     Matt_Group.create_dataset('mrfpw', data=MrightFrontPocketW)
#     Matt_Group.create_dataset('mljw', data=MleftJacketW)
#     # Jumping
#     Matt_Group.create_dataset('mbrpj', data=MbackRightPocketJ)
#     Matt_Group.create_dataset('mrhj', data=MrightHandJ)
#     Matt_Group.create_dataset('mrjj', data=MrightJacketJ)
#     Matt_Group.create_dataset('mrfpj', data=MrightFrontPocketJ)
#     Matt_Group.create_dataset('mljj', data=MleftJacketJ)
#
#     # Warren's Dataset Creation
#     Warren_Group = hdf.create_group('/Warren_Group')
#     Warren_Group.create_dataset('wbrpw', data=WbackRightPocketW)
#     Warren_Group.create_dataset('wrhw', data=WrightHandW)
#     Warren_Group.create_dataset('wrjw', data=WrightJacketW)
#     Warren_Group.create_dataset('wrfpw', data=WrightFrontPocketW)
#     Warren_Group.create_dataset('wljw', data=WleftJacketW)
#     # Jumping
#     Warren_Group.create_dataset('wbrpj', data=WbackRightPocketJ)
#     Warren_Group.create_dataset('wrhj', data=WrightHandJ)
#     Warren_Group.create_dataset('wrjj', data=WrightJacketJ)
#     Warren_Group.create_dataset('wrfpj', data=WrightFrontPocketJ)
#     Warren_Group.create_dataset('wljj', data=WleftJacketJ)
#
#     # Ellen's Dataset Creation
#     Ellen_Group = hdf.create_group('/Ellen_Group')
#     Ellen_Group.create_dataset('ebrpw', data=EbackRightPocketW)
#     Ellen_Group.create_dataset('erhw', data=ErightHandW)
#     Ellen_Group.create_dataset('erjw', data=ErightJacketW)
#     Ellen_Group.create_dataset('erfpw', data=ErightFrontPocketW)
#     Ellen_Group.create_dataset('eljw', data=EleftJacketW)
#     # Jumping
#     Ellen_Group.create_dataset('ebrpj', data=EbackRightPocketJ)
#     Ellen_Group.create_dataset('erhj', data=ErightHandJ)
#     Ellen_Group.create_dataset('erjj', data=ErightJacketJ)
#     Ellen_Group.create_dataset('erfpj', data=ErightFrontPocketJ)
#     Ellen_Group.create_dataset('eljj', data=EleftJacketJ)
#
#     # Testing HDF5 Output
#     with h5py.File('data.h5', 'r') as hdf:
#         items = list(hdf.items())
#         print(items)
#         Matt_Group = hdf.get('/Warren_Group')
#         print(list(Matt_Group.items()))
#         d1 = Matt_Group.get('wljj')
#         d1 = np.array(d1)
#         print('\n')
#         print(d1.sEllen)

