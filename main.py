from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import pandas as pd
from scipy.stats import skew, kurtosis
import statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
                             WleftBackPocketJ, WleftJacketPocketJ, WrightFrontPocketJ, WrightHandJ,
                             WrightJacketPocketJ])

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

# Visualization
time = combined_data.iloc[1000:1500, 0]
# Walking
# Left Back Pocket
EllenAVGW = EleftBackPocketW.iloc[1000:1500, 4]
WarrenAVGW = WleftBackPocketW.iloc[1000:1500, 4]
MattAVGW = MleftBackPocketW.iloc[1000:1500, 4]

EllenXW = EleftBackPocketW.iloc[1000:1500, 3]
WarrenXW = WleftBackPocketW.iloc[1000:1500, 3]
MattXW = MleftBackPocketW.iloc[1000:1500, 3]

# Jumping
EllenAVGJ = EleftBackPocketJ.iloc[1000:1500, 4]
WarrenAVGJ = WleftBackPocketJ.iloc[1000:1500, 4]
MattAVGJ = MleftBackPocketJ.iloc[1000:1500, 4]

EllenXJ = EleftBackPocketJ.iloc[1000:1500, 3]
WarrenXJ = WleftBackPocketJ.iloc[1000:1500, 3]
MattXJ = MleftBackPocketJ.iloc[1000:1500, 3]

fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
# using average acceleration, not very helpful
# ax.plot(time, EllenAVGW, label = 'Ellen Walking', color = 'red')
# ax.plot(time, WarrenAVGW, label = 'Warren Walking', color = 'green')
# ax.plot(time, MattAVGW, label = 'Matt Walking', color = 'blue')
#
# ax.plot(time, EllenAVGJ, label = 'Ellen Walking', color = 'red', linestyle = 'dashed')
# ax.plot(time, WarrenAVGJ,  label = 'Warren Walking', color = 'green' ,linestyle = 'dashed')
# ax.plot(time, MattAVGJ,  label = 'Matt Walking', color = 'blue', linestyle = 'dashed')

ax.plot(time, EllenXW, label='Ellen Walking', color='red')
ax.plot(time, WarrenXW, label='Warren Walking', color='green')
ax.plot(time, MattXW, label='Matt Walking', color='blue')

ax.plot(time, EllenXJ, label='Ellen Walking', color='red', linestyle='dashed')
ax.plot(time, WarrenXJ, label='Warren Walking', color='green', linestyle='dashed')
ax.plot(time, MattXJ, label='Matt Walking', color='blue', linestyle='dashed')
plt.show()

# Hand


# Jacket


# Jumping


with h5py.File('data.h5', 'w') as hdf:
    # Combined Dataset Creation
    combined_DataSet = hdf.create_group('/mainDataset')
    combined_DataSet.create_dataset('mainDataset', data=combined_data)

    comb = pd.read_csv('Data/Combined_Dataset.csv')

    window_size = 500
    segments = [comb.iloc[i:i + window_size] for i in range(0, len(comb), window_size)]
    count_segments = int(np.ceil(len(comb) / window_size))

    # Feature Extraction Part 1
    features = []
    feats = pd.DataFrame(columns=['Max Absolute Acceleration', 'Min Absolute Acceleration', 'Peak to Peak Range',
                                  'Mean Absolute Acceleration', 'Median Absolute Acceleration',
                                  'Variance', 'Skew', 'Kurtosis', 'Standard Deviation', 'Mode Absolute Acceleration'])

    # Shuffle Group Elements
    for i in range(count_segments):
        segments[i] = segments[i].sample(frac=1).reset_index(drop=True)

        # Feature Extraction Part 2
        features = [np.max(segments[i]['Absolute acceleration (m/s^2)']),
                    np.min(segments[i]['Absolute acceleration (m/s^2)']),
                    np.ptp(segments[i]['Absolute acceleration (m/s^2)']),
                    np.mean(segments[i]['Absolute acceleration (m/s^2)']),
                    np.median(segments[i]['Absolute acceleration (m/s^2)']),
                    np.var(segments[i]['Absolute acceleration (m/s^2)']),
                    skew(segments[i]['Absolute acceleration (m/s^2)']),
                    kurtosis(segments[i]['Absolute acceleration (m/s^2)']),
                    np.std(segments[i]['Absolute acceleration (m/s^2)']),
                    statistics.mode(segments[i]['Absolute acceleration (m/s^2)'])]
        feats = feats.append(pd.DataFrame([features], columns=feats.columns), ignore_index=True)
        feats.to_csv('tester.csv')
    print(feats)

    # Training and Testing File Creation
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
    # Walking
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
    # Walking
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


#  # Pre-processing step: (note: move imported libraries to top once finished*****)
#
# #  import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
#
#  # Load the dataset
# data = pd.read_csv('Data/Combined_Dataset.csv')
#
# # Apply moving average filter with window size 3
# data['smoothed'] = data['Acceleration x (m/s^2)'].rolling(window=3).mean()
#
# # Remove outliers using Z-score method
# data['z_score'] = np.abs((data['Acceleration x (m/s^2)'] - data['Acceleration x (m/s^2)'].mean()) / data['Acceleration x (m/s^2)'].std())
# data = data[data['z_score'] < 3]
#
# # Check class balance and remedy if necessary
# class_counts = data['Acceleration x (m/s^2)'].value_counts()
# minority_class = class_counts.idxmin()
# majority_class = class_counts.idxmax()
# class_ratio = class_counts[minority_class] / class_counts[majority_class]
# if class_ratio < 0.1:
#     # Upsample the minority class
#     minority_data = data[data['Acceleration x (m/s^2)'] == minority_class]
#     majority_data = data[data['Acceleration x (m/s^2)'] == majority_class]
#     minority_data_upsampled = minority_data.sample(n=len(majority_data), replace=True, random_state=42)
#     data = pd.concat([majority_data, minority_data_upsampled])
#     print(data)
#
# # Normalize the data using StandardScaler
# scaler = StandardScaler()
# data['normalized'] = scaler.fit_transform(data[['smoothed', 'feature1', 'feature2', 'feature3']])