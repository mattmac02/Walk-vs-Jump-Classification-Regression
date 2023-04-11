from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import pandas as pd
from scipy.stats import skew, kurtosis
import statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tkinter as tk
from tkinter.filedialog import askopenfilename


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
startTime = 3000
endTime = 3500
smallStart = 3000
smallEnd = 3300
# print(combined_data)
time = combined_data.iloc[startTime:endTime, 0]
timeSmall = combined_data.iloc[smallStart: smallEnd, 0]

# Left Back Pocket
# Walking
EllenZW = EleftBackPocketW.iloc[startTime:endTime, 3]
WarrenZW = WleftBackPocketW.iloc[startTime:endTime, 3]
MattZW = MleftBackPocketW.iloc[startTime:endTime, 3]

# Jumping

EllenZJ = EleftBackPocketJ.iloc[startTime:endTime, 3]
WarrenZJ = WleftBackPocketJ.iloc[startTime:endTime, 3]
MattZJ = MleftBackPocketJ.iloc[startTime:endTime, 3]

# plotting left back pocket Z axis
fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('Z-Axis Left Back Pocket Walking and Jumping (All)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Z Acceleration (m/s^2)', fontsize=15)
ax.scatter(time, EllenZW, label='Ellen Walking', color='red')
ax.scatter(time, WarrenZW, label='Warren Walking', color='green')
ax.scatter(time, MattZW, label='Matt Walking', color='blue')

ax.scatter(time, EllenZJ, label='Ellen Walking', color='red', linestyle='dashed')
ax.scatter(time, WarrenZJ, label='Warren Walking', color='green', linestyle='dashed')
ax.scatter(time, MattZJ, label='Matt Walking', color='blue', linestyle='dashed')
plt.legend(loc="upper left")


# plotting left back pocket xyz accel [Ellen]
fig1, ax1 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('XYZ-Axis Left Back Pocket Walking and Jumping (Ellen)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 1], label='X', color='red')
ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 2], label='Y', color='blue')
ax1.plot(time, EleftBackPocketW.iloc[startTime: endTime, 3], label='Z', color='green')

ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
ax1.plot(time, EleftBackPocketJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting hand xyz accel [Warren]
fig2, ax2 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('XYZ-Axis Hand Walking and Jumping (Warren)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
ax2.plot(time, WrightHandW.iloc[startTime: endTime, 1], label='X', color='red')
ax2.plot(time, WrightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
ax2.plot(time, WrightHandW.iloc[startTime: endTime, 3], label='Z', color='green')

ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
ax2.plot(time, WrightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting hand xyz accel [Ellen]
fig3, ax3 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('XYZ-Axis Hand Walking and Jumping (Ellen)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
ax3.plot(time, ErightHandW.iloc[startTime: endTime, 1], label='X', color='red')
ax3.plot(time, ErightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
ax3.plot(time, ErightHandW.iloc[startTime: endTime, 3], label='Z', color='green')

ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
ax3.plot(time, ErightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting hand xyz accel [Matthew]
fig4, ax4 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('XYZ-Axis Hand Walking and Jumping (Matthew)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('XYZ Acceleration (m/s^2)', fontsize=15)
ax4.plot(time, MrightHandW.iloc[startTime: endTime, 1], label='X', color='red')
ax4.plot(time, MrightHandW.iloc[startTime: endTime, 2], label='Y', color='blue')
ax4.plot(time, MrightHandW.iloc[startTime: endTime, 3], label='Z', color='green')

ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 1], label='Jump X', color='red', linestyle='dashed')
ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 2], label='Jump Y', color='blue', linestyle='dashed')
ax4.plot(time, MrightHandJ.iloc[startTime: endTime, 3], label='Jump Z', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting x-axis of right front pocket of everyone
fig5, ax5 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('X-Axis Right Front Walking and Jumping Pocket (All)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('X Acceleration (m/s^2)', fontsize=15)
ax5.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 1], label='M', color='red')
ax5.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 1], label='W', color='blue')
ax5.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 1], label='E', color='green')

ax5.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='M', color='red', linestyle='dashed')
ax5.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='W', color='blue', linestyle='dashed')
ax5.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 1], label='E', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting y-axis of right front pocket data of everyone
fig6, ax6 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('Y-Axis Right Front Walking and Jumping Pocket (All)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Y Acceleration (m/s^2)', fontsize=15)
ax6.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 2], label='M', color='red')
ax6.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 2], label='W', color='blue')
ax6.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 2], label='E', color='green')

ax6.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='M', color='red', linestyle='dashed')
ax6.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='W', color='blue', linestyle='dashed')
ax6.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 2], label='E', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting z-axis of right front pocket data
fig7, ax7 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('Z-Axis Right Front Walking and Jumping Pocket (All)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Z Acceleration (m/s^2)', fontsize=15)
ax7.plot(timeSmall, MrightFrontPocketW.iloc[smallStart: smallEnd, 3], label='M', color='red')
ax7.plot(timeSmall, WrightFrontPocketW.iloc[smallStart: smallEnd, 3], label='W', color='blue')
ax7.plot(timeSmall, ErightFrontPocketW.iloc[smallStart: smallEnd, 3], label='E', color='green')

ax7.plot(timeSmall, MrightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='M', color='red', linestyle='dashed')
ax7.plot(timeSmall, WrightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='W', color='blue', linestyle='dashed')
ax7.plot(timeSmall, ErightFrontPocketJ.iloc[smallStart: smallEnd, 3], label='E', color='green', linestyle='dashed')
plt.legend(loc="upper left")


# plotting walking x-axis standalone [Ellen]
fig8, ax8 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('X-Axis Left Jacket Pocket Walking and Jumping (Ellen)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('X Acceleration (m/s^2)', fontsize=15)
ax8.plot(time, EleftJacketPocketW.iloc[startTime: endTime, 1], label='walk', color='red')
ax8.plot(time, EleftJacketPocketJ.iloc[startTime: endTime, 1], label='jump', color='green')
plt.legend(loc="upper left")


# plotting y-axis standalone
fig9, ax9 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('Y-Axis Left Jacket Pocket Walking and Jumping (Ellen)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Y Acceleration (m/s^2)', fontsize=15)
ax9.plot(time, EleftJacketPocketW.iloc[startTime: endTime, 2], label='walk', color='red')
ax9.plot(time, EleftJacketPocketJ.iloc[startTime: endTime, 2], label='jump', color='green')
plt.legend(loc="upper left")


# plotting z-axis standalone
fig10, ax10 = plt.subplots(figsize=(10, 10), layout="constrained")
plt.title('Z-Axis Left Jacket Pocket Walking and Jumping (Ellen)')
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Z Acceleration (m/s^2)', fontsize=15)
ax10.plot(time, EleftJacketPocketW.iloc[startTime: endTime, 3], label='walk', color='red')
ax10.plot(time, EleftJacketPocketJ.iloc[startTime: endTime, 3], label='jump', color='green')
plt.legend(loc="upper left")

plt.show()

# GUI
# def open_file():
#     """Open a file for editing."""
#     filepath = askopenfilename(
#         filetypes=[("CSV Files", "*.csv")]
#     )
#     if not filepath:
#         return
#     file_display.delete("1.0", tk.END)
#     with open(filepath, mode="r", encoding="utf-8") as input_file:
#         text = input_file.read()
#         file_display.insert(tk.END, text)
#     window.title(f"Movement Tracking - {filepath}")
#
#
# window = tk.Tk()
# window.title("Movement Tracking")
#
# file_display = tk.Text(window)
# form_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
#
# btn_open = tk.Button(form_buttons, text="Open File", command=open_file)
# btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
#
# form_buttons.grid(row=0, column=0, sticky="ns")
# file_display.grid(row=0, column=1, sticky="nsew")
# window.mainloop()


with h5py.File('data.h5', 'w') as hdf:
    # Combined Dataset Creation
    combined_DataSet = hdf.create_group('/mainDataset')
    combined_DataSet.create_dataset('mainDataset', data=combined_data)
    window_size = 500

    comb = pd.read_csv('Data/Combined_Dataset.csv')

    segments = [comb.iloc[i:i + window_size] for i in range(0, len(comb), window_size)]
    count_segments = int(np.ceil(len(comb) / window_size))

    # Feature Extraction Part 1
    features = []
    feats = pd.DataFrame(columns=['Max Absolute Acceleration', 'Min Absolute Acceleration', 'Peak to Peak Range',
                                  'Mean Absolute Acceleration', 'Median Absolute Acceleration',
                                  'Variance', 'Skew', 'Kurtosis', 'Standard Deviation', 'Mode Absolute Acceleration'])

    # Shuffle Group Elements
    for i in range(count_segments):
        print(segments[i])
        dataframe = pd.DataFrame(segments[i])

        print("Preprocessing...")
        scaler = preprocessing.StandardScaler()
        df = pd.DataFrame(data=scaler.fit_transform(dataframe))
        print(df)

        fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
        plt.title('reprocessed Data 2')
        # plt.xlabel('Time (s)', fontsize=15)
        plt.ylabel('Absolute Acceleration', fontsize=15)
        ax.plot(df[0], df[3])
        plt.show()

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
        print(feats)

    # Data Shuffling
    segments[i] = segments[i].sample(frac=1).reset_index(drop=True)

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
