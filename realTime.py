# Import required libraries
import csv
import os
import test as te
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Initialize web driver
driver = webdriver.Chrome()
driver.get("http://172.20.10.1")
time.sleep(1)

# Click the div with id "more"
more_div = driver.find_element(By.ID, "measuring")
ActionChains(driver).click(more_div).perform()
time.sleep(1)

# Click the "Simple" button
simple_button = driver.find_element(By.XPATH, "//li[text()='Simple']")
ActionChains(driver).click(simple_button).perform()
time.sleep(1)

end_time = time.time() + 5

soup = BeautifulSoup(driver.page_source, 'html.parser')
element_blocks = soup.find_all('div', {'class': 'elementBlock'})

# Set up the CSV writer
with open('output.csv', mode='w') as output_file:
    writer = csv.writer(output_file)
    headers = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    element_blocks = soup.find_all('div', {'class': 'elementBlock'})
    for block in element_blocks:
        label = block.find('span', {'class': 'label'}).text
        headers.append(label)
    writer.writerow(headers)

    while time.time() < end_time:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element_blocks = soup.find_all('div', {'class': 'elementBlock'})
        row = []
        for block in element_blocks:
            value_number = block.find('span', {'class': 'valueNumber'})
            row.append(value_number.text)
        writer.writerow(row)
        time.sleep(0.1)

    more_div = driver.find_element(By.ID, "measuring")
    ActionChains(driver).click(more_div).perform()
    time.sleep(1)

# Close web driver
driver.close()

testing = pd.read_csv('output.csv')

# analyzedata(testing)
te.analyzedata(testing)

#
# # def analyzedata(testing):
#
# dataIn = testing
# # print(dataIn)
#
# if os.path.exists("Data/Features_Combined.csv"):
#     os.remove("Data/Features_Combined.csv")
#     print("File deleted")
# if os.path.exists("Data/Input_Features.csv"):
#     os.remove("Data/Input_Features.csv")
#     print("Input File Deleted")
#
#
#
# with h5py.File('data.h5', 'w') as hdf:
#     # Combined Dataset Creation
#     combined_DataSet = hdf.create_group('/mainDataset')
#     # combined_DataSet.create_dataset('mainDataset', data=Full_set)
#
#     print(dataIn)
#
#     print('here 1')
#     # Cleaning (Pre Processing)
#     for column in dataIn.columns:
#         dataIn[column] = dataIn[column].rolling(window=5).mean()
#
#     print(dataIn)
#
#     # for column in dataIn.columns:
#     #     #other
#     #     dataIn[column] =  dataIn[column].rolling(window=60).mean()
#
#
#         # Drop rows with missing values
#     dataIn.dropna(inplace=True)
#     dataIn.dropna(inplace=True)
#
#     # fig11, ax12 = plt.subplots(figsize=(10, 10), layout="constrained")
#     # plt.title('Full Set Data')
#     # plt.xlabel('Time (s)', fontsize=15)
#     # plt.ylabel('Acceleration (m/s^2)', fontsize=15)
#     # ax12.scatter(Full_set['Time (s)'], Full_set['Absolute acceleration (m/s^2)'], label='Full Set', color='red')
#     # plt.legend(loc="upper left")
#     #plt.show()
#
#     # Save preprocessed data
#     dataIn.to_csv('Data/Combined_Dataset_preprocessed.csv', index=False)
#
#
#     window_size = 500
#
#     print('here 2')
#
#     # Segmenting Data
#     segments = [dataIn.iloc[i:i + window_size] for i in range(0, len(dataIn), window_size)]
#     count_segments = int(np.ceil(len(dataIn) / window_size))
#
#     # input data
#
#     segmentsIn = [dataIn.iloc[i:i + window_size] for i in range(0, len(dataIn), window_size)]
#     count_segmentsIn = int(np.ceil(len(dataIn) / window_size))
#     print(count_segmentsIn)
#
#     for i in range(count_segmentsIn):
#         print('here 3')
#
#         dataframe = pd.DataFrame(segmentsIn[i])
#
#         # Normalizing (Pre Processing)
#         scaler = preprocessing.StandardScaler()
#         dfIn = pd.DataFrame(data=scaler.fit_transform(dataframe.iloc[:,:]))
#
#
#         w_size = 5
#
#         feature_namesIn = ['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance', 'Skew', 'Kurtosis',
#                          'Standard Deviation', 'Inter Quartile Range']
#
#         featsIn = pd.DataFrame(columns=feature_namesIn)
#
#         # Feature Extraction Part 2
#         featuresIn = [dfIn.max().values,
#                     dfIn.min().values,
#                     dfIn.max().values - dfIn.min().values,
#                     dfIn.mean().values,
#                     dfIn.median().values,
#                     dfIn.var().values,
#                     dfIn.skew().values,
#                     dfIn.kurt().values,
#                     dfIn.std(axis=1).values,
#                     dfIn.quantile(0.75).values - dfIn.quantile(0.25).values]
#         featsIn.loc[i] = [featuresIn[j][4] for j in range(len(featuresIn))]
#
#         with open('Data/Features_Combined.csv', 'a') as f:
#             featsIn.to_csv(f, header=f.tell() == 0, index=False)
#
#         featuresLab = pd.read_csv('Data/Features_Combined.csv')
#         featuresLab['label'] = 0
#         # Data Shuffling
#         segments[i] = segments[i].sample(frac=1).reset_index(drop=True)
#
#         # Training and Testing File Creation
#         # train_data, test_data = train_test_split(comb, test_size=0.1)
#         # train_data.to_csv('Data/Training_Data.csv')
#         # test_data.to_csv('Data/Testing_Data.csv')
#
#         # Training Dataset Creation
#         # training_Dataset = hdf.create_group('/mainDataset/Training')
#         # training_Dataset.create_dataset('training_dataset', data=train_data)
#
#         # Testing Dataset Creation
#         # testing_Dataset = hdf.create_group('/mainDataset/Testing')
#         # testing_Dataset.create_dataset('testing_dataset', data=test_data)
#
#         # Matt's Dataset Creation
#         # Walking
#         # Matt_Group = hdf.create_group('/Matt_Group')
#         # Matt_Group.create_dataset('mbrpw', data=MleftBackPocketW)
#         # Matt_Group.create_dataset('mrhw', data=MrightHandW)
#         # Matt_Group.create_dataset('mrjw', data=MrightJacketPocketW)
#         # Matt_Group.create_dataset('mrfpw', data=MrightFrontPocketW)
#         # Matt_Group.create_dataset('mljw', data=MleftJacketPocketW)
#         # # Jumping
#         # Matt_Group.create_dataset('mbrpj', data=MleftBackPocketJ)
#         # Matt_Group.create_dataset('mrhj', data=MrightHandJ)
#         # Matt_Group.create_dataset('mrjj', data=MrightJacketPocketJ)
#         # Matt_Group.create_dataset('mrfpj', data=MrightFrontPocketJ)
#         # Matt_Group.create_dataset('mljj', data=MleftJacketPocketJ)
#
#         # Warren's Dataset Creation
#         # Walking
#         # Warren_Group = hdf.create_group('/Warren_Group')
#         # Warren_Group.create_dataset('wbrpw', data=WleftBackPocketW)
#         # Warren_Group.create_dataset('wrhw', data=WrightHandW)
#         # Warren_Group.create_dataset('wrjw', data=WrightJacketPocketW)
#         # Warren_Group.create_dataset('wrfpw', data=WrightFrontPocketW)
#         # Warren_Group.create_dataset('wljw', data=WleftJacketPocketW)
#         # # Jumping
#         # Warren_Group.create_dataset('wbrpj', data=WleftBackPocketJ)
#         # Warren_Group.create_dataset('wrhj', data=WrightHandJ)
#         # Warren_Group.create_dataset('wrjj', data=WrightJacketPocketJ)
#         # Warren_Group.create_dataset('wrfpj', data=WrightFrontPocketJ)
#         # Warren_Group.create_dataset('wljj', data=WleftJacketPocketJ)
#
#         # Ellen's Dataset Creation
#         # Walking
#         # Ellen_Group = hdf.create_group('/Ellen_Group')
#         # Ellen_Group.create_dataset('ebrpw', data=EleftBackPocketW)
#         # Ellen_Group.create_dataset('erhw', data=ErightHandW)
#         # Ellen_Group.create_dataset('erjw', data=ErightJacketPocketW)
#         # Ellen_Group.create_dataset('erfpw', data=ErightFrontPocketW)
#         # Ellen_Group.create_dataset('eljw', data=EleftJacketPocketW)
#         # # Jumping
#         # Ellen_Group.create_dataset('ebrpj', data=EleftBackPocketJ)
#         # Ellen_Group.create_dataset('erhj', data=ErightHandJ)
#         # Ellen_Group.create_dataset('erjj', data=ErightJacketPocketJ)
#         # Ellen_Group.create_dataset('erfpj', data=ErightFrontPocketJ)
#         # Ellen_Group.create_dataset('eljj', data=EleftJacketPocketJ)
#
#
#         #---------------------------------------------------------------------------------------------------
#
#         #creating a classifier
#
#         print('HERE 2')
#
#         dfw = pd.read_csv('Data/Features_Combined.csv')
#         # dfw2 = pd.read_csv('Data/Input_Features.csv')
#         # print(dfw)
#
#         # def classifier(df):
#
#         #add label matrix for the y train  and test in line 471
#         df2 = dfw.dropna()
#
#         inData = dfw.dropna()
#
#         X_train, X_test, Y_train, Y_test = train_test_split(df2[['Max', 'Min', 'Peak to Peak Range', 'Mean', 'Median', 'Variance',
#                    'Skew', 'Kurtosis', 'Standard Deviation', 'Inter Quartile Range']], df2['label'].values[::window_size][:],
#                                                                 test_size=0.1, shuffle=True)
#         #other
#         text_input = pd.DataFrame(data=scaler.fit_transform(inData.iloc[:window_size,:]))
#
#
#
#             # print(f'X Train: \n{X_train}')
#         #print(f'X Test: \n{X_test}')
#             # print(f'Y Train: \n{Y_train}')
#             # print(f'Y Test: \n{Y_test}')
#
#         scaler = StandardScaler()
#         l_reg = LogisticRegression(max_iter=10000)
#         clf = make_pipeline(StandardScaler(), l_reg)
#         clf.fit(X_train, Y_train)
#         Y_pred = clf.predict(X_test)
#         y_clf_prob = clf.predict_proba(X_test)
#
#         normalization = StandardScaler()
#         l_reg = LogisticRegression(max_iter=10000)
#         pca = PCA(n_components=2)
#
#         pca_pipe = make_pipeline(normalization, pca)
#
#         X_train_pca = pca_pipe.fit_transform(X_train)
#         X_test_pca = pca_pipe.fit_transform(X_test)
#
#         clf = make_pipeline(l_reg)
#
#         clf.fit(X_train_pca, Y_train)
#
#         y_pred_pca = clf.predict(X_test_pca)
#
#         disp = DecisionBoundaryDisplay.from_estimator(clf, X_train_pca, response_method = "predict", xlabel='X1', ylabel='X2', alpha=0.5)
#         disp.ax_.scatter(X_train_pca[:,0], X_train_pca[:,1], c=Y_train)
#
#         acc = accuracy_score(Y_test, y_pred_pca)
#         print('accuracy is', acc)
#         # plt.show()
#
#         #other
#         pca_pipeIn = make_pipeline(normalization, pca)
#         test_pca = pca_pipeIn.fit_transform(text_input)
#         print(text_input)
#         text_predict_pca = clf.predict(test_pca)
#
#         print(text_predict_pca)
#         text_input["Action"] = text_predict_pca
#         text_input.to_csv('Data/input.csv', index=False)
#
#         fig14, ax14 =plt.subplots(figsize=(10,10), layout = "constrained")
#         ax14.plot(text_input.iloc[:, 0], text_input.iloc[:,3])
#
