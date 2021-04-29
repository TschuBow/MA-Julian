from json.decoder import JSONDecoder
import pandas as pd
from pandas import DataFrame
import streamlit as st
import json
import librosa
import librosa.display
import numpy as np
from scipy import signal
from scipy.io import wavfile
import own_functions
from pathlib import Path
import codecs, json
import seaborn as sns; sns.set_theme()
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import own_functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import math
import matplotlib.pyplot as plt


def main():
    
    st.header("Feature Regressor")
    st.write("Train and test your ML algorithm.")


    classifiers = ("Linear Regression", "Polynomial Regression Analysis", "KNN", "SVM", "Random Forrest")
    

    col1, col2, col3 = st.beta_columns((4,1,1))

    test_series_data_file = col1.file_uploader(
        "upload the test series features via the correct json file.",
        type=["json"],
        accept_multiple_files=False,
    )
    
    if test_series_data_file:
        open_test_series_data_file = json.load(test_series_data_file)
        audio_files_in_test_series = open_test_series_data_file.keys()
        


        audio_files_in_test_series_list = list(audio_files_in_test_series)
        audio_files_in_test_series_list = sorted(audio_files_in_test_series_list)

        intro_info_of_file = open_test_series_data_file[f"{audio_files_in_test_series_list[0]}"].keys()
        intro_info_of_file_list = list(intro_info_of_file)

        measuring_paths_of_each_file = open_test_series_data_file[f"{audio_files_in_test_series_list[0]}"]["features_in_measuring_paths"].keys()
        measuring_paths_of_each_file_list = list(measuring_paths_of_each_file)

        tsx_in_measuring_paths = open_test_series_data_file[f"{audio_files_in_test_series_list[0]}"]["features_in_measuring_paths"][f"{measuring_paths_of_each_file_list[0]}"].keys()
        tsx_in_measuring_paths_list = list(tsx_in_measuring_paths)

        features_in_measuring_paths = open_test_series_data_file[f"{audio_files_in_test_series_list[0]}"]["features_in_measuring_paths"]["1"]["tsl"].keys()
        features_in_measuring_paths_list = list(features_in_measuring_paths)
        
        width_features_in_measuring_paths = open_test_series_data_file[f"{audio_files_in_test_series_list[0]}"]["features_in_measuring_paths"]["1"]["tsl"]["width_features"].keys()
        width_features_in_measuring_paths_list = list(width_features_in_measuring_paths)

        # st.write(audio_files_in_test_series_list)
        # st.write(intro_info_of_file_list)
        # st.write(measuring_paths_of_each_file_list)
        # st.write(tsx_in_measuring_paths_list)
        # st.write(features_in_measuring_paths_list)
        # st.write(width_features_in_measuring_paths_list)

        number_of_features = col1.number_input("number of features you wanna use", min_value = 1, value = 3)

        col1.write(f"The test series contains data sets from {len(audio_files_in_test_series_list)} different objects")

        target_value_list = []
        

        # chosen_path = st.selectbox("path", measuring_paths_of_each_file_list)
        # chosen_tsx = st.radio("tsx", ("tsl", "tsr"))
        # chosen_width_feature = st.selectbox("width features", width_features_in_measuring_paths_list)
    
        target_value_list = []

        for audio_file_in_test_series_list in audio_files_in_test_series_list:
            optically_measured_deviation = open_test_series_data_file[f"{audio_file_in_test_series_list}"]["optically_measured_deviation"]
            target_value_list.append(optically_measured_deviation)
        
        target_value_matrix = np.array(target_value_list)
        # st.write(target_value_list)

        chosen_value_matrix = np.zeros((len(audio_files_in_test_series_list), number_of_features))
        # [[0, 0, 0], <- alle features für eine WSP
        #  [0, 0, 0]]
        #
        # chosen_path_list = np.zeros(number_of_features, dtype=str)
        # chosen_tsx_list = np.zeros(number_of_features, dtype=str)
        # chosen_feature_list = np.zeros(number_of_features, dtype=str)
    
        # col_title, col_path, col_tsx, col_feature = st.beta_columns(3)
        chosen_feature_combi_list = []
        for j in range(number_of_features):  # iterate over features

            col_title, col_path, col_tsx, col_feature, col_empty = st.beta_columns((2, 4, 4, 10, 10))
            col_title.write("")
            col_title.write("")
            col_title.write(f"feature {j+1}")
            chosen_path = col_path.selectbox("path", measuring_paths_of_each_file_list, key=f"chosen_path_list{j}")
            chosen_tsx = col_tsx.selectbox("tsx", tsx_in_measuring_paths_list, key=f"chosen_tsx_list{j}")
            chosen_feature = col_feature.selectbox("feature", width_features_in_measuring_paths_list, key=f"chosen_feature_list{j}")
            
            chosen_feature_combi_list.append(f"Path {chosen_path} {chosen_tsx} - {chosen_feature}")

            for i in range(len(audio_files_in_test_series_list)):  # iterate over WSP
                # print(f"{audio_files_in_test_series_list[i]}")
                # print(f"{chosen_path_list[j]}")
                # print(f"{chosen_tsx_list[j]}")
                # print(f"{chosen_feature_list[j]}")

                chosen_value_matrix[i][j] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{chosen_path}"][f"{chosen_tsx}"]["width_features"][f"{chosen_feature}"]
                
                
                
                #chosen_value_matrix[i][j] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{chosen_path_list[j]}"][f"{chosen_tsx_list[j]}"]["width_features"][f"{chosen_feature_list[j]}"]
                #feature value of feature j for WSP i
        
        col1, col2, col3 = st.beta_columns((6,2,4))
        
        # col1.write("feature values")
        # col1.write(chosen_value_matrix) 
        # col2.write("target values")
        # col2.write(target_value_matrix)

        # panda_target_value_matrix = pd.DataFrame(data = target_value_list, index = audio_files_in_test_series_list, columns = ["target value"])
        # st.write(panda_target_value_matrix)
        
        col1, col2, col3 = st.beta_columns((4,1,1))
        panda_chosen_value_matrix = pd.DataFrame(data = chosen_value_matrix, index = audio_files_in_test_series_list, columns = chosen_feature_combi_list)
        panda_chosen_value_matrix['target value'] = target_value_list
        
        expander = col1.beta_expander(
                f"Dataframe of the Selected Features"
            )
        with expander:
            st.dataframe(data=panda_chosen_value_matrix)

        expander = col1.beta_expander(
                f"Correlation Heatmap of the Selected Features"
            )
        col1, col2, col3 = st.beta_columns((4,1,1))
        with expander:       
            corr = panda_chosen_value_matrix.corr()
            f=plt.figure(figsize=(8,4))
            sns.set(font_scale=0.6)
            ax = sns.heatmap(corr, 
                vmin=-1, 
                vmax=1, 
                center=0, 
                linewidths= 0.5, 
                fmt="0.2g",
                annot=True, 
                square=True,
                annot_kws={"size": 4},
                

                cmap=sns.diverging_palette(20, 250, n=200, center="light"))
            
            ax.set_xticklabels(ax.get_xticklabels(), FontSize = 8, rotation=45, horizontalalignment='right', )
            ax.set_yticklabels(ax.get_yticklabels(), FontSize= 8)
            ax.set
            st.pyplot(f)
        

        

        classifiers = ("Linear Regression", "Polynomial Regression", "KNN", "SVM", "Random Forest")
        classifier_name = st.selectbox("Select Classifier", classifiers)
        
        params = own_functions.add_parameter_ui(classifier_name)
        clf = own_functions.get_classifier(classifier_name, params)
        
        selected_test_size = st.slider("ratio of testing data", 0.05, 0.95, 0.20, 0.05)
        x_train, x_test, y_train, y_test = train_test_split(chosen_value_matrix, target_value_list, test_size=selected_test_size, random_state = 1234) 



        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)


        if classifier_name == "Linear Regression" or "Polynomial Regression":
            calc_explained_variance = explained_variance_score(y_test, y_pred)
            calc_max_error = max_error(y_test, y_pred)
            calc_mean_absolute_error = mean_absolute_error(y_test, y_pred)
            calc_mean_squared_error = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(calc_mean_squared_error)

            st.write(f"regression = {classifier_name}")
            st.write(f"variance = {calc_explained_variance}")
            st.write(f"max error = {calc_max_error}")
            st.write(f"mean absolute error = {calc_mean_absolute_error}")
            st.write(f"mean squared error = {calc_mean_squared_error}")
            st.write(f"RMSE = {rmse}")


        elif classifier_name == "KNN" or "SVM" or "Random Forest":
            acc = accuracy_score(y_test, y_pred)
            st.write(f"classifier = {classifier_name}")
            st.write(f"accuracy = {acc}")


        # train_regression = st.button('train regression')

        # if train_regression == True:

        #     reg = LinearRegression()
        #     reg.fit(chosen_value_matrix, target_value_list)        #x kommt rein; y kommt raus

        else:
            None

    else:
        None