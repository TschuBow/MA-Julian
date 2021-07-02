from json.decoder import JSONDecoder
from threading import stack_size
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
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import own_functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
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
    

    tsx_same_radio = col1.radio("Do the paths represent same geometric values?", ["yes", "no"])

             #Falls alle Pfade ungleiche Größen representieren
        
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

        if tsx_same_radio == "yes": #Falls alle Pfade die gleiche Größe representieren

            # number_of_features = col1.number_input("Set the number of features you want to chose manually for the regression.", min_value = 1, value = 3)
            data_amount=len(audio_files_in_test_series_list)*len(measuring_paths_of_each_file_list)

            col1.write(f"The test series contains data sets from {len(audio_files_in_test_series_list)} different objects. The measuring paths of each objects represent the same part of the object. This results in {data_amount} data.")
            
            target_value_list = []
            data_name_list = []

            
            for audio_file_in_test_series_list in audio_files_in_test_series_list:
                optically_measured_deviation = open_test_series_data_file[f"{audio_file_in_test_series_list}"]["optically_measured_deviation"]
                for j in range(len(measuring_paths_of_each_file_list)):
                    target_value_list.append(optically_measured_deviation)
                    data_name_list.append(f"{audio_file_in_test_series_list} - Path {j+1}")
                        
            total_features_list_len = 2 * (len(width_features_in_measuring_paths_list)) #2 x für tsx
            total_value_matrix = np.zeros((data_amount, total_features_list_len))
            total_columns_list = []
           
            for a in range(2):
                if a == 0:
                    tsx = "tsl"
                else:
                    tsx = "tsr"
                wf = 0
                for width_feature_in_measuring_paths in width_features_in_measuring_paths_list:
                    col = a*len(width_features_in_measuring_paths_list) + wf
                    total_columns_list.append(f"{tsx} - {width_feature_in_measuring_paths}")
                    

            for i in range(len(audio_files_in_test_series_list)):              # iteriert für jedes Audio File
               
                for h in range(len(measuring_paths_of_each_file_list)):             # iteriert jeden Pfad
                    
                    line = i*len(measuring_paths_of_each_file_list) + h

                    for a in range(2):
                        if a == 0:
                            tsx = "tsl"
                        else:
                            tsx = "tsr"   
                        wf = 0 
                        for width_feature_in_measuring_paths in width_features_in_measuring_paths_list:

                            col = a*len(width_features_in_measuring_paths_list) + wf
                        
                            total_value_matrix[line][col] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{h+1}"][f"{tsx}"]["width_features"][f"{width_feature_in_measuring_paths}"]
                            
                            wf = wf + 1
            
            panda_total_value_matrix = pd.DataFrame(data = total_value_matrix, index = data_name_list, columns = total_columns_list)
            panda_total_value_matrix['target value'] = target_value_list

            expander = col1.beta_expander(
                    f"Data frame of total Features"
                )
            with expander:
                st.dataframe(data=panda_total_value_matrix)
                st.write(f"lines: {len(panda_total_value_matrix.index)}")
                st.write(f"columns: {len(panda_total_value_matrix.columns)}, including one column for target values")
            number_of_features = col1.number_input("Set the number of features you want to chose manually for the regression.", min_value = 1, value = 3)

            chosen_value_matrix = np.zeros((data_amount, number_of_features))
            
            chosen_feature_combi_list = []
            for j in range(number_of_features):  # iterate over features
                
                col_title, col_path, col_tsx, col_feature, col_empty = st.beta_columns((2, 4, 4, 10, 10))
                col_title.write("")
                col_title.write("")
                col_title.write(f"feature {j+1}")
            
                chosen_tsx = col_tsx.selectbox("tsx", tsx_in_measuring_paths_list, key=f"chosen_tsx_list{j}")
                chosen_feature = col_feature.selectbox("feature", width_features_in_measuring_paths_list, key=f"chosen_feature_list{j}")
                
                chosen_feature_combi_list.append(f"{chosen_tsx} - {chosen_feature}")

                for i in range(len(audio_files_in_test_series_list)):  # iterate over WSP
                    for h in range(len(measuring_paths_of_each_file_list)):
                        line = i*len(measuring_paths_of_each_file_list) + h
                        chosen_value_matrix[line][j] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{h+1}"][f"{chosen_tsx}"]["width_features"][f"{chosen_feature}"]
                    
                    
                   
            
            
            
                        
            col1, col2, col3 = st.beta_columns((4,1,1))
            panda_chosen_value_matrix = pd.DataFrame(data = chosen_value_matrix, index = data_name_list, columns = chosen_feature_combi_list)
            panda_chosen_value_matrix['target value'] = target_value_list
            
            expander = col1.beta_expander(
                    f"Data frame of the Selected Features"
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
           
           
       
        elif tsx_same_radio == "no": 

            # number_of_features = col1.number_input("Set the number of features you want to chose manually for the regression.", min_value = 1, value = 3)
            data_amount=len(audio_files_in_test_series_list)
            
            target_value_list = np.zeros(data_amount)
            
            data_name_list = audio_files_in_test_series_list   
    
            total_features_list_len = len(measuring_paths_of_each_file_list) * 2 * (len(width_features_in_measuring_paths_list)) #2 x für tsx
            
            total_value_matrix = np.zeros((data_amount, total_features_list_len))
            total_columns_list = []
                
            for j in range(len(measuring_paths_of_each_file_list)):
                
                for a in range(2):
                    if a == 0:
                        tsx = "tsl"
                    else:
                        tsx = "tsr"
                    
                    for width_feature_in_measuring_paths in width_features_in_measuring_paths_list:
                        
                        total_columns_list.append(f"Path {j+1} - {tsx} - {width_feature_in_measuring_paths}")
                                

            for i in range(len(audio_files_in_test_series_list)):              # iteriert für jedes Audio File
               
                for j in range(len(measuring_paths_of_each_file_list)):             # iteriert jeden Pfad
                    
                    line = i
                    target_value_list[line] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["optically_measured_deviation"]
                    for a in range(2):
                        if a == 0:
                            tsx = "tsl"
                        else:
                            tsx = "tsr"   
                        wf = 0 
                        for width_feature_in_measuring_paths in width_features_in_measuring_paths_list:

                            col = j * (2 * len(width_features_in_measuring_paths_list)) + a *len(width_features_in_measuring_paths_list) + wf
                        
                            total_value_matrix[line][col] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{j+1}"][f"{tsx}"]["width_features"][f"{width_feature_in_measuring_paths}"]
                            
                            wf = wf + 1
                    
            panda_total_value_matrix = pd.DataFrame(data = total_value_matrix, index = data_name_list, columns = total_columns_list)
            panda_total_value_matrix['target value'] = target_value_list
            
            expander = col1.beta_expander(
                    f"Data frame of total Features"
                )
            with expander:
                st.dataframe(data=panda_total_value_matrix)
                st.write(f"lines: {len(panda_total_value_matrix.index)}")
                st.write(f"columns: {len(panda_total_value_matrix.columns)}, including one column for target values")                            
            col1.write(f"The test series contains data sets from {len(audio_files_in_test_series_list)} different objects")
            
            number_of_features = col1.number_input("Set the number of features you want to chose manually for the regression.", min_value = 1, value = 3)


            chosen_value_matrix = np.zeros((len(audio_files_in_test_series_list), number_of_features))
            
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
                  
                    chosen_value_matrix[i][j] = open_test_series_data_file[f"{audio_files_in_test_series_list[i]}"]["features_in_measuring_paths"][f"{chosen_path}"][f"{chosen_tsx}"]["width_features"][f"{chosen_feature}"]
                    
            
            col1, col2, col3 = st.beta_columns((4,1,1))
            panda_chosen_value_matrix = pd.DataFrame(data = chosen_value_matrix, index = audio_files_in_test_series_list, columns = chosen_feature_combi_list)
            panda_chosen_value_matrix['target value'] = target_value_list
            
            

            expander = col1.beta_expander(
                    f"Data frame of the Selected Features"
                )
            col1, col2, col3 = st.beta_columns((4,1,1))
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
            
        expander = col1.beta_expander(
                "Scatter plot of data"
            )
        with expander:
            feature1 = st.selectbox("feature 1", panda_total_value_matrix.columns)
            feature2 = st.selectbox("feature 2", panda_total_value_matrix.columns)
            degreeset = st.number_input("Degree of regression", 1, 6, 1)
            x = panda_total_value_matrix[f'{feature1}']
            y = panda_total_value_matrix[f'{feature2}']
            # xlen = len(x)
            # ylen = len(y)
            # col1.write(f"{x}")
            # col1.write(f"{y}")
            # print(x)
            # print(y)
            x=x.to_numpy()
            y=y.to_numpy()
            x_size=x.size
            y_size=y.size

            x=x.reshape(x.size,1)
            y=y.reshape(y.size,1)
            # col1.write(f"{x}")
            # col1.write(f"{y}")
            #np.reshape(y,(-1,1))
            regr = LinearRegression()
            regr.fit(x,y)
            ypred = regr.predict(x)
            #Ich glaub, ich habs :D <3

            poly = make_pipeline(PolynomialFeatures(degree= degreeset), Ridge())
            poly.fit(x,y)
            ypred_poly = poly.predict(x)

            reg_graph=plt.figure(figsize=(15,10))
            reg=plt.scatter(x,y, color='red')
            plt.plot(x,ypred, label="linear", color='blue', linewidth = 2)
            plt.plot(x,ypred_poly, label=f"polynomial_degree_{degreeset}", color='orange', linewidth = 2)
            plt.plot()
            plt.xlabel(f"{feature1}", fontsize=16)
            plt.ylabel(f"{feature2}", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=12)
            st.write(reg_graph)




        classifiers = ("Linear Regression", "Polynomial Regression", "KNN") #"SVM", "Random Forest")
        col1.write("")
        col1.header("Regression of the manually selected features")
        classifier_name = col1.selectbox("Select Regressor", classifiers)
        
        params = own_functions.add_parameter_ui(classifier_name)
        clf = own_functions.get_classifier(classifier_name, params)
        
        selected_test_size = col1.slider("ratio of testing data", 0.05, 0.95, 0.20, 0.05)
        x_train, x_test, y_train, y_test = train_test_split(chosen_value_matrix, target_value_list, test_size=selected_test_size, random_state = 1234) 

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        col1, col2, col3 = st.beta_columns((4,1,1))

        
        if classifier_name == "Linear Regression" or classifier_name == "Polynomial Regression":
            calc_explained_variance = explained_variance_score(y_test, y_pred)
            calc_max_error = max_error(y_test, y_pred)
            calc_mean_absolute_error = mean_absolute_error(y_test, y_pred)
            calc_mean_squared_error = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(calc_mean_squared_error)
            coefficient_of_determination = r2_score(y_test, y_pred)

            col1.write(f"regression = {classifier_name}")
            col1.write(f"variance = {round(calc_explained_variance, 2)}")
            col1.write(f"max error = {round(calc_max_error, 2)}")
            col1.write(f"mean absolute error = {round(calc_mean_absolute_error, 2)}")
            col1.write(f"mean squared error = {round(calc_mean_squared_error, 2)}")
            col1.write(f"RMSE = {round(rmse, 2)}")
            col1.write(f"R² = {round(coefficient_of_determination, 2)}")

            pred_graph=plt.figure(figsize=(15,10))
            plt.axline((0, 0), (1, 1), linewidth=4, color='r')
            plt.scatter(y_test, y_pred)
            plt.xlabel("target value", fontsize=20)
            plt.ylabel("predicted value", fontsize=20)
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)

            expander = col1.beta_expander(
                f"Target and predicted values"
            )      
            expander.pyplot(pred_graph)
            pred_test_matrix = pd.DataFrame()
            pred_test_matrix['test data values'] = y_test
            pred_test_matrix['predicted values'] = y_pred
            expander.write(pred_test_matrix)

            st.write("")
            if classifier_name == "Linear Regression":
                st.header("Automatical feature selection by Recursive Feature Elimination")
                rfe_activate=st.button("Perform RFE")
            #initial_features = st.number_input("Select number of best features you want to determine", 1, (total_features_list_len - 1), 5, 1)
                if  rfe_activate==True:
                    col1, col2, col3 = st.beta_columns((4,1,1))
                    X = panda_total_value_matrix.drop("target value",1)   #Feature Matrix
                    y = panda_total_value_matrix["target value"]          #Target Variable

                    
                    ### Ranking aller Features
                    model = LinearRegression()
                    #Initializing RFE model
                    rfe = RFE(model, 7)
                    #Transforming data using RFE
                    X_rfe = rfe.fit_transform(X, y)
                    #Fitting the data to model
                    model.fit(X_rfe, y)
                    #st.write(rfe.support_)
                    #st.write(f"Feature ranking: {rfe.ranking_}")
                    
                    # no of features
                    nof_list=np.arange(1, total_features_list_len)  
                    high_score=0

                    # Variable to store the optimum features
                    nof=0           
                    score_list =[]
                    for n in range(len(nof_list)):
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                        model = LinearRegression()
                        rfe = RFE(model,nof_list[n])
                        X_train_rfe = rfe.fit_transform(X_train,y_train)
                        X_test_rfe = rfe.transform(X_test)
                        model.fit(X_train_rfe,y_train)
                        score = model.score(X_test_rfe,y_test)
                        score_list.append(score)
                        if(score>high_score):
                            high_score = score
                            nof = nof_list[n]
                    
                

                    col1.write(f"Optimum number of features: {nof}")
                    col1.write(f"Score with {nof} features: {high_score}")

                    #####################################
                    cols = list(X.columns)
                    model = LinearRegression()
                    #Initializing RFE model
                    rfe = RFE(model, nof)             
                    #Transforming data using RFE
                    X_rfe = rfe.fit_transform(X, y)  
                    #Fitting the data to model
                    model.fit(X_rfe, y)              
                    temp = pd.Series(rfe.support_,index = cols)
                    selected_features_rfe = temp[temp==True].index
                    col1.write(selected_features_rfe)

                    X_selected = X[selected_features_rfe]
                    # st.write(X_selected)
                    X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y, test_size = 0.2, random_state = 0)
                    model.fit(X_selected_train, y_selected_train)
                    y_selected_pred = model.predict(X_selected_test)
                    
                    
                    variance_selected = explained_variance_score(y_selected_test, y_selected_pred)
                    max_error_selected = max_error(y_selected_test, y_selected_pred)
                    mean_absolute_error_selected = mean_absolute_error(y_selected_test, y_selected_pred)
                    mean_squared_error_selected = mean_squared_error(y_selected_test, y_selected_pred)
                    rmse_selected = np.sqrt(mean_squared_error(y_selected_test,y_selected_pred))
                    coefficient_of_determination = r2_score(y_selected_test, y_selected_pred)

                    col1.write(f"variance = {round(variance_selected, 2)}")
                    col1.write(f"max error = {round(max_error_selected, 2)}")
                    col1.write(f"mean absolute error = {round(mean_absolute_error_selected, 2)}")
                    col1.write(f"mean squared error = {round(mean_squared_error_selected, 2)}")
                    col1.write(f"RMSE = {round(rmse_selected, 2)}")
                    col1.write(f"RMSE = {round(rmse_selected, 2)}")
                    col1.write(f"R² = {round(coefficient_of_determination, 2)}")

                    pred_graph=plt.figure(figsize=(15,10))
                    plt.axline((0, 0), (1, 1), linewidth=4, color='r')
                    plt.scatter(y_selected_test, y_selected_pred)
                    plt.xlabel("target value")
                    plt.ylabel("predicted value")
            
            

                    expander = col1.beta_expander(
                        f"Target and predicted values of the Regression by RFE"
                    )      
                    expander.pyplot(pred_graph)
                    pred_test_matrix = pd.DataFrame()
                    pred_test_matrix['test data values'] = y_selected_test
                    pred_test_matrix['predicted values'] = y_selected_pred
                    expander.write(pred_test_matrix)

            if classifier_name == "Polynomial Regression":
        
                col1.header("Automatical feature selection by Recursive Feature Elimination")
                poly_degree=col1.number_input("polynomial degree for polynomial RFE", 1, 8, 2)
                rfe_activate=col1.button("Perform RFE")
            #initial_features = st.number_input("Select number of best features you want to determine", 1, (total_features_list_len - 1), 5, 1)
                if  rfe_activate==True:
                  #  col1, col2, col3 = st.beta_columns((4,1,1))
                    X = panda_total_value_matrix.drop("target value",1)   #Feature Matrix
                    y = panda_total_value_matrix["target value"]          #Target Variable

                    
                    ### Ranking aller Features
                    #model = LinearRegression()

                    model = make_pipeline(PolynomialFeatures(poly_degree), Ridge())
                    #Initializing RFE model
                    rfe = RFE(model, 7)
                    #Transforming data using RFE
                    X_rfe = rfe.fit_transform(X, y)
                    #Fitting the data to model
                    model.fit(X_rfe, y)
                    #st.write(rfe.support_)
                    #st.write(f"Feature ranking: {rfe.ranking_}")
                    
                    # no of features
                    nof_list=np.arange(1, total_features_list_len)  
                    high_score=0

                    # Variable to store the optimum features
                    nof=0           
                    score_list =[]
                    for n in range(len(nof_list)):
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                        model = make_pipeline(PolynomialFeatures(poly_degree), Ridge())
                        rfe = RFE(model,nof_list[n])
                        X_train_rfe = rfe.fit_transform(X_train,y_train)
                        X_test_rfe = rfe.transform(X_test)
                        model.fit(X_train_rfe,y_train)
                        score = model.score(X_test_rfe,y_test)
                        score_list.append(score)
                        if(score>high_score):
                            high_score = score
                            nof = nof_list[n]
                    
                                
                    col1.write(f"Optimum number of features: {nof}")
                    col1.write(f"Score with {nof} features: {high_score}")

                    #####################################
                    cols = list(X.columns)
                    model = make_pipeline(PolynomialFeatures(poly_degree), Ridge())
                    #Initializing RFE model
                    rfe = RFE(model, nof)             
                    #Transforming data using RFE
                    X_rfe = rfe.fit_transform(X, y)  
                    #Fitting the data to model
                    model.fit(X_rfe, y)              
                    temp = pd.Series(rfe.support_,index = cols)
                    selected_features_rfe = temp[temp==True].index
                    col1.write(selected_features_rfe)

                    X_selected = X[selected_features_rfe]
                    # st.write(X_selected)
                    X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y, test_size = 0.2, random_state = 0)
                    model.fit(X_selected_train, y_selected_train)
                    y_selected_pred = model.predict(X_selected_test)
                    
                    
                    variance_selected = explained_variance_score(y_selected_test, y_selected_pred)
                    max_error_selected = max_error(y_selected_test, y_selected_pred)
                    mean_absolute_error_selected = mean_absolute_error(y_selected_test, y_selected_pred)
                    mean_squared_error_selected = mean_squared_error(y_selected_test, y_selected_pred)
                    rmse_selected = np.sqrt(mean_squared_error(y_selected_test,y_selected_pred))
                    coefficient_of_determination = r2_score(y_selected_test, y_selected_pred)

                    col1.write(f"variance = {round(variance_selected, 2)}")
                    col1.write(f"max error = {round(max_error_selected, 2)}")
                    col1.write(f"mean absolute error = {round(mean_absolute_error_selected, 2)}")
                    col1.write(f"mean squared error = {round(mean_squared_error_selected, 2)}")
                    col1.write(f"RMSE = {round(rmse_selected, 2)}")
                    col1.write(f"RMSE = {round(rmse_selected, 2)}")
                    col1.write(f"R² = {round(coefficient_of_determination, 2)}")
                    
                


                    pred_graph=plt.figure(figsize=(15,10))
                    plt.axline((0, 0), (1, 1), linewidth=4, color='r')
                    plt.scatter(y_selected_test, y_selected_pred)
                    plt.xlabel("target value")
                    plt.ylabel("predicted value")
            
            

                    expander = col1.beta_expander(
                        f"Target and predicted values of the Regression by RFE"
                    )      
                    expander.pyplot(pred_graph)
                    pred_test_matrix = pd.DataFrame()
                    pred_test_matrix['test data values'] = y_selected_test
                    pred_test_matrix['predicted values'] = y_selected_pred
                    expander.write(pred_test_matrix)


        elif classifier_name == "KNN" or classifier_name == "SVM" or classifier_name == "Random Forest":
            # acc = accuracy_score(y_test, y_pred)
            # st.write(f"classifier = {classifier_name}")
            # st.write(f"accuracy = {acc}")
            # y_selected_pred = model.predict(X_selected_test)
                
                
            variance_computed = explained_variance_score(y_test, y_pred)
            max_error_computed = max_error(y_test, y_pred)
            mean_absolute_error_computed = mean_absolute_error(y_test, y_pred)
            mean_squared_error_computed = mean_squared_error(y_test, y_pred)
            rmse_computed = np.sqrt(mean_squared_error(y_test,y_pred))
            coefficient_of_determination = r2_score(y_test, y_pred)

            col1.write(f"regression = {classifier_name}")
            col1.write(f"variance = {round(variance_computed, 2)}")
            col1.write(f"max error = {round(max_error_computed, 2)}")
            col1.write(f"mean absolute error = {round(mean_absolute_error_computed, 2)}")
            col1.write(f"mean squared error = {round(mean_squared_error_computed, 2)}")
            col1.write(f"RMSE = {round(rmse_computed, 2)}")
            col1.write(f"R² = {round(coefficient_of_determination, 2)}")

            pred_graph=plt.figure(figsize=(15,10))
            plt.axline((0, 0), (1, 1), linewidth=4, color='r')
            plt.scatter(y_test, y_pred)
            plt.xlabel("target value", fontsize=20)
            plt.ylabel("predicted value", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            
            

            expander = col1.beta_expander(
                f"Target and predicted values of the Regression by RFE"
            )      
            expander.pyplot(pred_graph)
            pred_test_matrix = pd.DataFrame()
            pred_test_matrix['test data values'] = y_test
            pred_test_matrix['predicted values'] = y_pred
            expander.write(pred_test_matrix)


    else:
        None

    
