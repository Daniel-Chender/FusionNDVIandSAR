import pandas as pd
import warnings
import os
import numpy as np
from scipy.optimize import least_squares, minimize
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import beta, norm, pearsonr, gaussian_kde
from scipy.signal import find_peaks
from tqdm import tqdm
import joblib
from random import sample
from statistics import mean
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

class RandomForestModel:
    def __init__(self):
        pass
    
    def calc_Rsquare(self, data1, data2):
        """
        Calculate the R-square value between two datasets.
        
        Args:
            data1 (array-like): First dataset.
            data2 (array-like): Second dataset.
            
        Returns:
            float: R-square value.
        """
        R = np.corrcoef(data1, data2)
        return R[0, 1] * R[0, 1]
    
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        data = pd.read_csv(file_path)
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the loaded data.
        
        Args:
            data (pandas.DataFrame): Loaded data.
            
        Returns:
            pandas.DataFrame: Preprocessed data.
        """

        # Clean data
        preprocessed_data = data.reset_index(drop=True)
        preprocessed_data = preprocessed_data.drop(columns=['index'])
        return preprocessed_data
    
    def train_model(self, data):
        """
        Preprocess the loaded data.
        
        Args:
            data (pandas.DataFrame): Loaded data.
            
        Returns:
            List: Regression result.
        """
        # Initialize the result list
        Regression_Result = list()
        
        count = 1
        train_col = ['C11','C22', 'C12i', 'VV', 'VH', 'C12r', 'DpRVI', 'DPSVI', 
                                'TrC2', 'detC2', 'm', 'beta', 'CR', 'NRPB', 'RVI', 'Span',
                                'evatc', 'tp', 't2m', 'skt','stl1', 'swvl1', 'ACHU_t2m', 'ACHU_skt', 
                                'ACHU_stl1', 'ACHU_stl2', 'ACHU_stl3', 'ACHU_stl4']
        real_col = ['NDVI']
        arr = np.arange(1,11)
        arr = np.delete(arr,np.where(arr==2))

        for sam_id in arr:
            train_data = data[(data.Sample_ID!=sam_id) & (data.Sample_ID!=2)]
            test_data = data[data.Sample_ID==sam_id]
            
            if not os.path.exists('RF_'+str(count)+'_sp.m'):
                    # Prepare training data
                    dataset = train_data[(train_data.ACHU_t2m>800) & (train_data.ACHU_t2m<3800)].dropna(subset=train_col+real_col)
                    
                    X = dataset[train_col].values
                    y = dataset[real_col].values
                    
                    regressor = RandomForestRegressor(n_estimators=400, random_state=2, n_jobs=8)
                    regressor.fit(X, y)
                    y_pred = regressor.predict(X)
                    joblib.dump(regressor, 'RF_'+str(count)+'_sp.m') 
                    
                    # Evaluate regression performance
                    print('##########'+'Regression accuracy_'+str(count)+'#########')
                    a = y.reshape(len(y))
                    b = y_pred.reshape(len(y_pred))
                    print('Mean Absolute Error:', metrics.mean_absolute_error(a, b))
                    print('Mean Squared Error:', metrics.mean_squared_error(a, b))
                    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(a, b)))
                    print('r2_score:', self.calc_Rsquare(a.reshape(len(a)), b.reshape(len(b))))
                    
                    Regression_Result.append([count, metrics.mean_absolute_error(a, b), metrics.mean_squared_error(a, b), 
                                            np.sqrt(metrics.mean_squared_error(a, b)), self.calc_Rsquare(a.reshape(len(a)), b.reshape(len(b)))])

        return Regression_Result
    
    def evaluate_model(self, data):
        """
        Evaluate a trained machine learning model on the test set.
        
        Args:
            model (object): Trained model.
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target variable.
            
        Returns:
            List: Predict result.
        """
        # Initialize the result list
        Prediction_Result = list()

        count = 1
        test_col = ['C11','C22', 'C12i', 'VV', 'VH', 'C12r', 'DpRVI', 'DPSVI', 
                                'TrC2', 'detC2', 'm', 'beta', 'CR', 'NRPB', 'RVI', 'Span',
                                'evatc', 'tp', 't2m', 'skt','stl1', 'swvl1', 'ACHU_t2m', 'ACHU_skt', 
                                'ACHU_stl1', 'ACHU_stl2', 'ACHU_stl3', 'ACHU_stl4']
        
        real_col = ['NDVI']
        arr = np.arange(1,11)
        arr = np.delete(arr,np.where(arr==2))

        for sam_id in arr:
            test_data = data[data.Sample_ID==sam_id]

            dataset_test = test_data[(test_data.ACHU_t2m>800) & (test_data.ACHU_t2m<3800)].dropna(subset=test_col+real_col)

            X_test = dataset_test[test_col].values
            y_test = dataset_test[real_col].values
            
            regressor = joblib.load('RF_'+str(count)+'_sp.m')
            y_pred_test = regressor.predict(X_test)
            
            print('##########'+'Prediction accuracy_'+str(count)+'#########')
            a = y_test.reshape(len(y_test))
            b = y_pred_test.reshape(len(y_pred_test))
            print('Mean Absolute Error:', metrics.mean_absolute_error(a, b))
            print('Mean Squared Error:', metrics.mean_squared_error(a, b))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(a, b)))
            print('r2_score:', self.calc_Rsquare(a.reshape(len(a)), b.reshape(len(b))))
            Prediction_Result.append([count, metrics.mean_absolute_error(a, b), metrics.mean_squared_error(a, b), 
                                    np.sqrt(metrics.mean_squared_error(a, b)), self.calc_Rsquare(a.reshape(len(a)), b.reshape(len(b)))])
            
            x = a
            y = b
            if sam_id == 1:
                    observed_NDVI = x
                    estimated_NDVI = y
            else:
                    observed_NDVI = np.hstack([observed_NDVI, x])
                    estimated_NDVI = np.hstack([estimated_NDVI, y])
            BIAS = np.mean(x - y)
            MAE = metrics.mean_absolute_error(x, y)
            RMSE = np.sqrt(metrics.mean_squared_error(x, y))
            R2 = self.calc_Rsquare(x,y)
            def best_fit_slope_and_intercept(xs, ys):
                    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
                    b = np.mean(ys) - m * np.mean(xs)
                    return m, b
            m,b= best_fit_slope_and_intercept(x, y)
            # ===========Calculate the point density==========
            xy = np.vstack([x, y])
            z = stats.gaussian_kde(xy)(xy)
            # ===========Sort the points by density, so that the densest points are plotted last===========
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            fig, ax = plt.subplots(figsize=(7,5),dpi=300)
            plt.plot([0, 1], [0, 1], 'black', lw=0.8, label='$1:1 line$')  # The drawn 1:1 line has a color of black and a width of 0.8
            plt.scatter(x, y, c=z, s=1,cmap='Spectral_r')
            plt.plot(x, m*x+b, 'red', lw=0.8, label='$pre_line$')      # Regression line between predicted and measured data
            plt.legend(loc="lower right", frameon=False) # add legend
            plt.plot()
            plt.axis([0, 1, 0, 1])  # Set the range of lines

            plt.ylabel('$Estimated NDVI$',family = 'Times New Roman', fontsize=15)
            plt.xlabel('$Observed NDVI$',family = 'Times New Roman', fontsize=15)
            plt.xticks(fontproperties='Times New Roman')
            plt.yticks(fontproperties='Times New Roman')
            plt.text(0.025, 0.95, '$bias=%.5f$' % BIAS, family = 'Times New Roman')
            plt.text(0.025, 0.90, '$MAE=%.3f$' % MAE, family = 'Times New Roman')
            plt.text(0.025, 0.85, '$RMSE=%.3f$'  % RMSE, family = 'Times New Roman')
            plt.text(0.025, 0.80, '$R^2=%.3f$' % R2, family = 'Times New Roman')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.colorbar()
            plt.show()
        
        return Prediction_Result
    
    def result_analysis(self, data, Regression_Result, Prediction_Result):
        Regression_ResultFrame = pd.DataFrame(Regression_Result, columns = ['count', 'MAE', 'MSE', 'RMSE', 'R2'])
        Prediction_ResultFrame = pd.DataFrame(Prediction_Result, columns = ['count', 'MAE', 'MSE', 'RMSE', 'R2'])

        # Print overall accuracy results
        print(Prediction_ResultFrame[-6:].mean())

        train_col = ['C11','C22', 'C12i', 'VV', 'VH', 'C12r', 'DpRVI', 'DPSVI', 
             'TrC2', 'detC2', 'm', 'beta', 'CR', 'NRPB', 'RVI', 'Span',
            'evatc', 'tp', 't2m', 'skt','stl1', 'swvl1', 'ACHU_t2m', 'ACHU_skt', 
            'ACHU_stl1', 'ACHU_stl2', 'ACHU_stl3', 'ACHU_stl4']
        data_selected = data.dropna(subset=train_col)
        data_selected = data_selected[(data_selected.ACHU_t2m>800) & (data_selected.ACHU_t2m<3800)]

        # Load the model for inference
        for i in range(9):
            print('Execute estimation '+str(i+1))
            regressor = joblib.load('RF_'+str(i+1)+'_sp.m')
            data_X = data_selected[train_col].values
            data_selected['Estimated_NDVI_'+str(i+1)] = regressor.predict(data_X)

        # Visualization error results
        NDVI_idx = data_selected.columns.tolist().index('Estimated_NDVI_1')
        data_selected['Error'] = data_selected.iloc[:,NDVI_idx:NDVI_idx+6].var(axis=1)
        data_selected['Estimated_NDVI'] = data_selected.iloc[:,NDVI_idx:NDVI_idx+6].median(axis=1)
        data_selected.Error.hist(bins=1000)

        # Draw a histogram of error distribution
        data_selected['Residual'] = data_selected['Estimated_NDVI']-data_selected['NDVI']
        data_selected.Residual.hist(bins=1000)



# Example usage of the RandomForestModel class
if __name__ == '__main__':
    # Create an instance of the RandomForestModel class
    RFM = RandomForestModel()

    # Load the data from a CSV file
    data = RFM.load_data(r'.csv')

    # Preprocess the loaded data
    preprocessed_data = RFM.preprocess_data(data)


    # Train and test a machine learning model on the training set
    Regression_Result = RFM.train_model(data)
    Prediction_Result = RFM.evaluate_model(data)

    # Result analysis and visualization
    evaluation_metrics = RFM.result_analysis(data, Regression_Result, Prediction_Result)
