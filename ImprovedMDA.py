import pandas as pd
import warnings
import numpy as np
from scipy.optimize import least_squares, minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import tqdm
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
import joblib as joblib
from random import sample
from sklearn import metrics
from scipy.stats import beta, norm
from collections import defaultdict

scores = defaultdict(list)
warnings.filterwarnings("ignore")


class ImprovedMDA:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, index_col=0)
        self.data.rename(columns={'AGDD_t2m': 'ACHU(t2m)', 'AGDD_skt': 'ACHU(skt)',
                                  'ACHU_stl1': 'SAT(stl1)', 'ACHU_stl2': 'SAT(stl2)',
                                  'ACHU_stl3': 'SAT(stl3)', 'ACHU_stl4': 'SAT(stl4)'}, inplace=True)

    def ParameterInitialization(self):
        # Set the name of the horizontal and vertical coordinates and the corresponding font format
        font1 = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 25,
                 }

        names = ['C11', 'C22', 'C12i', 'VV', 'VH', 'C12r', 'DpRVI', 'DPSVI',
                 'TrC2', 'detC2', 'm', 'beta', 'CR', 'NRPB', 'RVI', 'Span',
                 'evatc', 'tp', 't2m', 'skt', 'stl1', 'swvl1', 'ACHU_t2m', 'ACHU_skt',
                 'ACHU_stl1', 'ACHU_stl2', 'ACHU_stl3', 'ACHU_stl4']
        self.train_col = ['C11', 'C22', 'C12i', 'VV', 'VH', 'C12r', 'DpRVI', 'DPSVI',
                     'TrC2', 'detC2', 'm', 'beta', 'CR', 'NRPB', 'RVI', 'Span',
                     'evatc', 'tp', 't2m', 'skt', 'stl1', 'swvl1', 
                     'ACHU(t2m)','ACHU(skt)', 
                     'SAT(stl1)','SAT(stl2)','SAT(stl3)','SAT(stl4)']
        train_col_SAR = ['C11','C22','C12i','VV','VH','C12r','DpRVI','DPSVI',
                         'TrC2','detC2','m','beta','CR','NRPB','RVI','Span']
        train_col_RD = ['evatc','tp','t2m','skt','stl1','swvl1']
        train_col_AT = ['ACHU(t2m)','ACHU(skt)',
                         'SAT(stl1)','SAT(stl2)','SAT(stl3)','SAT(stl4)']
        train_col_SAR_idx = [self.train_col.index(i) for i in train_col_SAR]
        train_col_RD_idx = [self.train_col.index(i) for i in train_col_RD]
        train_col_AT_idx = [self.train_col.index(i) for i in train_col_AT]
        self.idx_List = [train_col_SAR_idx, train_col_RD_idx, train_col_AT_idx]
        self.group_names = ['SAR','RD','AT']

        self.real_col = ['NDVI']

    def calc_RMSE(x, y):
        return np.sqrt(metrics.mean_squared_error(x,y))
    
    def calc_Rsquare(data1, data2):
        R = np.corrcoef(data1, data2)
        return R[0, 1] * R[0, 1]

    # The importance ranking of features in the early vegetative stage
    def ImportanceRankingInEarlyVegetationStage(self, rootdir):
        # crossvalidate the scores on a number of different random splits of the data
        for sam_id in range(1,10):
            print('The accuracy of the {} model is being estimated...'.format(str(sam_id)))
            data_cut = self.data[(self.data['ACHU(t2m)']>500) & (self.data['ACHU(t2m)']<1000)].dropna(subset=self.train_col+self.real_col)
            train_data = data_cut[data_cut.Sample_ID!=sam_id]
            test_data = data_cut[data_cut.Sample_ID==sam_id]

            X_train, X_test = train_data[self.train_col].values, test_data[self.train_col].values
            Y_train, Y_test = train_data[self.real_col].values.flatten(), test_data[self.real_col].values.flatten()

            rootdir = r'/'
            regressor = joblib.load(rootdir+'RF_'+str(sam_id)+'_gt500_st1000'+'.m')

            acc = self.calc_Rsquare(Y_test, regressor.predict(X_test))
            for i in range(3):
                X_t = X_test.copy()
                per = np.random.permutation(X_t.shape[0])
                X_t[:, self.idx_List[i]] = X_t[per][:, np.array(self.idx_List[i])]
                shuff_acc = self.calc_Rsquare(Y_test, regressor.predict(X_t))
                scores[self.group_names[i]].append((acc-shuff_acc)/acc)
        print("Features sorted by their score:")
        print(sorted([(round(np.mean(score), 4), feat) for
                    feat, score in scores.items()], reverse=True))
        
    # The importance ranking of features in the late vegetative stage
    def ImportanceRankingInLateVegetationStage(self, rootdir):
        #crossvalidate the scores on a number of different random splits of the data
        for sam_id in range(1,10):
            print('The accuracy of the {} model is being estimated...'.format(str(sam_id)))
            data_cut = self.data[(self.data['ACHU(t2m)']>1000) & (self.data['ACHU(t2m)']<1800)].dropna(subset=self.train_col+self.real_col)
            train_data = data_cut[data_cut.Sample_ID!=sam_id]
            test_data = data_cut[data_cut.Sample_ID==sam_id]

            X_train, X_test = train_data[self.train_col].values, test_data[self.train_col].values
            Y_train, Y_test = train_data[self.real_col].values.flatten(), test_data[self.real_col].values.flatten()

            rootdir = r'/'
            regressor = joblib.load(rootdir+'RF_'+str(sam_id)+'_lt1000_st1800'+'.m')

            acc = self.calc_Rsquare(Y_test, regressor.predict(X_test))
            for i in range(3):
                X_t = X_test.copy()
                # np.random.shuffle(X_t[:, idx_List[i]])
                per = np.random.permutation(X_t.shape[0])
                X_t[:, self.idx_List[i]] = X_t[per][:, np.array(self.idx_List[i])]
                shuff_acc = self.calc_Rsquare(Y_test, regressor.predict(X_t))
                scores[self.group_names[i]].append((acc-shuff_acc)/acc)
        print("Features sorted by their score:")
        print(sorted([(round(np.mean(score), 4), feat) for
                    feat, score in scores.items()], reverse=True))

    # The importance ranking of features in the reproductive and mature stage
    def ImportanceRankingInLateVegetationStage(self, rootdir):
        for sam_id in range(1,10):
            print('The accuracy of the {} model is being estimated...'.format(str(sam_id)))
            data_cut = self.data[(self.data['ACHU(t2m)']>1800) & (self.data['ACHU(t2m)']<2500)].dropna(subset=self.train_col+self.real_col)
            train_data = data_cut[data_cut.Sample_ID!=sam_id]
            test_data = data_cut[data_cut.Sample_ID==sam_id]

            X_train, X_test = train_data[self.train_col].values, test_data[self.train_col].values
            Y_train, Y_test = train_data[self.real_col].values.flatten(), test_data[self.real_col].values.flatten()

            rootdir = r'/'
            regressor = joblib.load(rootdir+'RF_'+str(sam_id)+'_lt1800_st2500'+'.m')

            acc = self.calc_Rsquare(Y_test, regressor.predict(X_test))
            for i in range(3):
                X_t = X_test.copy()
                # np.random.shuffle(X_t[:, idx_List[i]])
                per = np.random.permutation(X_t.shape[0])
                X_t[:, self.idx_List[i]] = X_t[per][:, np.array(self.idx_List[i])]
                shuff_acc = self.calc_Rsquare(Y_test, regressor.predict(X_t))
                scores[self.group_names[i]].append((acc-shuff_acc)/acc)
        print("Features sorted by their score:")
        print(sorted([(round(np.mean(score), 4), feat) for
                    feat, score in scores.items()], reverse=True))
        
if __name__ == '__main__':
    # The path where the model parameter file (. pth) is located
    rootdir = '/'
    # Initiate the calss of ImprovedMDA
    IMDA = ImprovedMDA()
    IMDA.ParameterInitialization()
    IMDA.ImportanceRankingInEarlyVegetationStage()
    IMDA.ImportanceRankingInLateVegetationStage()
    IMDA.ImportanceRankingInLateVegetationStage()