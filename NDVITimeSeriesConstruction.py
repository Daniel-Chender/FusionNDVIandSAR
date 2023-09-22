import pandas as pd
import warnings
import numpy as np
from scipy.optimize import least_squares, curve_fit, minimize
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from scipy import optimize, signal, stats
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import joblib
from random import sample
from sklearn import metrics

warnings.filterwarnings("ignore")


class NDVITimeSeriesConstruction:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)
        self.data = self.data[self.data.Crop_Type_ID==2]
        self.data = self.data[self.data.Sample_ID==2]
        self.NDVIindex = self.data[self.data.NDVI.notnull()].index
        self.data.Date = pd.to_datetime(self.data.Date)
        self.data['Doy'] = self.data.Date.map(lambda x: x.strftime('%j'))
        self.data['Doy'] = self.data['Doy'].astype('int')
        self.data.set_index('Date', inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        self.data_NDVI = self.data.dropna(subset=['NDVI'])
        self.Doy_group = self.data_NDVI.groupby('Doy').mean()
        self.Doy_group.reset_index(inplace=True)
        self.font = {'family': 'Times New Roman',
                    'weight': 'normal', 'size': 10}

    def func(self, t, p0):
        a1, b1, c1, d1, a2, b2, c2, d2= p0
        return c1/(1+np.exp(a1+b1*t))+d1+c2/(1+np.exp(a2+b2*t))+d2

    def func_error(self, p0, y, x):
        return np.sum((y - self.func(x, p0))**2)

    def func_error_P(self, p0, y, x):
        P_ = np.zeros(1)
        return np.sum(P_*(y - self.func(x, p0))**2)

    def residuals(self, p0, y, x):
        P_ = np.zeros(1)
        return P_*(y - self.func(x, p0))

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mae(self, predictions, targets):
        return np.abs(predictions - targets).mean()

    def calc_Rsquare(self, data1, data2):
                R = np.corrcoef(data1, data2)
                return R[0, 1] * R[0, 1]

    def LogisticFunc(self, x, bias, slope):
        return 1/(1+np.exp(-slope*(x-bias)))
    
    def calc_Rsquare(data1, data2):
        R = np.corrcoef(data1, data2)
        return R[0, 1] * R[0, 1]
    
    def Relu(x, a, b):
        interval0 = [0 if (i<a) else 0 for i in x]
        interval1 = [1/(b-a)*(i-a) if (i>=a and i<b) else 0 for i in x]
        interval2 = [1 if (i>=b) else 0 for i in x]
        y = np.array(interval0) +  np.array(interval1) +  np.array(interval2)
        return y

    def InitializeTemplate(self):
        dataset = self.Doy_group.dropna(subset=['NDVI', 'Doy'])
        dataset.sort_values(by="Doy" ,ascending=True, inplace=True)
        # t = np.array(dataCut.Date.map(lambda x:x .strftime("%j"))).astype(np.double)
        t = np.array(dataset.Doy)
        # t = t-t[0]
        y = np.array(dataset.NDVI)

        d1_0 = min(y[0:np.argmax(y)])
        d2_0 = min(y[np.argmax(y):len(y)])
        c1_0 = max(y)-d1_0
        c2_0 = max(y)-d2_0

        B1 = np.ones((np.argmax(y),1))
        y1 = t[0:np.argmax(y)].reshape(np.argmax(y),1)
        B1 = np.hstack((B1, y1))
        l1 = (np.log(c1_0/(y[0:np.argmax(y)]-d1_0))-1).reshape(np.argmax(y),1)
        inf_index = np.where(l1==np.inf)
        B1 = np.delete(B1, inf_index, axis=0)
        l1 = np.delete(l1, inf_index, axis=0)
        B1 = np.mat(B1)
        l1 = np.mat(l1)
        X1 = np.dot(np.dot(np.linalg.inv(np.dot(B1.T,B1)),B1.T),l1)
        a1_0, b1_0 = X1

        B2 = np.ones((len(y)-np.argmax(y),1))
        y2 = t[np.argmax(y):].reshape(len(y)-np.argmax(y),1)
        B2 = np.hstack((B2, y2))
        l2 = (np.log(c2_0/(y[np.argmax(y):]-d2_0))-1).reshape(len(y)-np.argmax(y),1)
        inf_index = np.where(l2==np.inf)
        B2 = np.delete(B2, inf_index, axis=0)
        l2 = np.delete(l2, inf_index, axis=0)
        B2 = np.mat(B2)
        l2 = np.mat(l2)
        X2 = np.dot(np.dot(np.linalg.inv(np.dot(B2.T,B2)),B2.T),l2)

        a2_0, b2_0 = X2

        self.p0 = [np.double(i) for i in [a1_0,b1_0,c1_0,d1_0,a2_0,b2_0,c2_0,d2_0]]
        self.p0 = [-41.90958988,   0.15567363,   0.62383708,  -3.72372108,
            -21.18274721,   0.11521048,  -0.70802628,   3.97089538]

        r = optimize.basinhopping(self.func_error, self.p0,
            niter = 100,
            minimizer_kwargs={"method":"L-BFGS-B",
            "args":(y, t)})
        
        x = np.arange(100,330,1)
        plt.figure(dpi=150)
        plt.scatter(t, y, c='r', s=1, label=u"Real Data")
        plt.plot(x, self.func(x, r.x), 'k--', label=u"Fitting Data")
        plt.xlabel('Doy')
        plt.ylabel('NDVI')
        plt.text(300, 0.85,'R$^2$={:.2f}'.format(self.calc_Rsquare(y, self.func(t, r.x))))
        plt.legend(loc="upper left")
        plt.show()

    def BuildWeightConversionFunction(self):
        fig, ax1 = plt.subplots(dpi=300)
        Error = self.data[self.data.Error.notnull()].Error.values
        Error = np.sqrt(Error)
        plt.grid(ls='--')
        ax1.hist(Error, bins=1000, color='#BD2A2E')
        ax1.set_xlabel("Error", family='Times New Roman', size=15)
        ax1.set_ylabel("Frequency", family='Times New Roman', size=15)
        plt.xticks(family='Times New Roman')
        plt.yticks(family='Times New Roman')
        plt.ylim(0,1000)

        ax2 = ax1.twinx()
        res_freq = stats.relfreq(Error, numbins=100)
        x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
        cdf_value = np.cumsum(res_freq.frequency)
        ax2.plot(x, 1-cdf_value, '--', c='k', linewidth=2)
        ax2.set_ylabel("Weight", family='Times New Roman', size=15)

        plt.xlim(0,0.1)
        plt.ylim(-0.01, 1.01)
        plt.xticks(family='Times New Roman')
        plt.yticks(family='Times New Roman')
        plt.show()

        self.data.reset_index(inplace=True)
        self.data['P'] = np.nan

        No_NDVI_index = self.data[self.data.Error.notnull()].index
        Error = np.sqrt(self.data[self.data.Error.notnull()].Error.values)
        self.data.loc[No_NDVI_index, 'P'] = np.array([self.GetP(i, x, cdf_value) for i in Error])
        # LogisticFunc(np.log10(1/Error),np.mean(np.log10(1/Error)),np.std(np.log10(1/Error)))
        NDVI_index = self.data[self.data.NDVI.notnull()].index
        self.data.loc[NDVI_index, 'P'] = 1

        self.data[self.data.NDVI.isnull()&self.data.Error.notnull()][['Date','P']]

        self.data = self.data.set_index('Date')
        self.data.index = pd.to_datetime(self.data.index)
        self.data2019 = self.data['2019']
        self.data2020 = self.data['2020']
        self.data2021 = self.data['2021']
    
    # Weight Acquisition function
    def GetP(Error, x, cdf_value):
        cdf_value = 1-cdf_value
        idx = np.argmin(np.abs(x-Error))
        return cdf_value[idx]
    
    def TimeSeriesConstruction_VegetativeStage(self):
        #150-225 defect
        RecordForm = list()
        font = {'family': 'Times New Roman',
            'weight': 'normal', 'size':9}

        start = 160
        end = 200

        for i in tqdm(chosenIndex):
            NDVIcount = self.data2019[self.data2019.NDVI.notnull()].ID.value_counts()
            chosenIndex = list(NDVIcount[:int(1/10*len(NDVIcount))].index)
            data_cut = self.data2019[self.data2019.ID==i]
            data_cut.sort_values('Doy', inplace=True)

            y = data_cut.dropna(subset=['NDVI', 'Doy']).NDVI.values
            t = data_cut.dropna(subset=['NDVI', 'Doy']).Doy.values
            P = data_cut.dropna(subset=['NDVI', 'Doy']).P.values
            P_ = P
            plsq4 = least_squares(self.residuals, self.p0, args=(y, t))

            y_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).NDVI.values
            t_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).Doy.values
            P_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).P.values

            y_undefect = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).NDVI.values
            t_undefect = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).Doy.values
            P_1 = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).P.values
            P_ = P_1
            plsq1 = least_squares(self.residuals, self.p0, args=(y_undefect, t_undefect))

            y_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).Estimated_NDVI.values
            t_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).Doy.values
            P_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).P.values
            P_ = P_2
            plsq2 = least_squares(self.residuals, self.p0, args=(y_2, t_2))

            P_ = np.hstack([P_1, P_2])
            y_ = np.hstack([y_undefect, y_2])
            t_ = np.hstack([t_undefect, t_2])

            if len(y_)<6:
                continue
            plsq3 = least_squares(self.residuals, self.p0, args=(y_, t_))

            # plt.figure(dpi=500)
            fig, ax = plt.subplots(dpi=500)
            x = np.arange(100,340,1)
            ax.fill_betweenx([0, 1], start, end, facecolor ='lightgrey')
            plt.scatter(t_defect, y_defect, s=40, marker='o', edgecolor='k', c='white', label='Observed NDVI$_{defect}$')
            # plt.scatter(t_undefect, y_undefect, s=40, marker='o', edgecolor='k', c='#7D7D7D', label='Observed NDVI')
            plt.scatter(t_undefect, y_undefect, s=40, marker='o', edgecolor='k', c='green', label='Observed NDVI')
            # plt.scatter(t_2, y_2, s=8, c='r', label='Estimated NDVI')
            plt.errorbar(t_2, y_2,\
                    yerr=(1/P_2).reshape(len(P_2))/50,\
                    fmt="o",color="#EA3723", mec='k', ecolor='k',elinewidth=1,capsize=3,capthick=0.5,\
                        ms=6, label='Estimate NDVI')
            plt.plot(x, self.func(x, plsq1.x), 'k:', lw=2, label=u"Fitting Curve$_{defect}$")
            plt.plot(x, self.func(x, plsq2.x), ':', color='darkred', lw=2, label=u"Fitting Curve$_{ED}$")
            plt.plot(x, self.func(x, plsq4.x), ':', color='darkgreen', lw=2, label=u"Fitting Curve$_{OD}$")
            
            # plt.legend(loc='upper left', prop=font, frameon=False, ncol=1)
            plt.xticks(family='Times New Roman')
            plt.yticks(family='Times New Roman')
            plt.xlim(100,350)
            plt.ylim(0,1)
            plt.xlabel('DOY', family='Times New Roman', fontsize=15)
            plt.ylabel('NDVI', family='Times New Roman', fontsize=15)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.savefig('./2019/TP/'+str(i)+'.png', dpi=500)
            x_ = x
            x = np.arange(start,end+1,1)
            RecordForm.append([self.calc_Rsquare(self.func(x, plsq1.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq1.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq1.x), self.func(x, plsq4.x))), 
                            self.calc_Rsquare(self.func(x, plsq2.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq2.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq2.x), self.func(x, plsq4.x))),
                            self.calc_Rsquare(self.func(x, plsq3.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq3.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq3.x), self.func(x, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq1.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq1.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq1.x), self.func(x_, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq2.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq2.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq2.x), self.func(x_, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq3.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq3.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq3.x), self.func(x_, plsq4.x))),
                            ])
            RecordFrame = pd.DataFrame(RecordForm, columns=['DefectNDVI_R2', 'DefectNDVI_MAE', 'DefectNDVI_RMSE', 'OnlyEstimatedNDVI_R2', 'OnlyEstimatedNDVI_MAE', 'OnlyEstimatedNDVI_RMSE', 'EstimatedNDVIAndDefectNDVI_R2', 'EstimatedNDVIAndDefectNDVI_MAE', 'EstimatedNDVIAndDefectNDVI_RMSE',
                                                            'DefectNDVI_R2_', 'DefectNDVI_MAE_', 'DefectNDVI_RMSE_', 'OnlyEstimatedNDVI_R2_', 'OnlyEstimatedNDVI_MAE_', 'OnlyEstimatedNDVI_RMSE_', 'EstimatedNDVIAndDefectNDVI_R2_', 'EstimatedNDVIAndDefectNDVI_MAE_', 'EstimatedNDVIAndDefectNDVI_RMSE_'])
            RecordFrame.to_csv('.csv')
            plt.close()

    def TimeSeriesConstruction_ReproductiveStage(self):
        RecordForm = list()
        #150-225 defect
        font = {'family': 'Times New Roman',
            'weight': 'normal', 'size':9}

        start = 160
        end = 180

        for i in tqdm(self.chosenIndex):
            data_cut = self.data2019[self.data2019.ID==i]
            data_cut.sort_values('Doy', inplace=True)

            y = data_cut.dropna(subset=['NDVI', 'Doy']).NDVI.values
            t = data_cut.dropna(subset=['NDVI', 'Doy']).Doy.values
            P = data_cut.dropna(subset=['NDVI', 'Doy']).P.values
            P_ = P
            plsq4 = least_squares(self.residuals, self.p0, args=(y, t))

            y_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).NDVI.values
            t_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).Doy.values
            P_defect = data_cut[(data_cut.Doy>start)&(data_cut.Doy<end)].dropna(subset=['NDVI', 'Doy']).P.values

            y_undefect = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).NDVI.values
            t_undefect = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).Doy.values
            P_1 = data_cut[(data_cut.Doy<start)|(data_cut.Doy>end)].dropna(subset=['NDVI', 'Doy']).P.values
            P_ = P_1
            plsq1 = least_squares(self.residuals, self.p0, args=(y_undefect, t_undefect))

            y_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).Estimated_NDVI.values
            t_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).Doy.values
            P_2 = data_cut.dropna(subset=['Estimated_NDVI', 'Doy', 'P']).P.values
            P_ = P_2
            plsq2 = least_squares(self.residuals, self.p0, args=(y_2, t_2))

            P_ = np.hstack([P_1, P_2])
            y_ = np.hstack([y_undefect, y_2])
            t_ = np.hstack([t_undefect, t_2])

            if len(y_)<6:
                continue
            plsq3 = least_squares(self.residuals, self.p0, args=(y_, t_))

            # plt.figure(dpi=500)
            fig, ax = plt.subplots(dpi=500)
            x = np.arange(100,340,1)
            ax.fill_betweenx([0, 1], start, end, facecolor ='lightgrey')
            plt.scatter(t_defect, y_defect, s=40, marker='o', edgecolor='k', c='white', label='Observed NDVI$_{defect}$')
            # plt.scatter(t_undefect, y_undefect, s=40, marker='o', edgecolor='k', c='#7D7D7D', label='Observed NDVI')
            plt.scatter(t_undefect, y_undefect, s=40, marker='o', edgecolor='k', c='green', label='Observed NDVI')
            # plt.scatter(t_2, y_2, s=8, c='r', label='Estimated NDVI')
            plt.errorbar(t_2, y_2,\
                    yerr=(1/P_2).reshape(len(P_2))/40,\
                    fmt="o",color="#EA3723", mec='k', ecolor='k',elinewidth=1,capsize=3,capthick=0.5,\
                        ms=6, label='Estimate NDVI')
            plt.plot(x, self.func(x, plsq1.x), 'k:', lw=2, label=u"Fitting Curve$_{defect}$")
            plt.plot(x, self.func(x, plsq2.x), ':', color='darkred', lw=2, label=u"Fitting Curve$_{ED}$")
            plt.plot(x, self.func(x, plsq4.x), ':', color='darkgreen', lw=2, label=u"Fitting Curve$_{OD}$")
            # ax.fill_betweenx([0, 1], start, end, facecolor ='lightgrey')
            # plt.legend(loc='upper left', prop=font, frameon=False, ncol=1)
            plt.xticks(family='Times New Roman')
            plt.yticks(family='Times New Roman')
            plt.xlim(100,350)
            plt.ylim(0,1)
            plt.xlabel('DOY', family='Times New Roman', fontsize=15)
            plt.ylabel('NDVI', family='Times New Roman', fontsize=15)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.savefig('./2019/TP1/'+str(i)+'.png', dpi=500)
            x_ = x
            x = np.arange(start,end+1,1)
            RecordForm.append([self.calc_Rsquare(self.func(x, plsq1.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq1.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq1.x), self.func(x, plsq4.x))), 
                            self.calc_Rsquare(self.func(x, plsq2.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq2.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq2.x), self.func(x, plsq4.x))),
                            self.calc_Rsquare(self.func(x, plsq3.x), self.func(x, plsq4.x)), metrics.mean_absolute_error(self.func(x, plsq3.x), self.func(x, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x, plsq3.x), self.func(x, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq1.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq1.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq1.x), self.func(x_, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq2.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq2.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq2.x), self.func(x_, plsq4.x))),
                            self.calc_Rsquare(self.func(x_, plsq3.x), self.func(x_, plsq4.x)), metrics.mean_absolute_error(self.func(x_, plsq3.x), self.func(x_, plsq4.x)), np.sqrt(metrics.mean_squared_error(self.func(x_, plsq3.x), self.func(x_, plsq4.x))),
                            ])
            RecordFrame = pd.DataFrame(RecordForm, columns=['DefectNDVI_R2', 'DefectNDVI_MAE', 'DefectNDVI_RMSE', 'OnlyEstimatedNDVI_R2', 'OnlyEstimatedNDVI_MAE', 'OnlyEstimatedNDVI_RMSE', 'EstimatedNDVIAndDefectNDVI_R2', 'EstimatedNDVIAndDefectNDVI_MAE', 'EstimatedNDVIAndDefectNDVI_RMSE',
                                                            'DefectNDVI_R2_', 'DefectNDVI_MAE_', 'DefectNDVI_RMSE_', 'OnlyEstimatedNDVI_R2_', 'OnlyEstimatedNDVI_MAE_', 'OnlyEstimatedNDVI_RMSE_', 'EstimatedNDVIAndDefectNDVI_R2_', 'EstimatedNDVIAndDefectNDVI_MAE_', 'EstimatedNDVIAndDefectNDVI_RMSE_'])
            RecordFrame.to_csv('.csv')
            plt.close()

if __name__ == '__main__':
    NSC = NDVITimeSeriesConstruction()
    NSC.InitializeTemplate()
    NSC.BuildWeightConversionFunction()
    NSC.TimeSeriesConstruction_VegetativeStage()
    NSC.TimeSeriesConstruction_ReproductiveStage()