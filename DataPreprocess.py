import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, ReanalyseDataPath, VVVHPath, C2Path, NDVIPath, outPath):
        '''
        ReanalyseDataPath: Reanalysis data CSV file path
        VVVHPath: Backscatter coefficient and angle CSV file path
        C2Path: C2 Matrix CSV File Path
        NDVIPath: NVDI CSV File Path
        outPath: Output folder path
        '''
        
        # Initiate input and output file path
        self.ReanalyseDataPath = ReanalyseDataPath
        self.VVVHPath = VVVHPath
        self.C2Path = C2Path
        self.NDVIPath = NDVIPath
        self.outPath = outPath

        # Intermediate file
        self.Data = None
        self.DataMerge = None
        self.SAR_NDVI = None
        self.SAR_NDVI_Reanalyse = None
        self.ReanalyseData_GDD = None
        self.SAR_NDVI_Reanalyse_ACHU = None
        self.SAR_NDVI_Reanalyse_ACHU_selected_filtered = None

    def readData(self):
        # Importing data from different files into a single file
        SARpara_VVVH = pd.read_csv(self.ReanalyseDataPath, index_col=0)
        SARpara_C2 = pd.read_csv(self.C2Path, index_col=0)
        NDVI = pd.read_csv(self.NDVIPath, index_col=0)
        temp1 = pd.concat([SARpara_VVVH, SARpara_C2])
        temp1.Value = temp1.Value.astype(np.object)
        temp2 = pd.concat([temp1, NDVI])
        temp2.to_csv('Data_Unarranged.csv')
        Data_mean = self.Data[self.Data.Stastic_Type == 'mean']
        Data_mean.drop(columns = 'Stastic_Type', inplace=True)

        self.Data = Data_mean

    # Include raw SAR parameter data other than NDVI
    def addSAR(self):
        DataMerge = pd.DataFrame()
        parameterList = list(set(self.Data.Parameter_Type))
        parameterList.remove('NDVI')
        for paraName in tqdm(parameterList):
            Data_Cut = self.Data[self.Data.Parameter_Type == paraName]
            Data_Cut = Data_Cut.replace('--', np.nan)
            Data_Cut.drop(columns = 'Parameter_Type', inplace = True)
            Data_Cut['Value'] = Data_Cut['Value'].astype('float32')
            Data_Cut['std'] = Data_Cut['std'].astype('float32')
            if DataMerge.empty:
                Data_Cut = Data_Cut[['ID', 'Sample_ID', 'Crop_Type', 'lon', 'lat','Area', 'Date', 'Value', 'std']]
                Data_Cut = Data_Cut[Data_Cut['std']<=Data_Cut['std'].quantile(0.90)]
                Data_Cut.columns = ['ID', 'Sample_ID', 'Crop_Type', 'lon', 'lat', 'Area', 'Date', paraName, paraName+'_std']
                DataMerge = Data_Cut
            else:
                Data_Cut = Data_Cut[['ID', 'Date', 'Value', 'std']]
                Data_Cut = Data_Cut[Data_Cut['std']<=Data_Cut['std'].quantile(0.90)]
                Data_Cut.columns = ['ID', 'Date', paraName, paraName+'_std']
                DataMerge = pd.merge(DataMerge, Data_Cut, on=['ID', 'Date'], how='outer')
        self.DataMerge = DataMerge.drop_duplicates(subset=['ID', 'Date'])
    
    # Incorporating NDVI data
    def AddNDVI(self):
        NDVI = self.Data[(self.Data.Stastic_Type == 'mean') & (self.Data.Parameter_Type == 'NDVI')]
        NDVI = NDVI.replace('-999.0', np.nan)
        NDVI = NDVI.replace('--', np.nan)
        # NDVI = NDVI[~NDVI['Date'].isin([0830,0902])]
        NDVI.dropna(inplace = True)
        NDVI.Value = NDVI.Value.astype('float')
        NDVI = NDVI[NDVI.Value>0]
        NDVI.drop(columns = ['Stastic_Type', 'Sample_ID', 'Crop_Type', 'lon', 'lat', 'Parameter_Type', 'Area'], inplace=True)
        NDVI.columns = ['ID', 'Date', 'NDVI', 'NDVI_std']
        NDVI.NDVI = NDVI.NDVI.astype(np.double)
        NDVI.NDVI_std = NDVI.NDVI_std.astype(np.double)
        NDVI = NDVI[NDVI['NDVI_std']<=NDVI['NDVI_std'].quantile(0.9)]
        for i in NDVI.index:
            if (NDVI.loc[i,'NDVI']>1) or (NDVI.loc[i,'NDVI']<0.1):
                NDVI.loc[i,'NDVI'] = np.nan
        NDVI.dropna(inplace = True)
        # Establish a time index between NDVI and SAR parameters
        Data_SARPara = self.DataMerge
        Data_NDVI = NDVI
        Data_SARPara.Date = pd.to_datetime(Data_SARPara.Date, format = "%Y%m%d")
        Data_NDVI.Date = pd.to_datetime(Data_NDVI.Date, format = "%Y%m%d")

        Data_NDVI_Index_Array = np.sort(np.array(list(set(Data_NDVI.Date.map(lambda x:x.strftime('%Y%m%d')).astype('str')))))
        Data_SARPara_Index_Array = np.sort(np.array(list(set(Data_SARPara.Date.map(lambda x:x.strftime('%Y%m%d')).astype('str')))))    
        Data_NDVI_Index_Array_jint = np.sort(np.array(list(set(Data_NDVI.Date.map(lambda x:x.strftime('%j')).astype('int')))))
        Data_SARPara_Index_Array_jint = np.sort(np.array(list(set(Data_SARPara.Date.map(lambda x:x.strftime('%j')).astype('int')))))
        diffMatrix = np.abs(Data_NDVI_Index_Array_jint.reshape(len(Data_NDVI_Index_Array_jint),1) - Data_SARPara_Index_Array_jint.reshape(1,len(Data_SARPara_Index_Array_jint)))
        idx = np.argmin(diffMatrix, axis=1)

        idxDictory = dict(zip(Data_NDVI_Index_Array, Data_SARPara_Index_Array[idx]))

        popkey=list()
        for key in idxDictory:
            t1 = datetime.strptime(key, '%Y%m%d')
            t2 = datetime.strptime(idxDictory[key], '%Y%m%d')
            dif = (t1-t2).days
            if np.abs(dif)>1:
                popkey.append(key)
        for i in popkey:
            idxDictory.pop(i)

        MatchArray = np.vstack([np.array(list(idxDictory.keys())), np.array(list(idxDictory.values()))]).T
        MatchFrame = pd.DataFrame(MatchArray, columns=['Date','Nearest_Date'])
        MatchFrame.Date = pd.to_datetime(MatchFrame.Date)
        MatchFrame.Nearest_Date = pd.to_datetime(MatchFrame.Nearest_Date)
        Data_NDVI = pd.merge(Data_NDVI, MatchFrame, on='Date', how = 'left')

        Data_NDVI = Data_NDVI.reset_index()
        Data_NDVI.Nearest_Date = pd.to_datetime(Data_NDVI.Nearest_Date)
        self.SAR_NDVI = pd.merge(Data_SARPara,Data_NDVI,left_on=['Date', 'ID'],right_on=['Nearest_Date','ID'], how='outer')
        self.SAR_NDVI['Date'] = self.SAR_NDVI['Date_x']
        self.SAR_NDVI['Date'].loc[self.SAR_NDVI.Date.isnull()] = self.SAR_NDVI[self.SAR_NDVI.Date.isnull()]['Date_y']
        self.SAR_NDVI.drop(columns = ['Date_x', 'Date_y'], inplace=True)
    
    # Find the grid closest to longitude
    def searchNearestLon(self, lon):
            lonList = np.array([-82.0, -81.5, -81.0, -81.25, -81.75])
            Nearest_lon = lonList[np.argmin(np.abs(lonList-lon))]
            return Nearest_lon

    # Find the grid closest to latitude
    def searchNearestLat(self, lat):
        latList = np.array([42.0, 43.0, 42.75, 42.25, 42.5])
        Nearest_lat = latList[np.argmin(np.abs(latList-lat))]
        return Nearest_lat

    # Introducing reanalysis data
    def AddReanalysisData(self):
        ReanalyseData = pd.read_csv(self.ReanalyseDataPath)
        ReanalyseData.drop(columns=['Unnamed: 0'], inplace=True)
        ReanalyseData.set_index('index', inplace=True)
        ReanalyseData.index = pd.to_datetime(ReanalyseData.index)
        dataColumns = ['evabs', 'evatc', 'evavt', 'lai_hv', 'lai_lv', 'e', 'tp',
                'd2m', 't2m', 'skt', 'stl1', 'stl2', 'stl3', 'stl4', 'src', 'swvl1', 'swvl2',
            'swvl3', 'swvl4']
        ReanalyseData[dataColumns] = ReanalyseData[dataColumns].shift(-4)

        ReanalyseData_Rearranged = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                
                ReanalyseData_Cut = ReanalyseData.loc[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                ReanalyseData_Cut = ReanalyseData_Cut.resample('D').mean()
                if ReanalyseData_Rearranged.empty:
                    ReanalyseData_Rearranged = ReanalyseData_Cut
                else:
                    ReanalyseData_Rearranged = pd.concat([ReanalyseData_Rearranged, ReanalyseData_Cut])


        self.SAR_NDVI['Nearest_lon'] = self.SAR_NDVI.lon.map(lambda x:self.searchNearestLon(x))
        self.SAR_NDVI['Nearest_lat'] = self.SAR_NDVI.lat.map(lambda x:self.searchNearestLat(x))

        ReanalyseData_Rearranged.rename(columns = {'lon':'Nearest_lon','lat':'Nearest_lat'}, inplace = True)
        self.SAR_NDVI_Reanalyse = pd.merge(self.SAR_NDVI, ReanalyseData_Rearranged, left_on=['Date', 'Nearest_lon', 'Nearest_lat'], right_on=['index', 'Nearest_lon', 'Nearest_lat'], how='left')
        self.SAR_NDVI_Reanalyse.dropna(axis=0,subset = ['ID'], inplace=True)
    
    # Calculate cumulative temperature data based on analyzed data
    def generateAccumulatedTemperature(self):
        ReanalyseData = pd.read_csv(self.ReanalyseDataPath)
        ReanalyseData.drop(columns=['Unnamed: 0'], inplace=True)
        ReanalyseData = ReanalyseData.rename(columns={'index':'Date'})
        ReanalyseData.Date = pd.to_datetime(ReanalyseData.Date)
        ReanalyseData = ReanalyseData.set_index('Date')
        ReanalyseData = ReanalyseData.shift(-4, axis=0)
        ReanalyseData.head()

        # t2m
        ReanalyseData_GDD_t2m = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                t2m_mean_d = ReanalyseData_cut.resample('D').mean()['t2m']-273.15
                t2m_min_d = ReanalyseData_cut.resample('D').min()['t2m']-273.15
                t2m_max_d = ReanalyseData_cut.resample('D').max()['t2m']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), t2m_mean_d, t2m_min_d, t2m_max_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 't2m_mean', 't2m_min', 't2m_max']
                if ReanalyseData_GDD_t2m.empty:
                    ReanalyseData_GDD_t2m = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_t2m = pd.concat([ReanalyseData_GDD_t2m, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_t2m['CHU_t2m'] = (1.8 * (ReanalyseData_GDD_t2m.t2m_min - 4.4) + 3.33 * (ReanalyseData_GDD_t2m.t2m_max - 10) - 0.084 * (ReanalyseData_GDD_t2m.t2m_max - 10)**2) / 2.0
        ReanalyseData_GDD_t2m['CHU_t2m'] = ReanalyseData_GDD_t2m['CHU_t2m'].map(lambda x:x if x>0 else 0)
        ReanalyseData_GDD_t2m['ACHU_t2m'] = ReanalyseData_GDD_t2m.CHU_t2m.cumsum()

        # skt
        ReanalyseData_GDD_skt = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                skt_mean_d = ReanalyseData_cut.resample('D').mean()['skt']-273.15
                skt_min_d = ReanalyseData_cut.resample('D').min()['skt']-273.15
                skt_max_d = ReanalyseData_cut.resample('D').max()['skt']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), skt_mean_d, skt_min_d, skt_max_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 'skt_mean', 'skt_min', 'skt_max']
                if ReanalyseData_GDD_skt.empty:
                    ReanalyseData_GDD_skt = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_skt = pd.concat([ReanalyseData_GDD_skt, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_skt['CHU_skt'] = (1.8 * (ReanalyseData_GDD_skt.skt_min - 4.4) + 3.33 * (ReanalyseData_GDD_skt.skt_max - 10) - 0.084 * (ReanalyseData_GDD_skt.skt_max - 10)**2) / 2.0
        ReanalyseData_GDD_skt['CHU_skt'] = ReanalyseData_GDD_skt['CHU_skt'].map(lambda x:x if x>0 else 0)
        ReanalyseData_GDD_skt['ACHU_skt'] = ReanalyseData_GDD_skt.CHU_skt.cumsum()

        # stl1
        ReanalyseData_GDD_stl1 = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                stl1_max_d = ReanalyseData_cut.resample('D').max()['stl1']-273.15
                stl1_min_d = ReanalyseData_cut.resample('D').min()['stl1']-273.15
                stl1_mean_d = ReanalyseData_cut.resample('D').mean()['stl1']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), stl1_max_d, stl1_min_d, stl1_mean_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 'stl1_max', 'stl1_min', 'stl1_mean']
                if ReanalyseData_GDD_stl1.empty:
                    ReanalyseData_GDD_stl1 = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_stl1 = pd.concat([ReanalyseData_GDD_stl1, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_stl1['CHU_stl1'] = ReanalyseData_GDD_stl1.stl1_mean
        ReanalyseData_GDD_stl1['ACHU_stl1'] = ReanalyseData_GDD_stl1.CHU_stl1.cumsum()

        # stl2
        ReanalyseData_GDD_stl2 = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                stl2_max_d = ReanalyseData_cut.resample('D').max()['stl2']-273.15
                stl2_min_d = ReanalyseData_cut.resample('D').min()['stl2']-273.15
                stl2_mean_d = ReanalyseData_cut.resample('D').mean()['stl2']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), stl2_max_d, stl2_min_d, stl2_mean_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 'stl2_max', 'stl2_min', 'stl2_mean']
                if ReanalyseData_GDD_stl2.empty:
                    ReanalyseData_GDD_stl2 = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_stl2 = pd.concat([ReanalyseData_GDD_stl2, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_stl2['CHU_stl2'] = ReanalyseData_GDD_stl2.stl2_mean
        ReanalyseData_GDD_stl2['ACHU_stl2'] = ReanalyseData_GDD_stl2.CHU_stl2.cumsum()

        # stl3
        ReanalyseData_GDD_stl3 = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                stl3_max_d = ReanalyseData_cut.resample('D').max()['stl3']-273.15
                stl3_min_d = ReanalyseData_cut.resample('D').min()['stl3']-273.15
                stl3_mean_d = ReanalyseData_cut.resample('D').mean()['stl3']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), stl3_max_d, stl3_min_d, stl3_mean_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 'stl3_max', 'stl3_min', 'stl3_mean']
                if ReanalyseData_GDD_stl3.empty:
                    ReanalyseData_GDD_stl3 = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_stl3 = pd.concat([ReanalyseData_GDD_stl3, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_stl3['CHU_stl3'] = ReanalyseData_GDD_stl3.stl3_mean
        ReanalyseData_GDD_stl3['ACHU_stl3'] = ReanalyseData_GDD_stl3.CHU_stl3.cumsum()

        # stl4
        ReanalyseData_GDD_stl4 = pd.DataFrame()
        for i in tqdm(set(ReanalyseData.lon)):
            for j in set(ReanalyseData.lat):
                ReanalyseData_cut = ReanalyseData[(ReanalyseData.lon == i) & (ReanalyseData.lat == j)]
                stl4_max_d = ReanalyseData_cut.resample('D').max()['stl4']-273.15
                stl4_min_d = ReanalyseData_cut.resample('D').min()['stl4']-273.15
                stl4_mean_d = ReanalyseData_cut.resample('D').mean()['stl4']-273.15
                ReanalyseData_cut_ = pd.concat([ReanalyseData_cut.lon.resample('D').mean(), ReanalyseData_cut.lat.resample('D').mean(), stl4_max_d, stl4_min_d, stl4_mean_d],axis=1)
                ReanalyseData_cut_.columns = ['lon', 'lat', 'stl4_max', 'stl4_min', 'stl4_mean']
                if ReanalyseData_GDD_stl4.empty:
                    ReanalyseData_GDD_stl4 = ReanalyseData_cut_
                else:
                    ReanalyseData_GDD_stl4 = pd.concat([ReanalyseData_GDD_stl4, ReanalyseData_cut_],axis=0)

        ReanalyseData_GDD_stl4['CHU_stl4'] = ReanalyseData_GDD_stl4.stl4_mean
        ReanalyseData_GDD_stl4['ACHU_stl4'] = ReanalyseData_GDD_stl4.CHU_stl4.cumsum()

        ReanalyseData_GDD = pd.DataFrame()
        ReanalyseData_GDD = ReanalyseData_GDD_t2m
        ReanalyseData_GDD = pd.concat([ReanalyseData_GDD, ReanalyseData_GDD_skt], axis=1)
        ReanalyseData_GDD = pd.concat([ReanalyseData_GDD, ReanalyseData_GDD_stl1], axis=1)
        ReanalyseData_GDD = pd.concat([ReanalyseData_GDD, ReanalyseData_GDD_stl2], axis=1)
        ReanalyseData_GDD = pd.concat([ReanalyseData_GDD, ReanalyseData_GDD_stl3], axis=1)
        ReanalyseData_GDD = pd.concat([ReanalyseData_GDD, ReanalyseData_GDD_stl4], axis=1)

        ReanalyseData_GDD = ReanalyseData_GDD[['lon', 'lat', 'ACHU_t2m', 'ACHU_skt', 'ACHU_stl1', 'ACHU_stl2', 'ACHU_stl3', 'ACHU_stl4']]
        self.ReanalyseData_GDD = ReanalyseData_GDD.T.drop_duplicates().T

    # Introducing cumulative temperature data
    def introduceAccumulatedTemperature(self):
        self.SAR_NDVI_Reanalyse['Date'] = pd.to_datetime(self.SAR_NDVI_Reanalyse['Date'])
        ReanalyseData_ACHU = pd.read_csv('ReanalyseData_ACHU.csv', index_col=0)
        ReanalyseData_ACHU.index = pd.to_datetime(ReanalyseData_ACHU.index)

        SAR_NDVI_Reanalyse_temp = self.SAR_NDVI_Reanalyse
        SAR_NDVI_Reanalyse_temp['lon'] = SAR_NDVI_Reanalyse_temp.lon.map(lambda x:self.searchNearestLon(x))
        SAR_NDVI_Reanalyse_temp['lat'] = SAR_NDVI_Reanalyse_temp.lat.map(lambda x:self.searchNearestLat(x))

        ProcessCol = ['ACHU_t2m','ACHU_skt','ACHU_stl1','ACHU_stl2','ACHU_stl3','ACHU_stl4']
        ReanalyseData_ACHU0501 = ReanalyseData_ACHU['05-01']
        ReanalyseData_ACHU = ReanalyseData_ACHU[['lon','lat']]
        for lon in set(ReanalyseData_ACHU.lon):
            for lat in set(ReanalyseData_ACHU.lat):
                Specificlonlat = ReanalyseData_ACHU[(ReanalyseData_ACHU.lon==lon)&(ReanalyseData_ACHU.lat==lat)][ProcessCol]
                Specificlonlat0501 = ReanalyseData_ACHU0501[(ReanalyseData_ACHU0501.lon==lon)&(ReanalyseData_ACHU0501.lat==lat)][ProcessCol]
                ReanalyseData_ACHU.loc[(ReanalyseData_ACHU.lon==lon)&(ReanalyseData_ACHU.lat==lat), ProcessCol] = Specificlonlat.values-Specificlonlat0501.values
        ReanalyseData_ACHU.reset_index(inplace=True)

        self.SAR_NDVI_Reanalyse_ACHU = pd.merge(SAR_NDVI_Reanalyse_temp, ReanalyseData_ACHU, on=['lon','lat','Date'], how='left')

    def DataFilter(self):
        self.SAR_NDVI_Reanalyse_ACHU.set_index('Date', inplace=True)
        self.SAR_NDVI_Reanalyse_ACHU.index = pd.to_datetime( self.SAR_NDVI_Reanalyse_ACHU.index)
        plt.rc('font',family='Times New Roman')
        IDList = list(set(self.SAR_NDVI_Reanalyse_ACHU.ID))
        IDused = list()
        for i in tqdm(IDList, colour='red'):
            data_cut =  self.SAR_NDVI_Reanalyse_ACHU[ self.SAR_NDVI_Reanalyse_ACHU.ID==i].dropna(subset=['NDVI'])
            if ((data_cut[data_cut.ACHU_t2m<500].NDVI.mean()<=0.3) &
            (data_cut[((data_cut.ACHU_t2m>1500) & (data_cut.ACHU_t2m<3000))].NDVI.mean()>=0.6)):
                IDused.append(i)

        self.SAR_NDVI_Reanalyse_ACHU_selected_filtered = self.SAR_NDVI_Reanalyse_ACHU[np.isin(self.SAR_NDVI_Reanalyse_ACHU.ID, IDused)]

if __name__ == "__main__":
    # Input initialization file and output file path
    ReanalyseDataPath = '.csv' # Reanalysis data CSV file path
    VVVHPath = '.csv' # Backscatter coefficient and angle CSV file path
    C2Path = '.csv' # C2 Matrix CSV File Path
    NDVIPath = '.csv' # NVDI CSV File Path
    outPath = '.csv' # Output folder path

    dp = DataProcessor(ReanalyseDataPath, VVVHPath, C2Path, NDVIPath, outPath)
    dp.read_data()
    dp.addSAR()
    dp.AddNDVI()
    dp.AddReanalysisData()
    dp.generateAccumulatedTemperature()
    dp.introduceAccumulatedTemperature()
    dp.DataFilter()
    dp.SAR_NDVI_Reanalyse_ACHU_selected_filtered.to_csv(self.outPath)