# -*- coding: utf-8 -*-
#%% import packages
import os
from osgeo import gdal, osr, gdal_array,ogr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import warnings
warnings.filterwarnings('ignore')

class ImagePaser:
    def __init__(self, sigma0_datadir, C2_datadir, NDVIDir, angledir, fn_zone, outpath):
        '''
        sigma0_datadir: Backscatter coefficient file path
        C2_datadir: C2 matrix images file path
        NDVIDir: NDVI images file path
        angledir: Incident angle images file path
        fn_zone(.shp): Sub file grid vector file
        outpath: the path to the output file folder
        '''

        # Initiate the path to the image 
        self.sigma0_datadir = sigma0_datadir
        self.C2_datadir = C2_datadir
        self.NDVIDir = NDVIDir
        self.angledir = angledir
        self.fn_zone = fn_zone 

        # Initiate the path to the output file folder
        self.outpath = outpath
        self.fn_csv_SG = self.outpath+'SARpara_sigma0&angle.csv'
        self.fn_csv_C2 = self.outpath+'SARpara_C2.csv'
        self.fn_csv_NDVI = self.outpath+'NDVI.csv'

    def boundingBoxToOffsets(bbox, geot):
        col1 = int((bbox[0] - geot[0]) / geot[1])
        col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
        row1 = int((bbox[3] - geot[3]) / geot[5])
        row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
        return [row1, row2, col1, col2]

    def geotFromOffsets(row_offset, col_offset, geot):
        new_geot = [
        geot[0] + (col_offset * geot[1]),
        geot[1],
        0.0,
        geot[3] + (row_offset * geot[5]),
        0.0,
        geot[5]
        ]
        return new_geot

    # Process backscatter coefficient images and incident angle images to generate statistical results csv
    def processingBackscatterCoefficientAndAngleImages(self):
        os.chdir(self.sigma0_datadir[:-1])
        files = os.listdir()
        tifList_VV = list(filter(lambda file: file.endswith('.tif') and 'VV' in file, files))
        tifList_VH = list(filter(lambda file: file.endswith('.tif') and 'VH' in file, files))
        bandnames1 = [i[tifList_VV[0].find('20'):tifList_VV[0].find('20')+8] for i in tifList_VV]
        bandnames2 = [i[tifList_VH[0].find('20'):tifList_VH[0].find('20')+8] for i in tifList_VH]
        bandnamesindex = [i for i in range(len(bandnames1)) if (int(bandnames1[i])>20190501) and (int(bandnames1[i])<20191130)]
        tifList_VV = [self.sigma0_datadir+i for i in tifList_VV]
        tifList_VH = [self.sigma0_datadir+i for i in tifList_VH]
        if len(tifList_VV) == 0 or len(tifList_VV) == 0:
            print('Can only process tif files!')
    
        os.chdir(self.angledir[:-1])
        files = os.listdir()
        tifList_IA = list(filter(lambda file: file.endswith('.tif') and 'incidenceAngle' in file, files))
        tifList_LIA = list(filter(lambda file: file.endswith('.tif') and 'LocalIncidenceAngle' in file, files))
        bandnames3 = [i[tifList_IA[0].find('20'):tifList_IA[0].find('20')+8] for i in tifList_IA]
        bandnames4 = [i[tifList_LIA[0].find('20'):tifList_LIA[0].find('20')+8] for i in tifList_LIA]
        tifList_IA = [self.angledir+i for i in tifList_IA]
        tifList_LIA = [self.angledir+i for i in tifList_LIA]
        if (len(tifList_IA) == 0) or (len(tifList_LIA) == 0):
            print('Can only process tif files!')

        ds = gdal.Open(tifList_VV[0])
        bandNum = len(tifList_VV)
        Xsize = ds.RasterXSize
        Ysize = ds.RasterYSize
        gt = gdal.Dataset.GetGeoTransform(ds)

        bandnames1 = np.array(bandnames1)
        bandnamesindex = np.array(bandnamesindex)
        pbar = tqdm(bandnames1[bandnamesindex], colour='yellow')
        zstats = []
        ParaName = ['VV', 'VH', 'DPSVI', 'CR', 'NRPB', 'RVI', 'Span']

        bandnames1 = np.array(bandnames1)

        for date in pbar:
            bn = date
            bandnames1 = list(bandnames1)
            b1 = bandnames1.index(bn)
            b2 = bandnames2.index(bn)
            bandnames1 = np.array(bandnames1)
            try:
                b3 = bandnames3.index(bn)
                b4 = bandnames4.index(bn)
            except:
                continue
            
            ds_VV = gdal.Open(tifList_VV[b1])
            ds_VH = gdal.Open(tifList_VH[b2])
            ds_IA = gdal.Open(tifList_IA[b3])
            ds_LIA = gdal.Open(tifList_LIA[b4])
            
            array_VV = ds_VV.GetRasterBand(1).ReadAsArray()
            array_VH = ds_VH.GetRasterBand(1).ReadAsArray()
            array_IA = ds_IA.GetRasterBand(1).ReadAsArray()
            array_LIA =  ds_LIA.GetRasterBand(1).ReadAsArray()
            array_VV = array_VV*np.cos(37.5/180*np.pi)**2/np.cos(array_IA/180*np.pi)**2
            array_VH = array_VH*np.cos(37.5/180*np.pi)**2/np.cos(array_IA/180*np.pi)**2
            
            array_DPSVI = (array_VV+array_VH)/array_VV
            array_CR = array_VH/array_VV
            array_NRPB = (array_VH-array_VV)/(array_VH+array_VV)
            array_RVI = 4*array_VH/(array_VV+array_VH)
            array_Span = array_VV+array_VH
            array_VV = array_VV
            array_VH = array_VH
            
            arrayList = [array_VV, array_VH, array_DPSVI, array_CR ,array_NRPB, array_RVI, array_Span]

            print('\nProcess '+date+'...')
            p_ds = ogr.Open(self.fn_zones)
            mem_driver = ogr.GetDriverByName("Memory")
            mem_driver_gdal = gdal.GetDriverByName("MEM")
            shp_name = "temp"
            lyr = p_ds.GetLayer()
            geot = gt
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            p_feat = lyr.GetNextFeature()
            count = 0
            
            while p_feat:
                count = count+1
                if count!=1:
                    p_feat = lyr.GetNextFeature()
                if not p_feat:
                    continue
                if count%1000==0:
                    print(str(count)+'/'+str(len(lyr)))
                if p_feat.GetGeometryRef() is not None:
                    if os.path.exists(shp_name):
                        mem_driver.DeleteDataSource(shp_name)
                    tp_ds = mem_driver.CreateDataSource(shp_name)
                    tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
                    tp_lyr.CreateFeature(p_feat.Clone())
                    offsets = self.boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
                    geot)
                    new_geot = self.geotFromOffsets(offsets[0], offsets[2], geot)
            
                    tr_ds = mem_driver_gdal.Create(\
                    "", \
                    offsets[3] - offsets[2], \
                    offsets[1] - offsets[0], \
                    1, \
                    gdal.GDT_Byte)
            
                    tr_ds.SetGeoTransform(new_geot)
                    gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
                    tr_array = tr_ds.ReadAsArray()
            
                    r_array = list([0]*len(ParaName))
                    for paracount in range(len(ParaName)):
                        r_array[paracount] = arrayList[paracount][offsets[0]:offsets[1], offsets[2]:offsets[3]]
            
                    ID = p_feat.GetFID()
            
                    if r_array is not None:
                        maskarray = list([0]*len(ParaName))
                        try:
                            for paracount in range(len(ParaName)):
                                maskarray[paracount] = np.ma.masked_array(\
                                r_array[paracount],\
                                mask=np.logical_or(r_array[paracount]==nodata, np.logical_not(tr_array)))
                            
                        except:
                            continue
                        
                        for paracount in range(len(ParaName)): 
                            if maskarray[paracount] is not None:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], date, p_feat['lon'], p_feat['lat'], 'mean', ParaName[paracount], maskarray[paracount].mean(), maskarray[paracount].std()])
                            else:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', ParaName[paracount], None, None])
                            
                    else:
                        for paracount in range(len(ParaName)): 
                            zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', ParaName[paracount], None, None])
                    tp_ds = None
                    tp_lyr = None
                    tr_ds = None
                
            col_names = ['ID', 'Sample_ID', 'Crop_Type', 'Area',  'Date', 'lon', 'lat', 'Stastic_Type', 'Parameter_Type', 'Value', 'std']
            out_csv = pd.DataFrame(zstats, columns = col_names)
            out_csv.to_csv(self.fn_csv_SG)

    # Process C2 matrix images to generate statistical results csv
    def processingC2MatrixImages(self):
        os.chdir(self.C2_datadir[:-1])
        files = os.listdir()
        tifList_C11 = list(filter(lambda file: file.endswith('.tif') and 'C11' in file, files))
        tifList_C12i = list(filter(lambda file: file.endswith('.tif') and 'C12_imag' in file, files))
        tifList_C12r = list(filter(lambda file: file.endswith('.tif') and 'C12_real' in file, files))
        tifList_C22 = list(filter(lambda file: file.endswith('.tif') and 'C22' in file, files))
        bandnames1 = [i[tifList_C12i[0].find('20'):tifList_C12i[0].find('20')+8] for i in tifList_C12i]
        bandnamesindex = [i for i in range(len(bandnames1)) if (int(bandnames1[i])>20190501) and (int(bandnames1[i])<20191130)]
        tifList_C11 = [self.C2_datadir+i for i in tifList_C11]
        tifList_C12i = [self.C2_datadir+i for i in tifList_C12i]
        tifList_C12r = [self.C2_datadir+i for i in tifList_C12r]
        tifList_C22 = [self.C2_datadir+i for i in tifList_C22]
        if len(tifList_C11) == 0 or len(tifList_C12i) == 0 or (tifList_C12r == 0) or (tifList_C22 == 0):
            print('Can only process tif files!')
    
        ds = gdal.Open(tifList_C11[0])
        bandNum = len(tifList_C11)
        Xsize = ds.RasterXSize
        Ysize = ds.RasterYSize
        gt = gdal.Dataset.GetGeoTransform(ds)

        ParaName = ['C11', 'C12i', 'C12r', 'C22','TrC2', 'detC2', 'm', 'beta', 'DpRVI']

        bandnames1 = np.array(bandnames1)
        bandnamesindex = np.array(bandnamesindex)
        pbar = tqdm(bandnames1[bandnamesindex], colour='yellow')
        count = 0
        zstats = []

        for date in pbar:
            bn = date
            bandnames1 = list(bandnames1)
            b1 = bandnames1.index(bn)
            b2 = bandnames1.index(bn)
            b3 = bandnames1.index(bn)
            b4 = bandnames1.index(bn)
            ds_C11= gdal.Open(tifList_C11[b1])
            ds_C12i = gdal.Open(tifList_C12i[b2])
            ds_C12r = gdal.Open(tifList_C12r[b3])
            ds_C22 = gdal.Open(tifList_C22[b4])

            array_C11 = ds_C11.GetRasterBand(1).ReadAsArray()
            array_C12i = ds_C12i.GetRasterBand(1).ReadAsArray()
            array_C12r = ds_C12r.GetRasterBand(1).ReadAsArray()
            array_C22 = ds_C22.GetRasterBand(1).ReadAsArray()

            array_C11 = array_C11
            array_C12i = array_C12i
            array_C12r = array_C12r
            array_C22 = array_C22
            
            array_TrC2 = array_C11+array_C22
            array_detC2 = array_C11*array_C22-array_C12r**2-array_C12i**2
            array_m = np.sqrt(1-4*array_detC2/(array_TrC2**2))
            array_beta = (1+array_m)/2
            array_DpRVI = 1-array_m*array_beta
            array_beta = array_beta
            array_m = array_m
            
            arrayList = [array_C11, array_C12i, array_C12r, array_C22, array_TrC2, array_detC2, array_m, array_beta, array_DpRVI]

            print('\nProcess '+date+'...')
            p_ds = ogr.Open(self.fn_zones)
            mem_driver = ogr.GetDriverByName("Memory")
            mem_driver_gdal = gdal.GetDriverByName("MEM")
            shp_name = "temp"
            lyr = p_ds.GetLayer()
            geot = gt
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            p_feat = lyr.GetNextFeature()
            count = 0
            
            while p_feat:
                count = count+1
                if count!=1:
                    p_feat = lyr.GetNextFeature()
                if not p_feat:
                    continue
                if count%1000==0:
                    print(str(count)+'/'+str(len(lyr)))
                if p_feat.GetGeometryRef() is not None:
                    if os.path.exists(shp_name):
                        mem_driver.DeleteDataSource(shp_name)
                    tp_ds = mem_driver.CreateDataSource(shp_name)
                    tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
                    tp_lyr.CreateFeature(p_feat.Clone())
                    offsets = self.boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
                    geot)
                    new_geot = self.geotFromOffsets(offsets[0], offsets[2], geot)
            
                    tr_ds = mem_driver_gdal.Create(\
                    "", \
                    offsets[3] - offsets[2], \
                    offsets[1] - offsets[0], \
                    1, \
                    gdal.GDT_Byte)
            
                    tr_ds.SetGeoTransform(new_geot)
                    gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
                    tr_array = tr_ds.ReadAsArray()
            
                    r_array = list([0]*len(ParaName))
                    for paracount in range(len(ParaName)):
                        r_array[paracount] = arrayList[paracount][offsets[0]:offsets[1], offsets[2]:offsets[3]]
            
                    ID = p_feat.GetFID()
            
                    if r_array is not None:
                        maskarray = list([0]*len(ParaName))
                        try:
                            for paracount in range(len(ParaName)):
                                maskarray[paracount] = np.ma.masked_array(\
                                r_array[paracount],\
                                mask=np.logical_or(r_array[paracount]==nodata, np.logical_not(tr_array)))
                            
                        except:
                            continue
                        
                        for paracount in range(len(ParaName)): 
                            if maskarray[paracount] is not None:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], date, p_feat['lon'], p_feat['lat'], 'mean', ParaName[paracount], maskarray[paracount].mean(), maskarray[paracount].std()])
                            else:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', ParaName[paracount], None, None])
                            
                    else:
                        for paracount in range(len(ParaName)): 
                            zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', ParaName[paracount], None, None])
                    tp_ds = None
                    tp_lyr = None
                    tr_ds = None
            col_names = ['ID', 'Sample_ID', 'Crop_Type', 'Area',  'Date', 'lon', 'lat', 'Stastic_Type', 'Parameter_Type', 'Value', 'std']
            out_csv = pd.DataFrame(zstats, columns = col_names)
            out_csv.to_csv(self.fn_csv_C2)

    # Process NDVI images to generate statistical results csv
    def processingNDVIImages(self):
        os.chdir(self.NDVIDir[:-1])
        files = os.listdir()
        tifList_NDVI = list(filter(lambda file: file.endswith('.tif') and 'ndvi' in file, files))
        bandnames_NDVI = [i[tifList_NDVI[0].find('20'):tifList_NDVI[0].find('20')+8] for i in tifList_NDVI]
        tifList_NDVI = [NDVIDir+i for i in tifList_NDVI]
        if (len(tifList_NDVI) == 0):
            print('Can only process tif files!')

        ds = gdal.Open(tifList_NDVI[0])
        bandNum_NDVI = len(tifList_NDVI)
        Xsize = ds.RasterXSize
        Ysize = ds.RasterYSize
        NDVI_array = np.zeros(bandNum_NDVI*Xsize*Ysize).reshape(bandNum_NDVI, Ysize, Xsize)
            
        pbar = tqdm(range(bandNum_NDVI), colour='green')
        for b in pbar:
            ds_NDVI = gdal.Open(tifList_NDVI[b])
            array_NDVI = ds_NDVI.GetRasterBand(1).ReadAsArray()
            NDVI_array[b,:,:] = array_NDVI

        def boundingBoxToOffsets(bbox, geot):
            col1 = int((bbox[0] - geot[0]) / geot[1])
            col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
            row1 = int((bbox[3] - geot[3]) / geot[5])
            row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
            return [row1, row2, col1, col2]

        def geotFromOffsets(row_offset, col_offset, geot):
            new_geot = [
            geot[0] + (col_offset * geot[1]),
            geot[1],
            0.0,
            geot[3] + (row_offset * geot[5]),
            0.0,
            geot[5]
            ]
            return new_geot

        zstats = []
        arrayList = [NDVI_array]
        ParaName = ['NDVI']
        gt = gdal.Dataset.GetGeoTransform(gdal.Open(tifList_NDVI[0]))
        pbar = tqdm(zip(ParaName, arrayList), colour='green')
        for Para, r_array_ in pbar:
            bandCount = 0
            for date in bandnames_NDVI:
                p_ds = ogr.Open(self.fn_zones)
                mem_driver = ogr.GetDriverByName("Memory")
                mem_driver_gdal = gdal.GetDriverByName("MEM")
                shp_name = "temp"
                lyr = p_ds.GetLayer()
                geot = gt
                nodata = ds.GetRasterBand(1).GetNoDataValue()
                p_feat = lyr.GetNextFeature()
                niter = 0
                count = 0
                
                while p_feat:
                    count = count+1
                    if count!=1:
                        p_feat = lyr.GetNextFeature()
                    if not p_feat:
                        continue
                    if count%1000==0:
                        print(str(count)+'/'+str(len(lyr)))
                    if p_feat.GetGeometryRef() is not None:
                        if os.path.exists(shp_name):
                            mem_driver.DeleteDataSource(shp_name)
                        tp_ds = mem_driver.CreateDataSource(shp_name)
                        tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
                        tp_lyr.CreateFeature(p_feat.Clone())
                        offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
                        geot)
                        new_geot = geotFromOffsets(offsets[0], offsets[2], geot)
                
                        tr_ds = mem_driver_gdal.Create(\
                        "", \
                        offsets[3] - offsets[2], \
                        offsets[1] - offsets[0], \
                        1, \
                        gdal.GDT_Byte)
                
                        tr_ds.SetGeoTransform(new_geot)
                        gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
                        tr_array = tr_ds.ReadAsArray()
                
                        r_array = r_array_[bandCount ,offsets[0]:offsets[1], offsets[2]:offsets[3]]
                
                        ID = p_feat.GetFID()
                
                        if r_array is not None:
                            try:
                                maskarray = np.ma.masked_array(\
                                r_array,\
                                mask=np.logical_or(r_array==nodata, np.logical_not(tr_array)))
                            except:
                                continue
                            
                            if maskarray is not None:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], date, p_feat['lon'], p_feat['lat'], 'mean', Para, maskarray.mean(), maskarray.std()])
                            else:
                                zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', Para, None, None])
                        else:
                            zstats.append([p_feat['ID'], p_feat['Sample_ID'], p_feat['Crop_Type'], p_feat['Area'], p_feat['lon'], p_feat['lat'], date, 'mean', Para, None, None])
                
                        tp_ds = None
                        tp_lyr = None
                        tr_ds = None
                bandCount = bandCount+1
        col_names = ['ID', 'Sample_ID', 'Crop_Type', 'Area',  'Date', 'lon', 'lat', 'Stastic_Type', 'Parameter_Type', 'Value', 'std']
        out_csv = pd.DataFrame(zstats, columns = col_names)
        out_csv.to_csv(self.fn_csv_NDVI)

if __name__ == "__main__":
    # Enter the path to the image file
    sigma0_datadir = '/' # Backscatter coefficient file path
    C2_datadir = '/' # C2 matrix images file path
    NDVIDir = '/' # NDVI images file path
    angledir = '/' # Incident angle images file path
    fn_zone = '.shp' # Sub file grid vector file

    # Enter the path to the output file folder
    outpath = '/' 

    ip = ImagePaser(sigma0_datadir, C2_datadir, NDVIDir, angledir, fn_zone, outpath)
    print('Processing backscatter coefficient and incident angle images...')
    ip.processingBackscatterCoefficientAndAngleImages()
    print('Processing C2 matrix images...')
    ip.processingC2MatrixImages()
    print('Processing NDVI images...')
    ip.processingNDVIImages()
