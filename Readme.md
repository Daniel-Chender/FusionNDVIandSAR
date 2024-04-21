# Code for Journal of Agricultural and Forest Metrology Paper
This repository contains the code for the paper "Crop NDVI time series construction by fusing Sentinel-1, Sentinel-2, and environmental data with an ensemble-based framework" submitted to the Computers and Electronics in Agriculture. This paper proposes an ensemble-based data fusion framework that combines SAR and auxiliary environmental data including temperature, soil moisture, evaporation and precipitation to produce high-precision and dense NDVI time series for corn and soybean. Suggestions for model input parameters were provided based on an improved importance ranking method of RF model. The framework also constructs dense and precise NDVI time series using the weighted least square model, which assigns weights based on an ensemble-based uncertainty quantification method. The proposed method performed well in filling data gaps in both the vegetative and reproductive stages of crops for both crops.

## Data Processing Code
The data processing code includes two parts: ***ImageParser.py*** and ***DataPreprocess.py***.

- ***ImageParser.py:*** This script partitions and records the information of remote sensing images (including SRA and NDVI images) into CSV files. The input is a folder containing the remote sensing images in GeoTIFF format. The output is a CSV file containing the id, the date, statistical pixel value, crop type, latitude, and longitude for each image. The script also generates a plot of the spatial distribution of the images.

- ***DataPreprocess.py*** This script integrates, matches, and cleans the statistical data. The input is a CSV file generated by ***ImageParser.py*** and the analysis data from ERA5-Land. The output is a CSV file containing the cleaned and matched data for each location-date pair. The script also calculates the accumulated temperature data from the temperature data.

## Method Code
The method code consists of three parts: ***RandomForestModel.py***, ***ImprovedMDA.py***, and ***NDVITimeSeriesConstruction.py***.

- ***RandomForestModel.py***: This script trains, evaluates, and visualizes a random forest model for constructing the NDVI-SAR relationship model using the processed data. The input is a CSV file containing the cleaned and matched data for each location-date pair. The output is a CSV file containing the predicted NDVI and the evaluation metrics (RMSE, MAE, R2) for each location-date pair. The script also generates plots of the predicted vs. observed NDVI, the distribution of the prediction error.

- ***ImprovedMDA.py***: This script implements the improved feature importance ranking method based on random forest MDA proposed in the paper. The input is a CSV file containing the predicted NDVI for each location-date pair. The output is a CSV file containing the improved MDA score for each feature.

- ***NDVITimeSeriesConstruction.py***: This script constructs and visualizes the NDVI dense time series for each location using the predicted and observed NDVI recoded in ***RandomForestModel.py***. The input is a CSV file containing the predicted NDVI for each location-date pair. The output is a CSV file containing the NDVI value for each location-date pair. The script also generates plots of the NDVI time series and the correlation between ground true NDVI time series and defect NDVI time series filled by SAR and environmental data.

## How to Run
- To run the code, you need to have Python 3.9 installed on your system. You also need to install some Python packages, such as pandas, numpy, sklearn, matplotlib, gdal, etc. You can use pip or conda to install them. you can enter the following script in your terminal to ensure that you have the correct package version in your environment.
```python
pip install - r requirements. txt
```
- To run the data processing code, you need to have a folder containing the remote sensing images in GeoTIFF format and a folder for containing the CSV statistical data. You need to modify the paths and parameters in ImageParser.py and DataPreprocess.py according to your data.

- To run the method code, you need to have the output files generated by the data processing code and modify the paths and parameters in RandomForestModel.py, ImprovedMDA.py, and NDVITimeSeriesConstruction.py according to your data.

You can run each script by typing python script_name.py in your terminal or using an IDE such as PyCharm or VSCode.

## License
This code is licensed under MIT License. You can use it for any non commercial purpose, but please cite our paper if you use it for academic research.

## Contact
If you have any questions or suggestions about our code or paper, please feel free to contact us at chendr7@mail2.sysu.edu.cn. We appreciate your feedback!

## Citation
If you find this paper useful in your research, please consider citing:
```
@article{chen2023crop,
  title={Crop NDVI time series construction by fusing Sentinel-1, Sentinel-2, and environmental data with an ensemble-based framework},
  author={Chen, Dairong and Hu, Haoxuan and Liao, Chunhua and Ye, Junyan and Bao, Wenhao and Mo, Jinglin and Wu, Yue and Dong, Taifeng and Fan, Hong and Pei, Jie},
  journal={Computers and Electronics in Agriculture},
  volume={215},
  pages={108388},
  year={2023},
  publisher={Elsevier}
}
```
