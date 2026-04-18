# coding:utf-8
"""
@file: misc_function.py
@author: Zeping Liu
@ide: PyCharm
@createTime: 2024.07
@contactInformation: zeping.liu@utexas.edu
@Function: misc functions
"""
import os
import pickle
import shutil
from shapely.geometry import Point
import geopandas as gpd
import rasterio
import pandas as pd
import json

def load_pickle(file):
    '''

    Args:
        file: the path of pickle file

    Returns: dataframe

    '''
    try:
        with open(file, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"find error: {e}")
        return None


def arrange_gsv_path(gsv_path="/data/lzp/gsv_rs_project/temp/media/data_16T/huangyj/Treepedia/Data/GSV/LosAngelesGSV",
                     pickle_path="/data/lzp/gsv_rs_project/temp/media/data_16T/huangyj/Treepedia/Data/GSV/LosAngelesGSV/pano_2023-06-29 13:49:19.256586_224460.p"):
    panoid_file = load_pickle(pickle_path)
    panoids = list(panoid_file.panoid)
    points = []
    df = {"lat": [], "lon": []}
    for panoid in panoids:
        for angle in [0, 90, 180, 270]:
            p = f'{gsv_path}/{panoid[-2]}/{panoid[-1]}/{panoid}/{panoid}_{angle}.jpg'
            if os.path.exists(p):
                os.makedirs(rf"/data/lzp/gsv_rs_project/LosAngelesGSV/{panoid}", exist_ok=True)
                shutil.copy(p, rf"/data/lzp/gsv_rs_project/LosAngelesGSV/{panoid}/{panoid}_{angle}.jpg")


def assign_meta_information_by_panoid(panoid, meta_data, path):
    """
    This code should be modified to fit different data structure
    It is anticipated to create a json file containing the meta-data of each street view imagery

    """
    # This is for data provided by Fan Zhang
    indexed_meta_data = meta_data[meta_data['panoid']==panoid]
    lon, lat = indexed_meta_data['lon'].values[0], indexed_meta_data['lat'].values[0]
    year, month = indexed_meta_data['year'].values[0], indexed_meta_data['month'].values[0]
    data = {
        'lon': float(lon),
        'lat': float(lat),
        'year': int(year),
        'month': int(month),
    }
    json_filename = os.path.join(path, f'meta_data.json')
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Saved {json_filename}")
def convert_to_geopandas(df, lat_col, lon_col):
    '''

    Args:
        df: dataframe from panoids
        lat_col: the col name of lat
        lon_col: the col name pf lon

    Returns: geo dataframe version of the input dataframe

    '''
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    return geo_df


# def combine_sv_imagery(path="/data/lzp/gsv_rs_project/bostonGSV"):
#     all_paths = os.listdir(path)
#     for sub_path in all_paths:
#         sub_df = os.listdir(os.path.join(path, sub_path))
#         if len(sub_df) ==4:


if __name__ == '__main__':
    # tag = os.listdir(r"/data/lzp/gsv_rs_project/bostonGSV")
    # panoid_file = load_pickle(
    #     "/data/lzp/gsv_rs_project/media/data_16T/sanatani/Treepedia/scripts/gsv_download_Yuhao/BostonGSV/pano_2023-02-11 18:17:03.043108_120900.p")
    # filtered_df = panoid_file[panoid_file['panoid'].isin(tag)]
    # geo_df = convert_to_geopandas(filtered_df, "lat", "lon")
    # geo_df.set_crs(epsg=4326, inplace=True)
    #
    # output_path = "/home/zeping/lzp/code/SatMAE-main/bostonGSV.shp"
    # geo_df.to_file(output_path, driver='ESRI Shapefile')

    from tqdm import tqdm
    # arrange_gsv_path()
    path_1 = "/data/sphere2vec/lzp_data/gsv_rs_project/losangelesGSV"
    files = os.listdir(path_1)
    meta_data = pd.read_csv("/data/sphere2vec/lzp_data/sv_information.csv")
    for panoid in tqdm(files):
        path_2 = os.path.join(path_1, panoid)
        assign_meta_information_by_panoid(panoid, meta_data, path_2)
