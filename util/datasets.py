import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob
from typing import Any, Optional, List
import time

import skimage
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio import logging
import ast
import json
import pandas as pd
from util import misc
from skimage.transform import resize

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


def get_image_center_coordinates(image_path):
    # 打开影像
    with rasterio.open(image_path) as src:
        # 获取影像的仿射变换和坐标参考系统（CRS）
        transform = src.transform
        crs = src.crs

        # 获取影像的高度和宽度
        width = src.width
        height = src.height

        # 计算中心点的像素坐标
        center_x = width // 2
        center_y = height // 2

        # 使用仿射变换将像素坐标转换为地理坐标（经纬度）
        lon, lat = transform * (center_x, center_y)

        bounds = src.bounds

        lon_min, lat_max = bounds.left, bounds.top
        lon_max, lat_min = bounds.right, bounds.bottom

    return lat, lon, transform, lon_min, lat_max, lon_max, lat_min


class RS_Fmow_dataset(Dataset):
    rs_mean = [1532.2765542273994, 1320.6235723591594, 1245.7969259679662, 1255.6873570527202, 1451.4597726588897,
               2020.938144097032,
               2287.9023660494504, 2207.3614826004746, 2463.119533828435, 770.9425868838862, 18.239819377135618,
               2019.450240972937, 1426.6151914783677]
    rs_std = [187.31048009349252, 281.3815287179912, 309.0530739473631, 405.8805281334264, 355.24149685219277,
              445.42465976929725, 522.6256098757555,
              564.0999863239944, 569.4189759819108, 136.83656392446318, 3.088879407558455, 466.9946577787395,
              405.8510544322312]

    def __init__(self, file_path, main_path, transform, benchmark):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.main_path = main_path
        self.transform = transform
        self.benchmark = benchmark
        filename = os.path.basename(file_path)
        self.keep_bands = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
        # 检查文件名是否包含 "train" 或 "val"
        if "train" in filename:
            self.sub_folder = "train"
        elif "val" in filename:
            self.sub_folder = "val"
        else:
            self.sub_folder = "test_gt"

        self.labels_id = {'prison': 0,
                          'parking_lot_or_garage': 1,
                          'debris_or_rubble': 2,
                          'barn': 3,
                          'nuclear_powerplant': 4,
                          'stadium': 5,
                          'road_bridge': 6,
                          'crop_field': 7,
                          'educational_institution': 8,
                          'construction_site': 9,
                          'ground_transportation_station': 10,
                          'airport_hangar': 11,
                          'police_station': 12,
                          'toll_booth': 13,
                          'office_building': 14,
                          'runway': 15,
                          'surface_mine': 16,
                          'water_treatment_facility': 17,
                          'helipad': 18,
                          'waste_disposal': 19,
                          'tunnel_opening': 20,
                          'military_facility': 21,
                          'swimming_pool': 22,
                          'lighthouse': 23,
                          'solar_farm': 24,
                          'place_of_worship': 25,
                          'park': 26,
                          'flooded_road': 27,
                          'impoverished_settlement': 28,
                          'fire_station': 29,
                          'lake_or_pond': 30,
                          'oil_or_gas_facility': 31,
                          'wind_farm': 32,
                          'golf_course': 33,
                          'port': 34,
                          'shopping_mall': 35,
                          'recreational_facility': 36,
                          'multi-unit_residential': 37,
                          'interchange': 38,
                          'airport_terminal': 39,
                          'electric_substation': 40,
                          'dam': 41,
                          'car_dealership': 42,
                          'airport': 43,
                          'hospital': 44,
                          'shipyard': 45,
                          'aquaculture': 46,
                          'tower': 47,
                          'archaeological_site': 48,
                          'gas_station': 49,
                          'fountain': 50,
                          'border_checkpoint': 51,
                          'factory_or_powerplant': 52,
                          'single-unit_residential': 53,
                          'storage_tank': 54,
                          'railway_bridge': 55,
                          'space_facility': 56,
                          'amusement_park': 57,
                          'race_track': 58,
                          'smokestack': 59,
                          'burial_site': 60,
                          'zoo': 61}

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        category = self.database.iloc[index].category
        location_id = self.database.iloc[index].location_id
        image_id = self.database.iloc[index].image_id

        image_path = os.path.join(self.main_path, self.sub_folder, category, f"{category}_{location_id}",
                                  f"{category}_{location_id}_{image_id}.tif")

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)
        lat, lon, _, _, _, _, _ = get_image_center_coordinates(image_path)

        # print(image_as_tensor.shape)
        if self.benchmark == "spectral_gpt":
            image_as_tensor = image_as_tensor[:-1,:,:] # for spectral gpt, it needs to keep 12 bands
        else:
            image_as_tensor = image_as_tensor[self.keep_bands, :, :]

        if self.benchmark == "imagenet" or self.benchmark == "street_view_imagenet":
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 120, 120)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "spectral_gpt":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 128, 128)
        else:
            pass
        return {"image": image_as_tensor,
                'values': self.labels_id[category],
                "lat": lat,
                "lon": lon}

    def __len__(self):
        return len(self.database)


class EuroSatDataset(Dataset):
    rs_means = [
        1353.7269, 1117.2023, 1041.8847, 946.5543, 1199.1887,
        2003.0068, 2374.0084, 2301.2204, 732.1820, 12.0995,
        1820.6964, 1118.2027, 2599.7829
    ]

    rs_stds = [
        65.2886, 153.7550, 187.6764, 278.0907, 227.8963,
        355.8897, 455.0773, 530.7148, 98.9179, 1.1872,
        378.1152, 303.0695, 502.1025
    ]

    def __init__(self, file_path, transform, benchmark):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.keep_bands = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
        # 检查文件名是否包含 "train" 或 "val"

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        category_id = self.database.iloc[index].category_id
        lon = self.database.iloc[index].lon
        lat = self.database.iloc[index].lat

        image_path = self.database.iloc[index].image_path

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        # print(image_as_tensor.shape)
        image_as_tensor = image_as_tensor[self.keep_bands, :, :]
        if self.benchmark == "imagenet" or self.benchmark == "street_view_imagenet":
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 120, 120)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass
        return {"image": image_as_tensor,
                'values': torch.tensor(category_id),
                "lat": lat,
                "lon": lon}

    def __len__(self):
        return len(self.database)


class WorldStratDataset(Dataset): # IPCC
    # Mean 列表
    rs_means = [
        0.1254, 0.1361, 0.1613, 0.1759, 0.2115,
        0.2678, 0.2906, 0.2978, 0.3059, 0.3350,
        0.2669, 0.2081
    ]

    # Std 列表
    rs_stds = [
        0.0462, 0.0570, 0.0588, 0.0667, 0.0643,
        0.0654, 0.0689, 0.0747, 0.0706, 0.0790,
        0.0641, 0.0618
    ]

    def __init__(self, file_path, transform, benchmark):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.keep_bands = torch.tensor([0,1, 2, 3, 4, 5, 6, 7, 8, 11])
        self.category = {
            'Forest': 0,
            'Agriculture': 1,
            'Settlement': 2,
            'Water': 3,
            'Grassland': 4,
            'Other': 5,
            'Wetland': 6,
            '0': 7
        }

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        category_name = self.database.iloc[index].IPCC
        category_id = self.category[category_name]
        lon = self.database.iloc[index].lon
        lat = self.database.iloc[index].lat

        image_path = self.database.iloc[index].image_path

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        # print(image_as_tensor.shape)
        if self.benchmark == "spectral_gpt":
            image_as_tensor = image_as_tensor # for spectral gpt, it needs to keep 12 bands
        else:
            image_as_tensor = image_as_tensor[self.keep_bands, :, :]

        if self.benchmark == "imagenet" or self.benchmark == "street_view_imagenet":
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 120, 120)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "spectral_gpt":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 128, 128)
        else:
            pass
        return {"image": image_as_tensor,
                'values': torch.tensor(category_id),
                "lat": lat,
                "lon": lon}

    def __len__(self):
        return len(self.database)

class WorldStratDataset_lccs(Dataset): # LCCS
    # Mean 列表
    rs_means = [
        0.1254, 0.1361, 0.1613, 0.1759, 0.2115,
        0.2678, 0.2906, 0.2978, 0.3059, 0.3350,
        0.2669, 0.2081
    ]

    # Std 列表
    rs_stds = [
        0.0462, 0.0570, 0.0588, 0.0667, 0.0643,
        0.0654, 0.0689, 0.0747, 0.0706, 0.0790,
        0.0641, 0.0618
    ]

    def __init__(self, file_path, transform, benchmark):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.keep_bands = torch.tensor([0,1, 2, 3, 4, 5, 6, 7, 8, 11])
        self.category = { # 34 categories
            {50: 0,
             160: 1,
             11: 2,
             40: 3,
             10: 4,
             190: 5,
             20: 6,
             210: 7,
             200: 8,
             130: 9,
             110: 10,
             120: 11,
             180: 12,
             70: 13,
             30: 14,
             100: 15,
             122: 16,
             152: 17,
             60: 18,
             150: 19,
             12: 20,
             220: 21,
             90: 22,
             201: 23,
             62: 24,
             140: 25,
             80: 26,
             61: 27,
             170: 28,
             202: 29,
             71: 30,
             121: 31,
             81: 32,
             153: 33}
        }

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        category_name = self.database.iloc[index].IPCC
        category_id = self.category[category_name]
        lon = self.database.iloc[index].lon
        lat = self.database.iloc[index].lat

        image_path = self.database.iloc[index].image_path

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        # print(image_as_tensor.shape)
        if self.benchmark == "spectral_gpt":
            image_as_tensor = image_as_tensor # for spectral gpt, it needs to keep 12 bands
        else:
            image_as_tensor = image_as_tensor[self.keep_bands, :, :]

        if self.benchmark == "imagenet" or self.benchmark == "street_view_imagenet":
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 120, 120)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "spectral_gpt":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 128, 128)
        else:
            pass
        return {"image": image_as_tensor,
                'values': torch.tensor(category_id),
                "lat": lat,
                "lon": lon}

    def __len__(self):
        return len(self.database)

class SatelliteDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path: str, transform: Any):
        """
        Creates Dataset for temporal RGB image classification. Stacks images along temporal dim.
        Usually used for fMoW-RGB-temporal dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion
        """
        super().__init__(in_c=9)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat((img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0)  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        self.transforms = transforms.Compose([
            # transforms.Scale(224),
            transforms.RandomCrop(224),
        ])
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Scale(224)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        suffix = single_image_name_1[-15:]
        prefix = single_image_name_1[:-15].rsplit('_', 1)
        regexp = '{}_*{}'.format(prefix[0], suffix)
        regexp = os.path.join(self.dataset_root_path, regexp)
        single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)
        temporal_files = glob(regexp)

        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_img_2 = Image.open(single_image_name_2)
        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and \
                    img_as_tensor_2.shape[2] > 224 and \
                    img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w],
                    img_as_tensor_2[..., :min_w],
                    img_as_tensor_3[..., :min_w]
                ], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and \
                    img_as_tensor_2.shape[1] > 224 and \
                    img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([
                    img_as_tensor_1[..., :min_w, :],
                    img_as_tensor_2[..., :min_w, :],
                    img_as_tensor_3[..., :min_w, :]
                ], dim=-3)
            else:
                img_as_img_1 = Image.open(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class ImageNormalizer:
    def __init__(self, mean, std):
        """
        Initialize the normalizer with mean and std for each channel.

        Parameters:
        - mean: list or array of mean values for each channel, e.g., [mean_r, mean_g, mean_b]
        - std: list or array of standard deviation values for each channel, e.g., [std_r, std_g, std_b]
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, *args, **kwargs):
        """
        Normalize an image with the initialized mean and std.

        Parameters:
        - img: numpy array of shape (h, w, 3)

        Returns:
        - Normalized numpy array of shape (h, w, 3)
        """
        img = img.astype(np.float32)
        normalized_img = (img - self.mean) / self.std
        return normalized_img


class SentinelStreetViewPairedImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    sentinel_mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
                     1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
                     1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    sentinel_std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
                    948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
                    1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    def __init__(self,
                 csv_path,
                 sv_images_path,
                 sat_images_path,
                 sat_transform,
                 sv_transform,
                 repeat=1,
                 len_sv=20,
                 dropped_bands=None
                 ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path)
        self.sv_images_path = sv_images_path
        self.sat_images_path = sat_images_path
        self.len_sv = len_sv
        self.in_c = self.in_c - len(dropped_bands)
        self.sat_transform = sat_transform
        self.sv_transform = sv_transform
        self.dropped_bands = dropped_bands
        self.repeat = repeat
        self.df = pd.concat([self.df] * self.repeat, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def process_time_stamp(self, year, month, hour, min_year=2002):

        return np.array([year - min_year, month - 1, hour])

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """

        selection = self.df.iloc[idx]
        panoids = ast.literal_eval(selection["panoid"])
        sv_coordinate = ast.literal_eval(selection["sv_coordinates"])
        patch_number = selection["patch"]
        city = selection["city"]

        combined_panoids_sv_coordinate = list(zip(panoids, sv_coordinate))
        sv_location_information = []
        sv_temporal_information = []
        if len(combined_panoids_sv_coordinate) < self.len_sv:
            sv_image_as_tensor_list = []
            for panoid in panoids:
                sv_all_files_in_folder = os.listdir(os.path.join(self.sv_images_path, city + "GSV", panoid))
                sv_path = os.path.join(self.sv_images_path, city + "GSV", panoid, random.choice(list(
                    filter(lambda x: x.endswith('.jpg'), sv_all_files_in_folder))))
                meta_file_path = os.path.join(self.sv_images_path, city + "GSV", panoid, "meta_data.json")
                with open(meta_file_path, 'r') as f:
                    meta_data = json.load(f)

                lon = meta_data['lon']
                lat = meta_data['lat']
                year = meta_data['year']
                month = meta_data['month']

                sv_location_information.append([float(lon), float(lat)])
                sv_temporal_information.append([int(year), int(month)])

                # sv_path = os.path.join(self.sv_images_path, city+"GSV", panoid, f"{panoid}_90.jpg")
                sv_image = skimage.io.imread(sv_path).astype(np.float32)  # H,W,C
                sv_image_as_tensor = self.sv_transform(sv_image)  # C,H,W
                sv_image_as_tensor_list.append(sv_image_as_tensor.unsqueeze(0))  # 1,C,H,W

            sv_image_as_tensor = torch.cat(sv_image_as_tensor_list, dim=0)  # n,C,H,W

            _, c, h, w = sv_image_as_tensor.shape  # N, C, H, W
            n = self.len_sv - len(combined_panoids_sv_coordinate)
            pad_sv_tensor = torch.zeros((n, c, h, w))

            sv_image_as_tensor = torch.cat([sv_image_as_tensor, pad_sv_tensor], dim=0)  # N+n, C,H,W
            sv_coordinate_as_tensor = torch.tensor(sv_coordinate)
            pad_sv_coordinate_tensor = torch.ones((n)) * -1.0
            sv_coordinate_as_tensor = torch.cat([sv_coordinate_as_tensor, pad_sv_coordinate_tensor], dim=0)

            sv_location_information_as_tensor = torch.tensor(sv_location_information)
            sv_temporal_information_as_tensor = torch.tensor(sv_temporal_information)
            pad_sv_location_information_as_tensor = torch.zeros((n, 2))
            pad_sv_temporal_information_as_tensor = torch.zeros((n, 2))

            sv_location_information_as_tensor = torch.cat(
                [sv_location_information_as_tensor, pad_sv_location_information_as_tensor], dim=0)
            sv_temporal_information_as_tensor = torch.cat(
                [sv_temporal_information_as_tensor, pad_sv_temporal_information_as_tensor], dim=0)

        else:
            sv_image_as_tensor_list = []
            combined_panoids_sv_coordinate_need = random.sample(combined_panoids_sv_coordinate, self.len_sv)
            panoids = [tup[0] for tup in combined_panoids_sv_coordinate_need]
            sv_coordinate = [tup[1] for tup in combined_panoids_sv_coordinate_need]

            for panoid in panoids:
                sv_all_files_in_folder = os.listdir(os.path.join(self.sv_images_path, city + "GSV", panoid))
                sv_path = os.path.join(self.sv_images_path, city + "GSV", panoid, random.choice(list(
                    filter(lambda x: x.endswith('.jpg'), sv_all_files_in_folder))))
                meta_file_path = os.path.join(self.sv_images_path, city + "GSV", panoid, "meta_data.json")
                with open(meta_file_path, 'r') as f:
                    meta_data = json.load(f)

                lon = meta_data['lon']
                lat = meta_data['lat']
                year = meta_data['year']
                month = meta_data['month']

                sv_location_information.append([float(lon), float(lat)])
                sv_temporal_information.append([int(year), int(month)])
                sv_image = skimage.io.imread(sv_path).astype(np.float32)  # H,W,C
                sv_image_as_tensor = self.sv_transform(sv_image)
                sv_image_as_tensor_list.append(sv_image_as_tensor.unsqueeze(0))

            sv_image_as_tensor = torch.cat(sv_image_as_tensor_list, dim=0)
            sv_coordinate_as_tensor = torch.tensor(sv_coordinate)
            sv_location_information_as_tensor = torch.tensor(sv_location_information)
            sv_temporal_information_as_tensor = torch.tensor(sv_temporal_information)

        time_paths = os.listdir(os.path.join(self.sat_images_path, f"{city}_clip"))
        while True:
            time_path = random.choice(time_paths)
            image_path = os.path.join(self.sat_images_path, f"{city}_clip", time_path, patch_number + ".tif")

            images = self.open_image(image_path)  # (h, w, c)
            if not np.all(images == 0):
                break
            else:
                time_paths.remove(time_path)
        # images, image_path = self.load_valid_image(time_paths)
        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
            image_path)  # obtain the central point lon and lat of sat imagery

        bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                         lat_min])  # including the upper left coords and bottom right coords of the image
        sat_transform = torch.tensor(list(transform)[:6]).unsqueeze(0)

        # labels = self.categories.index(selection['category'])

        img_as_tensor = self.sat_transform(images)  # (c, h, w)

        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        timestamps = self.process_time_stamp(int(time_path.split("_")[0]), int(time_path.split("_")[1]),
                                             int(time_path.split("_")[2]))

        timestamps = torch.from_numpy(timestamps)

        img_as_tensor = torch.nan_to_num(img_as_tensor, nan=0.0)
        sv_image_as_tensor = torch.nan_to_num(sv_image_as_tensor, nan=0.0)
        sv_coordinate_as_tensor = torch.nan_to_num(sv_coordinate_as_tensor, nan=0.0)
        sv_location_information_as_tensor = torch.nan_to_num(sv_location_information_as_tensor, nan=0.0)

        sample = {
            'images': img_as_tensor.float(),
            "sv_images": sv_image_as_tensor.float(),
            'sv_coordinates': sv_coordinate_as_tensor.float(),  # this coordinate is
            'sv_location': sv_location_information_as_tensor.float(),
            'sv_time': sv_temporal_information_as_tensor.float(),
            'timestamps': timestamps.float(),
            'sat_transform': sat_transform,
            'bbox_information': bbox_information
        }

        return sample

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.Resize(input_size, interpolation=interpol_mode),
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(input_size, interpolation=interpol_mode),
        )

        return transforms.Compose(t)


class SentinelColorJitterTransform:
    def __init__(self, brightness=0.2, contrast=0.2, rgb_indices=(0, 1, 2), probability=0.7):
        """
        Apply ColorJitter only to specified RGB bands in Sentinel-2 data with a given probability.
        :param brightness: brightness jitter factor
        :param contrast: contrast jitter factor
        :param rgb_indices: indices of RGB bands (default for Sentinel-2 is B2, B3, B4 -> indices (1, 2, 3))
        :param probability: probability of applying ColorJitter to the RGB bands
        """
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        self.rgb_indices = rgb_indices
        self.probability = probability

    def __call__(self, sample):
        # Apply ColorJitter with the specified probability
        if torch.rand(1).item() < self.probability:
            # Extract RGB bands and convert them to a PyTorch tensor format
            rgb_bands = sample[self.rgb_indices, :, :]
            rgb_bands = self.color_jitter(rgb_bands)

            # Replace original RGB bands with jittered versions
            sample[self.rgb_indices, :, :] = rgb_bands
        return sample


class SentinelStreetViewPairedImageDataset_for_INF(SatelliteDataset):
    """
    For v2, instead of using zero padding, I repeat the sv images to fill the missing images
    """
    label_types = ['value', 'one-hot']
    sentinel_mean = [1172.9397, 1378.0846, 1509.3327,
                     1750.0275, 2073.2758, 2207.3283, 2245.1629,
                     2284.6384, 2231.7283, 1899.1932]
    sentinel_std = [706.6250, 720.1862, 783.1424,
                    707.1962, 714.2782, 748.1827, 852.3585,
                    762.0849, 690.0165, 669.9036]

    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    def __init__(self,
                 root_path,
                 meta_data_csv_path,
                 sat_transform,
                 sv_transform
                 ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """

        super().__init__(in_c=13)
        self.main_path = os.path.join(root_path, "images")  # image main path including many subfolders
        # self.meta_data_main_path = os.path.join(root_path, "image_meta_data")
        self.rs_main_path = os.path.join(root_path, "rs_data")
        self.all_cities = os.listdir(self.main_path)
        self.jpeg_paths = self.get_all_jpeg_paths(self.main_path)
        self.sat_transform = sat_transform
        self.sv_transform = sv_transform
        # self.meta_data_df = pd.read_csv(meta_data_csv_path)
    #
    # def get_all_jpeg_paths(self, main_path, samples=10000): # for debugging the code, shrinking the time
    #     jpeg_paths = []
    #     count = 0
    #     # 遍历主文件夹及所有子文件夹
    #     for root, dirs, files in os.walk(main_path):
    #         # 查找所有 JPEG 图片并将路径添加到列表中
    #         for file in files:
    #             if file.lower().endswith(('.jpeg', '.jpg')):
    #                 jpeg_paths.append(os.path.join(root, file))
    #                 count +=1
    #                 if count == 15000:
    #                     break
    #         if count == 15000:
    #             break
    #     jpeg_paths = random.sample(jpeg_paths, samples)
    #     return jpeg_paths

    def get_all_jpeg_paths(self, main_path, samples=1500000):
        jpeg_paths = []
        # 遍历主文件夹及所有子文件夹
        for root, dirs, files in os.walk(main_path):
            # 查找所有 JPEG 图片并将路径添加到列表中
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg')):
                    jpeg_paths.append(os.path.join(root, file))
        jpeg_paths = random.sample(jpeg_paths, samples)
        return jpeg_paths

    def __len__(self):
        return len(self.jpeg_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)
        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def process_time_stamp(self, year, month, hour, min_year=2002):
        return np.array([year - min_year, month - 1, hour])

    def extend_list(self, original_list, target_length):
        # For enlarge the number of sv images list
        extended_list = original_list[:]
        while len(extended_list) < target_length:
            tuple_to_add = random.choice(original_list)
            extended_list.append(tuple_to_add)

        return extended_list

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        sv_path = self.jpeg_paths[idx]
        sv_uuid = os.path.splitext(os.path.basename(sv_path))[0]
        city_id = os.path.basename(os.path.dirname(sv_path))
        meta_data_path = os.path.join("/data/zeping_data/data/gsv_rs_project/street_scapes/image_meta_data",
                                      f"{sv_uuid}.json")
        alternative_meta_data_path = os.path.join("/home/zl22853/data/sv_meta_data", f"{sv_uuid}_meta_data.json")
        rs_path = random.choice(glob(os.path.join(self.rs_main_path, city_id, "*.tif")))
        try:
            with open(meta_data_path, 'r') as f:
                meta_data_json = json.load(f)
                lon = meta_data_json['lon']
                lat = meta_data_json['lat']
                # year = meta_data_json['year']
                # month = meta_data_json['month']
        except:
            with open(alternative_meta_data_path, 'r') as f:
                meta_data_json = json.load(f)
                lon = meta_data_json['lon']
                lat = meta_data_json['lat']
                # year = meta_data_json['year']
                # month = meta_data_json['month']
            # except:
            #     meta_data = self.meta_data_df[self.meta_data_df['uuid'] == sv_uuid].to_dict(orient='records')
            #     lon = meta_data[0]['lon']
            #     lat = meta_data[0]['lat']
            #     # year = meta_data[0]['year']
            #     # month = meta_data[0]['month']

        # lon = meta_data[0]['lon']
        # lat = meta_data[0]['lat']
        #
        # year = meta_data[0]['year']
        # month = meta_data[0]['month']

        sv_image = skimage.io.imread(sv_path).astype(np.float32)
        sv_image_as_tensor = self.sv_transform(sv_image)
        sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
            rs_path)
        rs_image = self.open_image(rs_path)
        rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 10:]), axis=2)
        rs_image_as_tensor = self.sat_transform(rs_image)

        bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                         lat_min])  # including the upper left coords and bottom right coords of the image

        sv_location_information_as_tensor = torch.tensor([lon, lat])
        sample = {
            'rs_image': rs_image_as_tensor.float(),
            "sv_image": sv_image_as_tensor.float(),
            'sv_location': sv_location_information_as_tensor.float(),
            'bbox_information': bbox_information
        }
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                print(f"NaN detected in {key}")
                sample[key] = torch.nan_to_num(value, nan=0.0)
        return sample

    @staticmethod
    def build_transform_sat(is_train, input_size, mean, std):

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        t = []

        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.Resize((input_size, input_size), interpolation=interpol_mode),
            )
            t.append(SentinelColorJitterTransform(brightness=0.2, contrast=0.2, rgb_indices=(1, 2, 3), probability=0.7))
            # t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)
        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize((input_size, input_size), interpolation=interpol_mode),
        )

        # t.append(transforms.CenterCrop(input_size))
        return transforms.Compose(t)

    @staticmethod
    def build_transform_sv(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        t = []
        if is_train:
            t.append(ImageNormalizer(mean, std))
            t.append(transforms.ToTensor())
            t.append(
                transforms.Resize((input_size, input_size), interpolation=interpol_mode),
            )
            # t.append(transforms.Normalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.RandomHorizontalFlip())
            t.append(transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.3))
            # transforms.RandomGrayscale(p=0.1)
            return transforms.Compose(t)
        t.append(ImageNormalizer(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize((input_size, input_size), interpolation=interpol_mode),
            # to maintain same ratio w.r.t. 224 images
        )
        # t.append(transforms.Normalize(mean, std))
        #  t.append(transforms.CenterCrop(input_size))
        return transforms.Compose(t)

    # @staticmethod
    # def build_transform_sv(is_train, input_size, mean, std):
    #     # train transform
    #     interpol_mode = transforms.InterpolationMode.BICUBIC
    #     t = []
    #
    #     def check_nan(tensor, step_name):
    #         if torch.isnan(tensor).any():
    #             print(f"NaN values found after {step_name}")
    #         return tensor
    #
    #     if is_train:
    #         t.append(ImageNormalizer(mean, std))
    #         # Apply ToTensor and check for NaN
    #         t.append(transforms.ToTensor())
    #         t.append(lambda img: check_nan(img, "ToTensor"))
    #
    #         # Resize image
    #         t.append(transforms.Resize((input_size, input_size), interpolation=interpol_mode))
    #
    #         # Normalize and check for NaN
    #
    #         # Additional augmentations with NaN check
    #         t.append(transforms.RandomHorizontalFlip())
    #         t.append(lambda img: check_nan(img, "RandomHorizontalFlip"))
    #
    #         t.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3))
    #         t.append(lambda img: check_nan(img, "RandomApply(ColorJitter)"))
    #
    #         return transforms.Compose(t)
    #
    #     # Validation transform steps
    #     t.append(transforms.ToTensor())
    #     t.append(lambda img: check_nan(img, "ToTensor"))
    #
    #     t.append(transforms.Resize((input_size, input_size), interpolation=interpol_mode))
    #
    #     t.append(transforms.Normalize(mean, std))
    #     t.append(lambda img: check_nan(img, "Normalize"))
    #
    #     return transforms.Compose(t)


class SentinelStreetViewPairedImageDataset_for_INF_debug(SatelliteDataset):
    """
    For v2, instead of using zero padding, I repeat the sv images to fill the missing images
    """
    label_types = ['value', 'one-hot']
    sentinel_mean = [1051.6516, 1172.9397, 1378.0846, 1509.3327,
                     1750.0275, 2073.2758, 2207.3283, 2245.1629,
                     2284.6384, 2337.5222, 2231.7283, 1899.1932]
    sentinel_std = [441.2129, 706.6250, 720.1862, 783.1424,
                    707.1962, 714.2782, 748.1827, 852.3585,
                    762.0849, 681.2236, 690.0165, 669.9036]

    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    def __init__(self,
                 root_path,
                 meta_data_csv_path,
                 sat_transform,
                 sv_transform
                 ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """

        super().__init__(in_c=13)
        self.main_path = os.path.join(root_path, "images")  # image main path including many subfolders
        self.meta_data_main_path = os.path.join(root_path, "images_meta_data")
        self.rs_main_path = os.path.join(root_path, "rs_data")
        self.all_cities = os.listdir(self.main_path)
        self.jpeg_paths = self.get_all_jpeg_paths(self.main_path)
        self.sat_transform = sat_transform
        self.sv_transform = sv_transform
        self.meta_data_df = pd.read_csv(meta_data_csv_path)

    def get_all_jpeg_paths(self, main_path, samples=1000000):
        jpeg_paths = []
        # 遍历主文件夹及所有子文件夹
        for root, dirs, files in os.walk(main_path):
            # 查找所有 JPEG 图片并将路径添加到列表中
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg')):
                    jpeg_paths.append(os.path.join(root, file))
        jpeg_paths = random.sample(jpeg_paths, samples)
        return jpeg_paths

    def __len__(self):
        return len(self.jpeg_paths)

    def open_image(self, img_path):
        start_time = time.time()

        with rasterio.open(img_path) as data:
            img = data.read()

        end_time = time.time()
        print(f"[Timing] open_image: {end_time - start_time:.2f} seconds")

        return img.transpose(1, 2, 0).astype(np.float32)

    def process_time_stamp(self, year, month, hour, min_year=2002):
        return np.array([year - min_year, month - 1, hour])

    def extend_list(self, original_list, target_length):
        # For enlarge the number of sv images list
        extended_list = original_list[:]
        while len(extended_list) < target_length:
            tuple_to_add = random.choice(original_list)
            extended_list.append(tuple_to_add)

        return extended_list

    def __getitem__(self, idx):
        total_start_time = time.time()

        # Measure JPEG image loading time
        sv_load_start = time.time()
        sv_path = self.jpeg_paths[idx]
        sv_uuid = os.path.splitext(os.path.basename(sv_path))[0]
        city_id = os.path.basename(os.path.dirname(sv_path))
        sv_image = skimage.io.imread(sv_path).astype(np.float32)
        sv_image_as_tensor = self.sv_transform(sv_image)
        sv_load_end = time.time()
        print(f"[Timing] sv_image loading and transformation: {sv_load_end - sv_load_start:.2f} seconds")

        # Measure metadata loading time
        meta_load_start = time.time()
        meta_data_path = os.path.join(self.meta_data_main_path, f"{sv_uuid}_meta_data.json")
        try:
            with open(meta_data_path, 'r') as f:
                meta_data_json = json.load(f)
                lon = meta_data_json['lon']
                lat = meta_data_json['lat']
        except:
            raise None
        # meta_data = self.meta_data_df[self.meta_data_df['uuid'] == sv_uuid].to_dict(orient='records')
        # lon = meta_data[0]['lon']
        # lat = meta_data[0]['lat']
        meta_load_end = time.time()
        print(f"[Timing] Metadata loading: {meta_load_end - meta_load_start:.2f} seconds")

        # Measure RS image loading time and get bounding box information
        rs_load_start = time.time()
        rs_path = random.choice(glob(os.path.join(self.rs_main_path, city_id, "*.tif")))
        rs_image = self.open_image(rs_path)
        rs_image_as_tensor = self.sat_transform(rs_image)

        # 获取 lon_min, lat_max, lon_max, lat_min
        sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(rs_path)
        bbox_information = torch.tensor([lon_min, lat_max, lon_max, lat_min])

        rs_load_end = time.time()
        # print(f"[Timing] rs_image loading, transformation, and bbox extraction: {rs_load_end - rs_load_start:.2f} seconds")

        # Measure total __getitem__ time
        total_end_time = time.time()
        # print(f"[Timing] Total __getitem__: {total_end_time - total_start_time:.2f} seconds")

        # Construct the sample
        sv_location_information_as_tensor = torch.tensor([lon, lat])
        sample = {
            'rs_image': rs_image_as_tensor.float(),
            "sv_image": sv_image_as_tensor.float(),
            'sv_location': sv_location_information_as_tensor.float(),
            'bbox_information': bbox_information
        }

        return sample

    @staticmethod
    def build_transform_sat(is_train, input_size, mean, std):

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        t = []

        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.Resize((input_size, input_size), interpolation=interpol_mode),
            )
            # t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)
        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize((input_size, input_size), interpolation=interpol_mode),
        )

        # t.append(transforms.CenterCrop(input_size))
        return transforms.Compose(t)

    @staticmethod
    def build_transform_sv(is_train, input_size, mean, std):
        """
        Build the transform with timing for each transformation step.
        """
        interpol_mode = transforms.InterpolationMode.BICUBIC
        transform_steps = []

        if is_train:
            transform_steps += [
                ("ToTensor", transforms.ToTensor()),
                ("Resize", transforms.Resize((input_size, input_size), interpolation=interpol_mode)),
                ("Normalize", transforms.Normalize(mean, std)),
                ("RandomHorizontalFlip", transforms.RandomHorizontalFlip()),
                # ("ColorJitter", transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3)),
                # ("RandomGrayscale", transforms.RandomGrayscale(p=0.1)),
            ]
        else:
            transform_steps += [
                ("Normalize", SentinelNormalize(mean, std)),
                ("ToTensor", transforms.ToTensor()),
                ("Resize", transforms.Resize((input_size, input_size), interpolation=interpol_mode))
            ]

        def timed_transform(image):
            transformed_image = image
            for name, transform in transform_steps:
                start_time = time.time()
                transformed_image = transform(transformed_image)
                end_time = time.time()
                # print(f"[Timing] Transform step '{name}': {end_time - start_time:.4f} seconds")
            return transformed_image

        return timed_transform


class SentinelStreetViewPairedImageDataset_V2(SatelliteDataset):
    """
    For v2, instead of using zero padding, I repeat the sv images to fill the missing images
    """
    label_types = ['value', 'one-hot']
    sentinel_mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
                     1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
                     1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    sentinel_std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
                    948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
                    1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    def __init__(self,
                 csv_path,
                 sv_images_path,
                 sat_images_path,
                 sat_transform,
                 sv_transform,
                 repeat=1,
                 len_sv=20,
                 dropped_bands=None,
                 imagenet_initialized=False
                 ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path)
        self.sv_images_path = sv_images_path
        self.sat_images_path = sat_images_path
        self.len_sv = len_sv
        self.in_c = self.in_c - len(dropped_bands)
        self.sat_transform = sat_transform
        self.sv_transform = sv_transform
        self.dropped_bands = dropped_bands
        self.repeat = repeat
        self.imagenet_initialized = imagenet_initialized
        self.df = pd.concat([self.df] * self.repeat, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def process_time_stamp(self, year, month, hour, min_year=2002):

        return np.array([year - min_year, month - 1, hour])

    def extend_list(self, original_list, target_length):
        # For enlarge the number of sv images list
        extended_list = original_list[:]
        while len(extended_list) < target_length:
            tuple_to_add = random.choice(original_list)
            extended_list.append(tuple_to_add)

        return extended_list

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """

        selection = self.df.iloc[idx]
        panoids = ast.literal_eval(selection["panoid"])
        sv_coordinate = ast.literal_eval(selection["sv_coordinates"])
        patch_number = selection["patch"]
        city = selection["city"]

        combined_panoids_sv_coordinate = list(zip(panoids, sv_coordinate))
        sv_location_information = []
        sv_temporal_information = []
        if len(combined_panoids_sv_coordinate) < self.len_sv:
            combined_panoids_sv_coordinate = self.extend_list(combined_panoids_sv_coordinate, self.len_sv)
            sv_image_as_tensor_list = []
            combined_panoids_sv_coordinate_need = random.sample(combined_panoids_sv_coordinate, self.len_sv)
            panoids = [tup[0] for tup in combined_panoids_sv_coordinate_need]
            sv_coordinate = [tup[1] for tup in combined_panoids_sv_coordinate_need]
            for panoid in panoids:
                sv_all_files_in_folder = os.listdir(os.path.join(self.sv_images_path, city + "GSV", panoid))
                sv_path = os.path.join(self.sv_images_path, city + "GSV", panoid, random.choice(list(
                    filter(lambda x: x.endswith('.jpg'), sv_all_files_in_folder))))
                meta_file_path = os.path.join(self.sv_images_path, city + "GSV", panoid, "meta_data.json")
                with open(meta_file_path, 'r') as f:
                    meta_data = json.load(f)

                lon = meta_data['lon']
                lat = meta_data['lat']
                year = meta_data['year']
                month = meta_data['month']

                sv_location_information.append([float(lon), float(lat)])
                sv_temporal_information.append([int(year), int(month)])
                sv_image = skimage.io.imread(sv_path).astype(np.float32)
                if self.imagenet_initialized:
                    sv_image = resize(sv_image, (224, 224))
                sv_image_as_tensor = self.sv_transform(sv_image)
                sv_image_as_tensor_list.append(sv_image_as_tensor.unsqueeze(0))

            sv_image_as_tensor = torch.cat(sv_image_as_tensor_list, dim=0)
            sv_coordinate_as_tensor = torch.tensor(sv_coordinate)
            sv_location_information_as_tensor = torch.tensor(sv_location_information)
            sv_temporal_information_as_tensor = torch.tensor(sv_temporal_information)

        else:
            sv_image_as_tensor_list = []
            combined_panoids_sv_coordinate_need = random.sample(combined_panoids_sv_coordinate, self.len_sv)
            panoids = [tup[0] for tup in combined_panoids_sv_coordinate_need]
            sv_coordinate = [tup[1] for tup in combined_panoids_sv_coordinate_need]

            for panoid in panoids:
                sv_all_files_in_folder = os.listdir(os.path.join(self.sv_images_path, city + "GSV", panoid))
                sv_path = os.path.join(self.sv_images_path, city + "GSV", panoid, random.choice(list(
                    filter(lambda x: x.endswith('.jpg'), sv_all_files_in_folder))))
                meta_file_path = os.path.join(self.sv_images_path, city + "GSV", panoid, "meta_data.json")
                with open(meta_file_path, 'r') as f:
                    meta_data = json.load(f)

                lon = meta_data['lon']
                lat = meta_data['lat']
                year = meta_data['year']
                month = meta_data['month']

                sv_location_information.append([float(lon), float(lat)])
                sv_temporal_information.append([int(year), int(month)])
                sv_image = skimage.io.imread(sv_path).astype(np.float32)  # H,W,C
                if self.imagenet_initialized:
                    sv_image = resize(sv_image, (224, 224))
                sv_image_as_tensor = self.sv_transform(sv_image)
                sv_image_as_tensor_list.append(sv_image_as_tensor.unsqueeze(0))

            sv_image_as_tensor = torch.cat(sv_image_as_tensor_list, dim=0)
            sv_coordinate_as_tensor = torch.tensor(sv_coordinate)
            sv_location_information_as_tensor = torch.tensor(sv_location_information)
            sv_temporal_information_as_tensor = torch.tensor(sv_temporal_information)

        time_paths = os.listdir(os.path.join(self.sat_images_path, f"{city}_clip"))
        while True:
            time_path = random.choice(time_paths)
            image_path = os.path.join(self.sat_images_path, f"{city}_clip", time_path, patch_number + ".tif")

            images = self.open_image(image_path)  # (h, w, c)
            if not np.all(images == 0):
                break
            else:
                time_paths.remove(time_path)
        # images, image_path = self.load_valid_image(time_paths)
        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
            image_path)  # obtain the central point lon and lat of sat imagery

        bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                         lat_min])  # including the upper left coords and bottom right coords of the image
        sat_transform = torch.tensor(list(transform)[:6]).unsqueeze(0)

        # labels = self.categories.index(selection['category'])

        img_as_tensor = self.sat_transform(images)  # (c, h, w)

        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        timestamps = self.process_time_stamp(int(time_path.split("_")[0]), int(time_path.split("_")[1]),
                                             int(time_path.split("_")[2]))

        timestamps = torch.from_numpy(timestamps)

        img_as_tensor = torch.nan_to_num(img_as_tensor, nan=0.0)
        sv_image_as_tensor = torch.nan_to_num(sv_image_as_tensor, nan=0.0)
        sv_coordinate_as_tensor = torch.nan_to_num(sv_coordinate_as_tensor, nan=0.0)
        sv_location_information_as_tensor = torch.nan_to_num(sv_location_information_as_tensor, nan=0.0)

        sample = {
            'images': img_as_tensor.float(),
            "sv_images": sv_image_as_tensor.float(),
            'sv_coordinates': sv_coordinate_as_tensor.float(),
            'sv_location': sv_location_information_as_tensor.float(),
            'sv_time': sv_temporal_information_as_tensor.float(),
            'timestamps': timestamps.float(),
            'sat_transform': sat_transform,
            'bbox_information': bbox_information
        }

        return sample

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC
        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(input_size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }

        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

class SV_perception(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    """

    def __init__(self, file_path, transform, benchmark, sat_transform, multi_model=""):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.selected_indicators = ['Beautiful', 'Boring', 'Depressing', 'Lively', 'Safe', 'Wealthy']
        self.multi_model = multi_model
        self.sat_transform = sat_transform

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        image_path = self.database.iloc[index].image_path
        uuid = self.database.iloc[index].uuid

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass

        if self.multi_model !="":
            updated_image_path = "/".join(image_path.split("/")[:-1])
            rs_image_path = os.path.abspath(os.path.join(updated_image_path, random.choice(os.listdir(updated_image_path))))
            rs_image = skimage.io.imread(rs_image_path)
            meta_data_path = os.path.join("/data/zeping_data/data/gsv_rs_project/street_scapes/image_meta_data",
                                          f"{uuid}.json")
            alternative_meta_data_path = os.path.join("/home/zl22853/data/sv_meta_data", f"{uuid}_meta_data.json")
            try:
                with open(meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            except:
                with open(alternative_meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
                rs_image_path)
            rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 10:]), axis=2)
            rs_image_as_tensor = self.sat_transform(rs_image)

            bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                             lat_min])  # including the upper left coords and bottom right coords of the image

            sv_location_information_as_tensor = torch.tensor([lon, lat])


            values = [self.database.iloc[index][key] for key in self.selected_indicators]
            values = torch.tensor(values).float()
            sample = {
                'rs_image': rs_image_as_tensor.float(),
                "sv_image": image_as_tensor.float(),
                'sv_location': sv_location_information_as_tensor.float(),
                'bbox_information': bbox_information,
                'values': values
            }
            return sample




        values = [self.database.iloc[index][key] for key in self.selected_indicators]
        values = torch.tensor(values).float()

        return {"image": image_as_tensor,
                'values': values}

    def __len__(self):
        return len(self.database)


class SV_metadata_quality(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    """

    def __init__(self, file_path, transform, benchmark, sat_transform, multi_model=""):
        super().__init__()
        self.file_path = file_path
        self.database= self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.labels = {"good":0,"slightly poor":1, "very poor": 2}
        self.multi_model=multi_model
        self.sat_transform = sat_transform
    def process_data(self, file_path):
        quality_data = pd.read_csv(file_path)
        return quality_data

    def __getitem__(self, index):

        image_path = self.database.iloc[index].img_path
        uuid = self.database.iloc[index].uuid
        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)
        quality = self.database.iloc[index].quality
        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass


        label = torch.tensor(self.labels[quality])
        if self.multi_model != "":
            updated_image_path = "/".join(image_path.split("/")[:-1])
            rs_image_path = os.path.abspath(
                os.path.join(updated_image_path, random.choice(os.listdir(updated_image_path))))
            rs_image = skimage.io.imread(rs_image_path)
            meta_data_path = os.path.join("/data/zeping_data/data/gsv_rs_project/street_scapes/image_meta_data",
                                          f"{uuid}.json")
            alternative_meta_data_path = os.path.join("/home/zl22853/data/sv_meta_data", f"{uuid}_meta_data.json")
            try:
                with open(meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            except:
                with open(alternative_meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
                rs_image_path)
            rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 10:]), axis=2)
            rs_image_as_tensor = self.sat_transform(rs_image)

            bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                             lat_min])  # including the upper left coords and bottom right coords of the image

            sv_location_information_as_tensor = torch.tensor([lon, lat])

            sample = {
                'rs_image': rs_image_as_tensor.float(),
                "sv_image": image_as_tensor.float(),
                'sv_location': sv_location_information_as_tensor.float(),
                'bbox_information': bbox_information,
                'values': label
            }
            return sample

        return {"image": image_as_tensor,
                'values': label}

    def __len__(self):
        return len(self.database)

class SV_metadata_platform(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    """

    def __init__(self, file_path, transform, benchmark,sat_transform,multi_model=""):
        super().__init__()
        self.file_path = file_path
        self.database= self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.labels = {
                'driving surface': 0,
                'walking surface': 1,
                'cycling surface': 2,
                'tunnel': 3,
                'fields': 4,
                'railway': 5
                }
        self.sat_transform = sat_transform
        self.multi_model=multi_model
    def process_data(self, file_path):
        platform_data = pd.read_csv(file_path)
        return platform_data

    def __getitem__(self, index):

        image_path = self.database.iloc[index].img_path
        uuid = self.database.iloc[index].uuid
        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)
        quality = self.database.iloc[index].platform
        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass

        label = torch.tensor(self.labels[quality])
        if self.multi_model != "":
            updated_image_path = "/".join(image_path.split("/")[:-1])
            rs_image_path = os.path.abspath(
                os.path.join(updated_image_path, random.choice(os.listdir(updated_image_path))))
            rs_image = skimage.io.imread(rs_image_path)
            meta_data_path = os.path.join("/data/zeping_data/data/gsv_rs_project/street_scapes/image_meta_data",
                                          f"{uuid}.json")
            alternative_meta_data_path = os.path.join("/home/zl22853/data/sv_meta_data", f"{uuid}_meta_data.json")
            try:
                with open(meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            except:
                with open(alternative_meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
                rs_image_path)
            rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 10:]), axis=2)
            rs_image_as_tensor = self.sat_transform(rs_image)

            bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                             lat_min])  # including the upper left coords and bottom right coords of the image

            sv_location_information_as_tensor = torch.tensor([lon, lat])

            sample = {
                'rs_image': rs_image_as_tensor.float(),
                "sv_image": image_as_tensor.float(),
                'sv_location': sv_location_information_as_tensor.float(),
                'bbox_information': bbox_information,
                'values': label
            }
            return sample

        return {"image": image_as_tensor,
                'values': label}

    def __len__(self):
        return len(self.database)


class SV_metadata_view_direction(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    """

    def __init__(self, file_path, transform, benchmark,sat_transform,multi_model=""):
        super().__init__()
        self.file_path = file_path
        self.database= self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.labels = {
                'front/back': 0,
                'side': 1
                }
        self.multi_model=multi_model
        self.sat_transform = sat_transform
    def process_data(self, file_path):
        view_direction = pd.read_csv(file_path)
        return view_direction

    def __getitem__(self, index):

        image_path = self.database.iloc[index].img_path
        uuid = self.database.iloc[index].uuid
        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)
        quality = self.database.iloc[index].view_direction
        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass

        label = torch.tensor(self.labels[quality])
        if self.multi_model != "":
            updated_image_path = "/".join(image_path.split("/")[:-1])
            rs_image_path = os.path.abspath(
                os.path.join(updated_image_path, random.choice(os.listdir(updated_image_path))))
            rs_image = skimage.io.imread(rs_image_path)
            meta_data_path = os.path.join("/data/zeping_data/data/gsv_rs_project/street_scapes/image_meta_data",
                                          f"{uuid}.json")
            alternative_meta_data_path = os.path.join("/home/zl22853/data/sv_meta_data", f"{uuid}_meta_data.json")
            try:
                with open(meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            except:
                with open(alternative_meta_data_path, 'r') as f:
                    meta_data_json = json.load(f)
                    lon = meta_data_json['lon']
                    lat = meta_data_json['lat']
            sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
                rs_image_path)
            rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 10:]), axis=2)
            rs_image_as_tensor = self.sat_transform(rs_image)

            bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                             lat_min])  # including the upper left coords and bottom right coords of the image

            sv_location_information_as_tensor = torch.tensor([lon, lat])

            sample = {
                'rs_image': rs_image_as_tensor.float(),
                "sv_image": image_as_tensor.float(),
                'sv_location': sv_location_information_as_tensor.float(),
                'bbox_information': bbox_information,
                'values': label
            }
            return sample

        return {"image": image_as_tensor,
                'values': label}

    def __len__(self):
        return len(self.database)

class SV_meta_data(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    """

    def __init__(self, file_path, transform, benchmark):
        super().__init__()
        self.file_path = file_path
        self.database = self.process_data(file_path)
        self.transform = transform
        self.benchmark = benchmark
        self.selected_indicators = ['Beautiful', 'Boring', 'Depressing', 'Lively', 'Safe', 'Wealthy']

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def __getitem__(self, index):
        image_path = self.database.iloc[index].image_path


        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass
        values = [self.database.iloc[index][key] for key in self.selected_indicators]
        values = torch.tensor(values)

        return {"image": image_as_tensor,
                'values': values}

    def __len__(self):
        return len(self.database)


class SV_Ecomoical_Indicator_dataset(Dataset):
    sv_mean = [135.47184384, 142.28621995, 145.12271992]
    sv_std = [59.19978436, 59.35279219, 72.02295734]

    """
    Street view imagery is following
    -parent path
        -panoid
            -panoid_xx.jpg
            -panoid_xx.jpg
            ...
        ...
    Given a csv file (specified by file_path), it should contains the following entry:
    panoid, path (path is for reference of images, see above structure)
    """

    def __init__(self, file_path, transform, eco_indicators, benchmark, sat_transform=None, multi_model=""):
        super().__init__()
        self.file_path = file_path
        self.database, self.sv_rs_database = self.process_data(file_path)
        self.transform = transform
        self.eco_indicators = eco_indicators
        self.benchmark = benchmark
        self.multi_model = multi_model
        self.sat_transform = sat_transform
        """
        popden_cbg: Population density in the census block group.
        bachelor_p:Percentage of people with a bachelor's degree or higher.
        health:Crude rates of health conditions like high blood pressure, diabetes, obesity.
        poc_cbg:Percentage of people of color in the census block group.
        mhincome_c:Median household income in the census tract.
        stationdis:Distance to the nearest station (public transportation).
        walkbike_p:Percentage of people who walk or bike.
        over65_per: Percentage of the population over 65 years old.
        violent: Crime rates (violent and non-violent).
        sky: The area of sky
        """
        self.selected_indicators = ["popden_cbg", "bachelor_p", "health", "poc_cbg", "mhincome_c", "stationdis",
                                    "walkbike_p", "over65_per", "violent", "sky"]

    def process_data(self, file_path):
        data = pd.read_csv(file_path)
        sv_rs_database = pd.read_csv("/data/zeping_data/data/gsv_rs_project/satellite_imagery/summary.csv")
        # data = data.sample(n=500, random_state=46)
        return data,sv_rs_database

    def __getitem__(self, index):
        panoid = self.database.iloc[index].panoid
        path = self.database.iloc[index].path

        files = os.listdir(os.path.join(path, panoid))

        image_path = os.path.join(path, panoid,
                                  random.choice(list(filter(lambda x: x.endswith('.jpg'), files))))

        image = skimage.io.imread(image_path)
        image_as_tensor = self.transform(image)

        if self.benchmark in ["imagenet", "street_view_imagenet", "street_view_liif", "moco_v3", "mae"]:
            # transform the data to be 224*224
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "satmae":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "croma":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 96, 96)
        elif self.benchmark == "pis":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        elif self.benchmark == "random":
            image_as_tensor = misc.resize_tensor(image_as_tensor, 224, 224)
        else:
            pass

        with open(os.path.join(path, panoid, "meta_data.json"), 'r') as f:
            meta_data = json.load(f)
        # print(image_as_tensor.shape)
        lon = torch.tensor(meta_data['lon'])
        lat = torch.tensor(meta_data['lat'])
        year = torch.tensor(meta_data['year'])
        month = torch.tensor(meta_data['month'])

        with open(os.path.join(path, panoid, "eco_data_normalized.json"), 'r') as f:
            eco_data = json.load(f)
            values = [eco_data[key] for key in self.selected_indicators]
            values = torch.tensor(values)

        if self.multi_model != "":
            city_name = os.path.basename(os.path.normpath(path)).replace("GSV", "") # boston or losangeles

            rs_image_path = os.path.join("/data/zeping_data/data/gsv_rs_project/satellite_imagery/patches/",f"{city_name}_{panoid}.tif")
            rs_image = skimage.io.imread(rs_image_path)
            with open(os.path.join(path, panoid, "meta_data.json"), 'r') as f:
                meta_data = json.load(f)

            lon = torch.tensor(meta_data['lon'])
            lat = torch.tensor(meta_data['lat'])

            sat_lon, sat_lat, transform, lon_min, lat_max, lon_max, lat_min = get_image_center_coordinates(
                rs_image_path)

            rs_image = np.concatenate((rs_image[:, :, 1:9], rs_image[:, :, 11:]), axis=2)
            rs_image_as_tensor = self.sat_transform(rs_image)

            bbox_information = torch.tensor([lon_min, lat_max, lon_max,
                                             lat_min])  # including the upper left coords and bottom right coords of the image

            sv_location_information_as_tensor = torch.tensor([lon, lat])

            with open(os.path.join(path, panoid, "eco_data_normalized.json"), 'r') as f:
                eco_data = json.load(f)
                values = [eco_data[key] for key in self.selected_indicators]
                values = torch.tensor(values).float()

            sample = {
                'rs_image': rs_image_as_tensor.float(),
                "sv_image": image_as_tensor.float(),
                'sv_location': sv_location_information_as_tensor.float(),
                'bbox_information': bbox_information,
                'values': values
            }
            return sample

        return {"image": image_as_tensor,
                "lon": lon,
                "lat": lat,
                "year": year,
                "month": month,
                'values': values
                # "attribute": attribute.unsqueeze(0)
                }


    def __len__(self):
        return len(self.database)


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)
    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == 'temporal':
        dataset = CustomDatasetFromImagesTemporal(args.csv_path)

    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(args.csv_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)

    elif args.dataset_type == 'sentinel_sv':
        sat_mean = SentinelStreetViewPairedImageDataset.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset.sentinel_std
        sv_mean = SentinelStreetViewPairedImageDataset.sv_mean
        sv_std = SentinelStreetViewPairedImageDataset.sv_std
        sat_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_size, sat_mean,
                                                                             sat_std)
        sv_transform = SentinelStreetViewPairedImageDataset_V2.build_transform(is_train, args.input_sv_size, sv_mean,
                                                                               sv_std)
        dataset = SentinelStreetViewPairedImageDataset_V2(sat_transform=sat_transform, sv_transform=sv_transform,
                                                          sat_images_path="/data/zeping_data/data/gsv_rs_project/satellite_imagery",
                                                          csv_path="/data/zeping_data/data/gsv_rs_project/satellite_imagery/summary.csv",
                                                          sv_images_path="/data/zeping_data/data/gsv_rs_project",
                                                          dropped_bands=[0, 9, 10], len_sv=args.len_sv,
                                                          repeat=args.repeat,
                                                          imagenet_initialized=args.imagenet_initialized)
    elif args.dataset_type == "sentinel_sv_inf":
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sv_mean = SentinelStreetViewPairedImageDataset_for_INF.sv_mean
        sv_std = SentinelStreetViewPairedImageDataset_for_INF.sv_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        sv_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sv(is_train,
                                                                                       args.input_sv_size,
                                                                                       sv_mean,
                                                                                       sv_std)
        dataset = SentinelStreetViewPairedImageDataset_for_INF(
            root_path="/data/zeping_data/data/gsv_rs_project/street_scapes",
            meta_data_csv_path="/data/zeping_data/data/gsv_rs_project/street_scapes/metadata_common_attributes.csv",
            sat_transform=sat_transform, sv_transform=sv_transform)
    elif args.dataset_type == 'rgb_temporal_stacked':
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == 'naip':
        from util.naip_loader import NAIP_train_dataset, NAIP_test_dataset, NAIP_CLASS_NUM
        dataset = NAIP_train_dataset if is_train else NAIP_test_dataset
        args.nb_classes = NAIP_CLASS_NUM
    elif args.dataset_type == 'economical_indicator':

        mean = SV_Ecomoical_Indicator_dataset.sv_mean
        std = SV_Ecomoical_Indicator_dataset.sv_std
        sv_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                            std)
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        if is_train:
            dataset = SV_Ecomoical_Indicator_dataset(file_path=args.train_path, transform=sv_transform,
                                                     eco_indicators=args.eco_indicators, benchmark=args.benchmark, sat_transform=sat_transform, multi_model=args.multi_model)
        else:
            dataset = SV_Ecomoical_Indicator_dataset(file_path=args.test_path, transform=sv_transform,
                                                     eco_indicators=args.eco_indicators, benchmark=args.benchmark, sat_transform=sat_transform, multi_model=args.multi_model)
    elif args.dataset_type == 'perception':

        mean = SV_perception.sv_mean
        std = SV_perception.sv_std
        sv_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                            std)
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        if is_train:
            dataset = SV_perception(file_path=args.train_path, transform=sv_transform, benchmark=args.benchmark, sat_transform=sat_transform,multi_model=args.multi_model)
        else:
            dataset = SV_perception(file_path=args.test_path, transform=sv_transform, benchmark=args.benchmark, sat_transform=sat_transform,multi_model=args.multi_model)
    elif args.dataset_type == 'meta_data_quality':
        mean = SV_metadata_quality.sv_mean
        std = SV_metadata_quality.sv_std
        sv_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                            std)
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        if is_train:
            dataset = SV_metadata_quality(file_path=args.train_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
        else:
            dataset = SV_metadata_quality(file_path=args.test_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
    elif args.dataset_type == 'meta_data_view_direction':
        mean = SV_metadata_view_direction.sv_mean
        std = SV_metadata_view_direction.sv_std
        sv_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                            std)
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        if is_train:
            dataset = SV_metadata_view_direction(file_path=args.train_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
        else:
            dataset = SV_metadata_view_direction(file_path=args.test_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
    elif args.dataset_type == 'meta_data_platform':
        mean = SV_metadata_platform.sv_mean
        std = SV_metadata_platform.sv_std
        sv_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                            std)
        sat_mean = SentinelStreetViewPairedImageDataset_for_INF.sentinel_mean
        sat_std = SentinelStreetViewPairedImageDataset_for_INF.sentinel_std
        sat_transform = SentinelStreetViewPairedImageDataset_for_INF.build_transform_sat(is_train, args.input_size,
                                                                                         sat_mean,
                                                                                         sat_std)
        if is_train:
            dataset = SV_metadata_platform(file_path=args.train_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
        else:
            dataset = SV_metadata_platform(file_path=args.test_path, transform=sv_transform, benchmark=args.benchmark,sat_transform=sat_transform,multi_model=args.multi_model)
    elif args.dataset_type == 'fmow_classification':
        mean = RS_Fmow_dataset.rs_mean
        std = RS_Fmow_dataset.rs_std
        sat_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                             std)
        if is_train:
            dataset = RS_Fmow_dataset(file_path=args.train_path, transform=sat_transform,
                                      main_path=args.main_path, benchmark=args.benchmark)
        else:
            dataset = RS_Fmow_dataset(file_path=args.test_path, transform=sat_transform,
                                      main_path=args.main_path, benchmark=args.benchmark)
    elif args.dataset_type == 'euro_sat':
        mean = EuroSatDataset.rs_means
        std = EuroSatDataset.rs_stds
        sat_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                             std)
        if is_train:
            dataset = EuroSatDataset(file_path=args.train_path, transform=sat_transform,
                                     benchmark=args.benchmark)
        else:
            dataset = EuroSatDataset(file_path=args.test_path, transform=sat_transform,
                                     benchmark=args.benchmark)
    elif args.dataset_type == 'world_strat': # ipcc
        mean = WorldStratDataset.rs_means
        std = WorldStratDataset.rs_stds
        sat_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                             std)
        if is_train:
            dataset = WorldStratDataset(file_path=args.train_path, transform=sat_transform,
                                        benchmark=args.benchmark)
        else:
            dataset = WorldStratDataset(file_path=args.test_path, transform=sat_transform,
                                        benchmark=args.benchmark)
    elif args.dataset_type == 'world_strat_lccs':
        mean = WorldStratDataset.rs_means
        std = WorldStratDataset.rs_stds
        sat_transform = SentinelStreetViewPairedImageDataset.build_transform(is_train, args.input_sv_size, mean,
                                                                             std)
        if is_train:
            dataset = WorldStratDataset_lccs(file_path=args.train_path, transform=sat_transform,
                                        benchmark=args.benchmark)
        else:
            dataset = WorldStratDataset_lccs(file_path=args.test_path, transform=sat_transform,
                                        benchmark=args.benchmark)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset
