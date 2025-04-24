#! /usr/bin/env python3

import os
import requests
import time

TIME_LIST = [
    "00-00",
    "06-00",
    "09-00",
    "13-00",
    "14-00",
    "17-00",
    "18-00"
]

WEATHER_LIST = [
    "CLEAR_SKY",
    "LIGHT_RAIN",
    "HEAVY_RAIN"    
]

# https://apollosyntheticdataset.bj.bcebos.com/RGB_13-00_HEAVY_RAIN.zip
DOWNLOAD_URL_FORMAT = "https://apollosyntheticdataset.bj.bcebos.com/RGB_{time}_{weather}.zip"

DOWNLOAD_URLS = [
    DOWNLOAD_URL_FORMAT.format(time=time, weather=weather)
    for time in TIME_LIST
    for weather in WEATHER_LIST
]

DATASET_DIR = "/media/zyb/b67d07e3-885a-4495-8b3d-9b1be469ee5d1/datasets/ApolloSynthetic"


def download_file(url, save_path, retry=3):
    for _ in range(retry):
        try:
            response = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            time.sleep(1)
    return False

def download_all_files():
    failed_urls = []
    for url in DOWNLOAD_URLS:
        save_path = os.path.join(DATASET_DIR, os.path.basename(url))
        if not os.path.exists(save_path):
            print(f"Downloading {url} to {save_path}")
            ret = download_file(url, save_path)
            if ret:
                print(f"Downloaded {url} to {save_path}")
            else:
                print(f"Failed to download {url} to {save_path}")
                failed_urls.append(url)
        else:
            print(f"File {save_path} already exists")

    if len(failed_urls) > 0:
        print(f"Failed to download {len(failed_urls)} files")
        for url in failed_urls:
            print(url)

        print("Please check the failed urls and try again")


if __name__ == "__main__":
    download_all_files()

