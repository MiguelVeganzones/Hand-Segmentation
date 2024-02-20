import os
import zipfile 
import math
import logging
import subprocess

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
EGOHANDS_DATASET_URL = 'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip'
EGOHANDS_DIR = rf'{CURRENT_DIR_PATH}/egohands'
EGOHANDS_DATA_DIR = rf'{EGOHANDS_DIR}/_LABELLED_SAMPLES'

def download_file(url, dest=None):
    """Download file from an URL."""
    from tqdm import tqdm
    import requests

    if not dest:
        dest = url.split('/')[-1]

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    assert total_size != 0
    block_size = 1024
    wrote = 0
    with open(dest, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size//block_size),
                         unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    assert wrote == total_size
    
    
if __name__ == "__main__":
    logging.info("Checking directories.")
    if not os.path.exists(EGOHANDS_DIR):
        os.mkdir(EGOHANDS_DIR)
    if not os.path.exists(EGOHANDS_DATA_DIR):
        os.mkdir(EGOHANDS_DATA_DIR)

    zip_path = rf"{EGOHANDS_DIR}/{EGOHANDS_DATASET_URL.split('/')[-1]}"
    if not os.path.exists(zip_path):
        logging.info("Obtaining egohands dataset from the web.")
        download_file(EGOHANDS_DATASET_URL, zip_path)
        logging.info("Successfully downloaded egohands dataset from the web.")
    if not os.listdir(EGOHANDS_DIR):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            logging.info("Extracting egohands dataset form zip file.")
            zf.extractall(EGOHANDS_DIR)
            logging.info("Successfully extracted egohands dataset form zip file.")
    
    if "_GROUND_TRUTH_DISJUNCT_HANDS" not in os.listdir(EGOHANDS_DIR):
        subprocess.run([r"C:\Program Files\MATLAB\R2023b\bin\matlab", "-r", rf"run('{EGOHANDS_DIR}\DEMO_1') ; exit"])