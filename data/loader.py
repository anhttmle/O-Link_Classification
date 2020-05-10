import requests
import pandas as pd
import numpy as np


"""# Get data from Google Drive"""


def download_file_from_drive_id(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_from_shareable_link(url, destination):
    file_id = url.split("=")[-1]
    download_file_from_drive_id(file_id, destination)


"""# Parsing data to features/label"""


def parse_csv_data(feature_path, label_path):
    features = pd.read_csv(feature_path).values
    labels = pd.read_csv(label_path).values
    rnn_input = np.stack([features[:, (i*20):((i+1)*20)] for i in range(29)], axis=1)

    return rnn_input, labels
