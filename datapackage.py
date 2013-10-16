from path import path
import hashlib
import json
import numpy as np
import os
import pandas as pd


def md5(data_path):
    # we need to compute the md5 sum one chunk at a time, because some
    # files are too large to fit in memory
    md5 = hashlib.md5()
    with open(data_path, 'r') as fh:
        while True:
            chunk = fh.read(128)
            if not chunk:
                break
            md5.update(chunk)
    data_hash = md5.hexdigest()
    return data_hash


def load_datapackage(pth, verify_checksums=True):
    dpjson_pth = path(os.path.join(pth, "datapackage.json"))
    if not dpjson_pth.exists():
        raise IOError("No metadata file datapackage.json")
    with open(dpjson_pth, "r") as fh:
        dpjson = json.load(fh)

    resource_info = dpjson['resources']
    resources = {}
    for resource in resource_info:
        if resource.get('path', None):
            resource_pth = path(os.path.join(pth, resource['path']))

            # check format and load data
            if resource['format'] == 'csv':
                data = pd.DataFrame.from_csv(resource_pth)
            elif resource['format'] == 'json':
                with open(resource_pth, "r") as fh:
                    data = json.load(fh)
            elif resource['format'] == 'npy':
                data = np.load(resource_pth, mmap_mode='c')
            else:
                raise ValueError("unsupported format: %s" % resource['format'])

            # check the file size
            size = resource_pth.getsize()
            if size != resource['bytes']:
                raise IOError("resource has changed size on disk")

            # load the raw data and check md5
            if verify_checksums:
                if md5(resource_pth) != resource['hash']:
                    raise IOError("resource checksum has changed")

        elif 'data' in resource:
            data = resource['data']

        else:
            raise ValueError("malformed resource")

        resources[resource['name']] = data

    return resources
