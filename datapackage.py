from path import path
import hashlib
import json
import numpy as np
import os
import pandas as pd
from datetime import datetime


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


class DataPackage(dict):

    def __init__(self, name, licenses):
        self['name'] = name
        self['datapackage_version'] = '1.0-beta.5'
        self['licenses'] = []
        for lid in licenses:
            if lid == 'odc-by':
                url = 'http://opendefinition.org/licenses/odc-by'
            else:
                raise ValueError("unrecognized license: %s" % lid)

            self['licenses'].append(dict(id=lid, url=url))

        self['title'] = None
        self['description'] = None
        self['homepage'] = None
        self['version'] = '0.0.1'
        self['sources'] = []
        self['keywords'] = None
        self['last_modified'] = datetime.now().isoformat(" ")
        self['image'] = None
        self['contributors'] = []
        self['resources'] = []

    def add_contributor(self, name, email):
        self['contributors'].append(dict(name=name, email=email))

    def add_resource(self, resource):
        self['resources'].append(resource)

    def save(self, pth):
        with open(pth, "w") as fh:
            json.dump(self, fh, indent=2)


class Resource(dict):

    def __init__(self, name, pth=None, data=None):
        self['name'] = name
        self['modified'] = datetime.now().isoformat(" ")

        if not pth and not data:
            raise ValueError("must specify either a path OR give raw data")
        if pth and data:
            raise ValueError("cannot specify both a pth and raw data")

        if pth:
            self['path'] = str(path(pth).joinpath(name))
        elif data:
            self['data'] = data

    def calc_size(self, pth):
        rpath = path(pth).joinpath(self['path'])
        rsize = rpath.get_size()
        self['size'] = rsize
        return rsize

    def calc_hash(self, pth):
        rpath = path(pth).joinpath(self['path'])
        rhash = md5(rpath)
        self['hash'] = rhash
        return rhash
