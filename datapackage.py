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

        self._load_path = None
        self._resources = {}

    @classmethod
    def load(cls, pth):
        dpjson_pth = path(os.path.join(pth, "datapackage.json"))
        if not dpjson_pth.exists():
            raise IOError("No metadata file datapackage.json")
        with open(dpjson_pth, "r") as fh:
            dpjson = json.load(fh)

        name = dpjson['name']
        licenses = dpjson['licenses']
        resources = dpjson['resources']
        del dpjson['name']
        del dpjson['licenses']
        del dpjson['resources']

        dp = cls(name=name, licenses=licenses)
        dp.update(dpjson)

        for resource in resources:
            rname = resource['name']
            rpth = resource.get('path', None)
            rdata = resource.get('data', None)

            del resource['name']
            if 'path' in resource:
                del resource['path']
            if 'data' in resource:
                del resource['data']

            r = Resource(name=rname, pth=rpth, data=rdata)
            r.update(resource)
            dp.add_resource(r)

        dp._load_path = pth
        return dp

    def add_contributor(self, name, email):
        self['contributors'].append(dict(name=name, email=email))

    def add_resource(self, resource):
        self['resources'].append(resource)
        self._resources[resource['name']] = (len(self['resources'])-1, None)

    def get_resource(self, name, verify_checksums=True):
        if not self._resources.get(name, None):
            raise ValueError("no such resource: %s" % name)
        idx, data = self._resources[name]
        if not data:
            r = self['resources'][idx]
            data = r.load(self._load_path, verify_checksums=verify_checksums)
            self._resources[name] = (idx, data)
        return data

    def load_resources(self, verify_checksums=True):
        for resource in self._resources:
            self.get_resource(resource, verify_checksums=verify_checksums)

    def save(self, pth=None):
        if not pth and not self._load_path:
            raise ValueError("no target path given")
        if pth:
            self._load_path = pth
        with open(self._load_path, "w") as fh:
            json.dump(self, fh, indent=2)

    def bump_major_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        major += 1
        minor = 0
        patch = 0
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']

    def bump_minor_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        minor += 1
        patch = 0
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']

    def bump_patch_version(self):
        major, minor, patch = map(int, self['version'].split("."))
        patch += 1
        self['version'] = "%d.%d.%d" % (major, minor, patch)
        return self['version']


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
        rsize = rpath.getsize()
        self['bytes'] = rsize
        return rsize

    def calc_hash(self, pth):
        rpath = path(pth).joinpath(self['path'])
        rhash = md5(rpath)
        self['hash'] = rhash
        return rhash

    def load(self, pth=None, verify_checksums=True):
        if self.get('data', None):
            return self['data']

        if not self.get('path', None):
            raise ValueError("malformed resource")

        rpth = path(pth).joinpath(self['path'])

        # check format and load data
        if self['format'] == 'csv':
            data = pd.DataFrame.from_csv(rpth)
        elif self['format'] == 'json':
            with open(rpth, "r") as fh:
                data = json.load(fh)
        elif self['format'] == 'npy':
            data = np.load(rpth, mmap_mode='c')
        else:
            raise ValueError("unsupported format: %s" % self['format'])

        # check the file size
        size = self['size']
        if size != self.calc_size():
            raise IOError("resource has changed size on disk")

        # load the raw data and check md5
        if verify_checksums:
            rhash = self['hash']
            if rhash != self.calc_hash():
                raise IOError("resource checksum has changed")

        return data
