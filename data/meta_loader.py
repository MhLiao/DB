import pickle
import hashlib
import os
import io
import urllib.parse as urlparse
import warnings
import numpy as np
from concern.charset_tool import stringQ2B
from hanziconv import HanziConv
from concern.config import Configurable, State
from data.text_lines import TextLines

class DataIdMetaLoader(MetaLoader):
    return_dict = State(default=False)
    scan_meta = False

    def __init__(self, return_dict=None, cmd={}, **kwargs):
        super().__init__(cmd=cmd, **kwargs)
        if return_dict is not None:
            self.return_dict = return_dict

    def parse_meta(self, data_id):
        return dict(data_id=data_id)

    def post_prosess(self, meta):
        if self.return_dict:
            return meta
        return meta['data_id']

class MetaCache(Configurable):
    META_FILE = 'meta_cache.pickle'
    client = State(default='all')

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def cache(self, nori_path, meta=None):
        if meta is None:
            return self.read(nori_path)
        else:
            return self.save(nori_path, meta)

    def read(self, nori_path):
        raise NotImplementedError

    def save(self, nori_path, meta):
        raise NotImplementedError


class FileMetaCache(MetaCache):
    storage_dir = State(default='/data/.meta_cache')

    def __init__(self, storage_dir=None, cmd={}, **kwargs):
        super(FileMetaCache, self).__init__(cmd=cmd, **kwargs)

        self.storage_dir = cmd.get('storage_dir', self.storage_dir)
        if storage_dir is not None:
            self.storage_dir = storage_dir
        self.debug = cmd.get('debug', False)

    def ensure_dir(self):
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def storate_path(self, nori_path):
        return os.path.join(self.storage_dir, self.hash(nori_path) + '.pickle')

    def hash(self, nori_path: str):
        return hashlib.md5(nori_path.encode('utf-8')).hexdigest() + '-' + self.client

    def read(self, nori_path):
        file_path = self.storate_path(nori_path)
        if not os.path.exists(file_path):
            warnings.warn(
                'Meta cache not found: ' + file_path)
            warnings.warn('Now trying to read meta from nori')
            return None
        with open(file_path, 'rb') as reader:
            try:
                return pickle.load(reader)
            except EOFError as e:  # recover from broken file
                if self.debug:
                    raise e
                return None

    def save(self, nori_path, meta):
        self.ensure_dir()

        with open(self.storate_path(nori_path), 'wb') as writer:
            pickle.dump(meta, writer)
        return True
        