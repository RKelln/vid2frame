import pickle
import shutil
from io import BytesIO
from pathlib import Path

import h5py
import lmdb
import numpy as np
from PIL import Image

# From https://github.com/jinyu121/video2frame/blob/master/storage.py

class Storage:
    def __init__(self, path):
        self.path = Path(path)
        self.database = None
        self.display_info = self.__display_info

    def put(self, video_key, clip_tmp_dir, frame_files):
        raise NotImplementedError()

    def close(self):
        pass

    def __display_info(self):
        self.__class__.display_info(self.path)

    @staticmethod
    def info(path):
        """Returns a meta info dict and the last Image"""
        raise NotImplementedError()

    @classmethod
    def display_info(cls, path):
        print(path, ":")
        meta, img = cls.info(path)
        for size, count in meta.items():
             print(f"   Image size (W,H): {size} count: {count}")


class LMDBStorage(Storage):
    map_size = 1 << 40

    def __init__(self, path):
        super().__init__(path)
        self.database = lmdb.open(str(path), map_size=LMDBStorage.map_size)

    def put(self, video_key, clip_tmp_dir, frame_files):
        with self.database.begin(write=True, buffers=True) as txn:
            for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
                data = (clip_tmp_dir / frame_path).open("rb").read()
                key = f"{video_key}/{ith_frame:08d}"
                txn.put(key.encode(), data)

    def close(self):
        self.database.sync()
        self.database.close()

    @staticmethod
    def info(path):
        path = Path(path)
        info = {}
        db = lmdb.open(str(path), map_size=LMDBStorage.map_size)
        with db.begin(write=False) as txn:
            cur = txn.cursor()
            f = cur.first()
            while f:
                k,v = cur.key(), cur.value()
                try:
                    img = Image.open(BytesIO(v))
                    ik = str(img.size)
                    info[ik] = info.get(ik, 0) + 1
                except Exception as e:
                    print(f"reading failed for {k}: {e}")
                f = cur.next()
        return info, img


class HDF5Storage(Storage):
    def __init__(self, path):
        super().__init__(path)
        self.database = h5py.File(str(path), 'w-') # fail if exists

    def put(self, video_key, clip_tmp_dir, frame_files):
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            key = f"{video_key}/{ith_frame:08d}"
            self.database[key] = np.void(data)

    def close(self):
        self.database.close()

    @staticmethod
    def info(path):
        path = Path(path)
        info = {}
        frame_db = h5py.File(str(path), 'r')
        if len(frame_db.keys()) == 0:
            print("No video frames found")
        for vid in frame_db:
            db_vid = frame_db[vid]
            for fid in db_vid:
                b = np.asarray(db_vid[fid]).tobytes()
                try:
                    img = Image.open(BytesIO(b))
                    ik = str(img.size)
                    info[ik] = info.get(ik, 0) + 1
                except:
                    print(f"reading failed for {vid}/{fid}")
        frame_db.close()
        return info, img


class PKLStorage(Storage):
    pickle_path = "frames.pkl"

    def __init__(self, path):
        super().__init__(path)

    def put(self, video_key, clip_tmp_dir, frame_files):
        save_dir = self.path / video_key
        save_dir.mkdir(exist_ok=True, parents=True)
        frame_data = []
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            data = (clip_tmp_dir / frame_path).open("rb").read()
            frame_data.append(data)
        pickle.dump(frame_data, (save_dir / PKLStorage.pickle_path).open("wb"))

    @staticmethod
    def info(path):
        path = Path(path)
        info = {}
        for frame_dir in path.iterdir():
            with (frame_dir / PKLStorage.pickle_path).open('rb') as f: 
                frame_data = pickle.load(f)
            for frame in frame_data:
                try:
                    img = Image.open(BytesIO(frame))
                    ik = str(img.size)
                    info[ik] = info.get(ik, 0) + 1
                except Exception as e:
                    print(f"reading failed for {frame_dir}: {e}")
        return info, img


class FileStorage(Storage):
    def __init__(self, path):
        super().__init__(path)

    def put(self, video_key, clip_tmp_dir, frame_files):
        save_dir = self.path / video_key
        save_dir.mkdir(exist_ok=True, parents=True)
        for ith_frame, (frame_id, frame_path) in enumerate(frame_files):
            shutil.copy(str((clip_tmp_dir / frame_path)), str(save_dir / "{:08d}.jpg".format(ith_frame)))

    @staticmethod
    def info(path):
        path = Path(path)
        info = {}
        for f in path.glob('**/*.jpg'):
            try:
                img = Image.open(f)
                ik = str(img.size)
                info[ik] = info.get(ik, 0) + 1
            except:
                print(f"reading failed for {f}")
        return info, img


STORAGE_TYPES = {
    "HDF5": HDF5Storage,
    "LMDB": LMDBStorage,
    "FILE": FileStorage,
    "PKL": PKLStorage
}
