import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from subprocess import call, check_output

import h5py
import imagehash
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from storage import STORAGE_TYPES

# Derived from:
# - https://github.com/forwchen/vid2frame
# - https://github.com/jinyu121/video2frame/blob/master/video2frame.py
# - https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py
#

VIDEO_FORMATS = ['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']

def read_img(path):
    with open(path, 'rb') as f:
        return f.read()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="The video path (single file or dir)")
    # database and file options
    parser.add_argument("--db_name", type=str, required=True, help="The database to store extracted frames")
    parser.add_argument("--db_type", type=str, required=True, choices=["LMDB", "HDF5", "FILE", "PKL"], default="LMDB",
                        help="Type of the database")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Temporary folder")
    # resize options
    parser.add_argument("-s", "--short", type=int, default=0, help="Keep the aspect ration and scale the shorter side to s")
    parser.add_argument("-H", "--height", type=int, default=0, help="The resized height")
    parser.add_argument("-W", "--width", type=int, default=0, help="The resized width")
    # frame sampling options
    parser.add_argument("-k", "--skip", type=int, default=1, help="Only store frames with (ID-1) mod skip==0, frame ID starts from 1")
    parser.add_argument("-n", "--num_frame", type=int, default=-1, help="Uniformly sample n frames, this will override --skip")
    parser.add_argument("-r", "--interval", type=float, default=0, help="Extract one frame every r seconds")
    # similarity options
    parser.add_argument("-d", "--no_duplicates", type=float, default=0, help="Remove duplicates within percentage threshold similarity of another image")
    parser.add_argument("--hash_size", type=int, default=8, help="For duplicate detection the size the image will be resized to for comparison")
    parser.add_argument("--hash_alg", type=str, choices=["average_hash", "phash", "dhash", "dhash_vertical", "whash"], default="average_hash")
    
    args = parser.parse_args()

    # sanity check of the options
    if args.short > 0:
        assert args.height == 0 and args.width == 0
    if args.height > 0 or args.width > 0:
        assert args.height > 0 and args.width > 0 and args.short == 0
    assert(args.no_duplicates >= 0 and args.no_duplicates < 1.0)
    assert(args.hash_size > 0 and args.hash_size <= 1024)

    return args


def get_frame_rate(vid_path):
    call = ["ffprobe","-v", "quiet", "-show_entries", "stream=r_frame_rate", "-print_format", "json", str(vid_path)]
    output = subprocess.check_output(call)
    output = json.loads(output)
    r_frame_rate = 0
    if len(output.keys()) == 0:
        return r_frame_rate
    elif output['streams'] == []:
        return r_frame_rate

    for line in output['streams']:
        nums = line['r_frame_rate'].split('/')
        if float(nums[1]) == 0:
            continue
        frame_rate = 1.0*float(nums[0]) / float(nums[1])
        if frame_rate != 0:
            r_frame_rate = frame_rate

    return r_frame_rate


if "__main__" == __name__:
    args = parse_args()

    # temp dir
    args.tmp_dir = Path(args.tmp_dir)
    if args.tmp_dir.exists() and args.tmp_dir != Path('/tmp'):
        print(f"Warning: data may be deleted from {args.tmp_dir}")
    args.tmp_dir.mkdir(exist_ok=True)

    # video path
    args.video_path = Path(args.video_path).expanduser().resolve()
    if not args.video_path.exists():
        print(args.video_path, "does not exist")
        sys.exit(1)

    if args.video_path.is_dir():
        video_ext = set(VIDEO_FORMATS)
        all_videos = []
        unique_vids = set()
        for f in args.video_path.iterdir():
            if f.suffix in video_ext and (not (f.stem in unique_vids)):
                all_videos.append(f)
                unique_vids.add(f.stem)
    else:
        all_videos = [args.video_path]

    # handle existing db
    args.db_name = Path(args.db_name).expanduser().resolve()
    if args.db_name.exists():
        val = input(f"Remove existing database ({args.db_name})? (y/N) ")
        if val.upper() != 'Y':
            STORAGE_TYPES[args.db_type].display_info(args.db_name)
            sys.exit(1)
        else:
            try:
                if args.db_name.is_dir():
                    call(["rm", "-rf", str(args.db_name)])
                else:
                    args.db_name.unlink()
            except OSError as e:
                print(f"Error: {args.db_name} : {e}")
                sys.exit(1)
    
    # create frame db
    frame_db = STORAGE_TYPES[args.db_type](args.db_name)

    # scaling options
    if args.short == 0 and args.width == 0 and args.height == 0:
        vf_options = []
    elif args.short > 0:
        vf_options = ["-vf",
                    "scale='iw*1.0/min(iw,ih)*%d':'ih*1.0/min(iw,ih)*%d'" \
                            % (args.short, args.short)]
    elif args.height > 0 and args.width > 0:
        vf_options = ["-vf", "scale=%d:%d" % (args.width, args.height)]
    else:
        raise Exception('Invalid frame scale option')

    # sampling options
    if args.interval > 0:
        vf_options += ["-vsync","vfr"]
        assert args.num_frame <= 0 and args.skip == 1, \
                "No other sampling options are allowed when --interval is set"

    # similarity
    if args.no_duplicates > 0:
        similarity_threshold = 1. - args.no_duplicates
        diff_limit = round(similarity_threshold * (args.hash_size**2)) # image converted to hash_size x hash_size
        hash_algorithm = eval(f"imagehash.{args.hash_alg}")

    # process videos
    start_time = time.time()
    done_videos = set()
    vid_pbar = tqdm(all_videos, desc="video", unit="", ncols=64)
    for vid in vid_pbar:
        video_key = vid.stem
        
        if video_key in done_videos:
            print(f"video {video_key} seen before, ignored.")

        vid_pbar.set_description(str(vid), refresh=True)

        v_dir = args.tmp_dir / video_key
        call(["rm", "-rf", v_dir])
        v_dir.mkdir()    # caching directory to store ffmpeg extracted frames

        # sampling
        if args.interval > 0:
            r_frame_rate = get_frame_rate(vid)
            if r_frame_rate == 0:
                print(f"frame rate is 0, skip: {vid}")
                continue
            select_str = "select=not(mod(n\,%d))" % (int(round(args.interval*r_frame_rate)))
            # if already vf options then add this to them
            if '-vf' in vf_options:
                vf_index = vf_options.index('-vf')
                vf_options[vf_index + 1] = vf_options[vf_index + 1] + "," + select_str
            else:
                vf_options = ["-vf", select_str]

        call(["ffmpeg",
                "-loglevel", "panic",
                "-i", vid,
                ]
                + vf_options
                +
                [
                "-qscale:v", "2",
                str(v_dir / "%8d.jpg")])

        ids = [int(f.stem) for f in v_dir.iterdir()]
        sample = (args.num_frame > 0)
        if sample:
            sample_ids = set(list(np.linspace(min(ids), max(ids),
                                    args.num_frame, endpoint=True, dtype=np.int32)))

        hashes = []
        total_files = 0
        duplicates = 0
        def detect_duplicate(f:Path) -> bool:

            with Image.open(f) as img:
                hash1 = hash_algorithm(img, args.hash_size)
            for hash2 in hashes:
                if hash1 - hash2 <= diff_limit:
                    return True
            hashes.append(hash1)
            return False

        frame_files = []
        for f in tqdm(v_dir.iterdir(), desc="frame", unit="", total=len(ids), ncols=64):
            fid = int(f.stem)
            if sample:
                if fid not in sample_ids:
                    continue
            elif args.skip > 1:
                if (fid-1) % args.skip != 0:
                    continue
            total_files += 1
            # duplicate detection
            if args.no_duplicates > 0:
                if detect_duplicate(f):
                    duplicates +=1
                    continue
            frame_files.append((fid, f.name))

        frame_files.sort() # sort by fid
        frame_db.put(video_key, v_dir, frame_files)

        call(["rm", "-rf", v_dir])

        done_videos.add(video_key)
        if args.no_duplicates > 0:
            percent = float(duplicates) / float(total_files) * 100.
            print(f"{video_key}: {duplicates} duplicates removed ({percent:.1f}% of {total_files})")

    frame_db.close()
    frame_db.display_info()
    print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
