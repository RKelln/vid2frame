# vid2frame
An easy-to-use tool to extract frames from video and store into database.
Basically, this is a python wrapper of ffmpeg which additionally stores the frames into database. 
Additionally, some basic capabilities for detecting duplicates and deduplication has been added
such that you can quite easily extract 100 different frames from a video, or if you scrape
a large number of images from the net you can resize and dedupe before placing them into an image database.

## This is based on others work

This is a fork and combination of a number of projects:
* https://github.com/forwchen/vid2frame
* https://github.com/jinyu121/video2frame/blob/master/video2frame.py
* https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py


## Why this tool
* Extracting frames from large video datasets (usually 10k ~ 100k, hundreds of GBs on disk) is tedious, automate it.
* Storing millions of frames on disk makes subsequent processing SLOW.
* Common mistakes I once made:
    * Decode all frames (using scikit-video) and store them into a **LARGE** .npy file, nice way to blow up the disk.
    * Extract all frames using ffmpeg and write to disk. Takes **foreeeeever** to move or delete.
    * Extract JPEG frames using ffmpeg but ignores the JPEG **quality**. For deep learning and computer vision, a good quality of images (JPEG quality around 95) is required. 

* Good practice in my opinion:
    * Add `-qscale:v 2` to [ffmpeg](https://stackoverflow.com/questions/10225403/how-can-i-extract-a-good-quality-jpeg-image-from-an-h264-video-file-with-ffmpeg) command.
    * Store extracted frames into a database, [LMDB](https://lmdb.readthedocs.io/en/release/) or [HDF5](http://docs.h5py.org/en/stable/).
    * (Optional) Use [Tensorpack dataflow](https://tensorpack.readthedocs.io/modules/dataflow.html) to accelerate reading from the database.
    * Suggestions are welcome.

## Usage

### Install

Conda (Linux):
```bash
conda env create -f environment.yml
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Extract frames for videos in a specific split using `vid2frame.py`
```
usage: vid2frame.py [-h] --db_name DB_NAME --db_type {LMDB,HDF5,FILE,PKL}
                    [--tmp_dir TMP_DIR] [-s SHORT] [-H HEIGHT] [-W WIDTH]
                    [-k SKIP] [-n NUM_FRAME] [-r INTERVAL] [-d NO_DUPLICATES]
                    [--hash_size HASH_SIZE]
                    [--hash_alg {average_hash,phash,dhash,whash}]
                    video_path

positional arguments:
  video_path            The video path (single file or dir)

optional arguments:
  -h, --help            show this help message and exit
  --db_name DB_NAME     The database to store extracted frames
  --db_type {LMDB,HDF5,FILE,PKL}
                        Type of the database
  --tmp_dir TMP_DIR     Temporary folder
  -s SHORT, --short SHORT
                        Keep the aspect ration and scale the shorter side to s
  -H HEIGHT, --height HEIGHT
                        The resized height
  -W WIDTH, --width WIDTH
                        The resized width
  -k SKIP, --skip SKIP  Only store frames with (ID-1) mod skip==0, frame ID
                        starts from 1
  -n NUM_FRAME, --num_frame NUM_FRAME
                        Uniformly sample n frames, this will override --skip
  -r INTERVAL, --interval INTERVAL
                        Extract one frame every r seconds
  -d NO_DUPLICATES, --no_duplicates NO_DUPLICATES
                        Remove duplicates within threshold of another image
  --hash_size HASH_SIZE
                        For duplicate detection the size the image will be
                        resized to for comparison
  --hash_alg {average_hash,phash,dhash,whash}
```

#### Notes
* The frames will be stored as strings of their binary content, i.e. they are NOT decoded. Both LMDB and HDF5 are key-value storage, the keys are in the format of `video_name/frame_id` (assuming there are no two videos with the same name).
* The frames are in JPEG format, with JPEG quality ~95. Note the `-qscale:v 2` option in `vid2frame.py`. This is **important** for subsequent deep learning tasks.
* The database to use is either LMDB or HDF5, choose one according to:
    * Reading from HDF5 is convenient, if you do not plan to use [Tensorpack](https://tensorpack.readthedocs.io/_modules/tensorpack/dataflow/format.html#HDF5Data), which does not support HDF5 well currently, always choose HDF5.
    * LMDB integrates better with [Tensorpack](https://tensorpack.readthedocs.io/modules/dataflow.html#tensorpack.dataflow.LMDBData), but reading from it is less flexible (though much much faster than HDF5).
* Resizing options (exclusive):
    1. Resize the shorter edge and keep aspect ratio (the longer edge adapts) (--short)
    2. Resize to specific height & width (--height --width)
* Sampling options (exclusive):
    1. Keep one of frame every `k` frames (default 1, i.e. keep every frame) (--skip)
    2. Uniformly sample `n` frames (--num_frame). For example: If there are 10 frames, --skip=2 will sample frames 1,3,5,7,9 and --num_frame=4 will sample frames 1,4,7,10.
    3. Sample one frame every `r` seconds (--interval) or 1/r FPS. For r==1, its 1 FPS, and r==2, its 0.5 FPS.
* Duplicate removal options:
    * Requires testing the two parameters: `no_duplicates` and `hash_size` based on video size and frame similarity
    * For 1920x1080 (HD) video with minimal duplicate removal try `no_duplicates=0.99` and `hash_size=32`
    * For aggressive HD resolution removal try `no_duplicates=0.98` and `hash_size=8`
    * Tip: try exporting to FILE type and then using ffmpeg to make a video (at 25 fps):
        * `ffmpeg path/to/frames/%08d.jpg -r 25 test_frames.mp4`
* Video files are identified with extensions, currently recognizing `['.mp4', '.avi', '.flv', '.mkv', '.webm', '.mov']`.
* Videos with the same name (without extension) are considered duplicates. Only one of them will be processed.

### Usage examples

#### Basic usage:
```
python vid2frame.py path/to/my/video.mp4 --db_name my_db.lmdb --db_type LMDB
```

#### Resize:
```
python vid2frame.py path/to/my/video.mp4 --db_name my_frames --db_type FILE -W 512 -H 512
```

#### Remove duplicates:
```
python vid2frame.py path/to/my/video.mp4 --db_name my_frames --db_type FILE -W 512 -H 512 --no_duplicates 0.98 --hash_size 32
```


### (Optional) Test reading from database using `test_db.py`
`test_db.py` provides sample code to iterate, read and decode frames in databases, it also checks for broken images. 
```
usage: test_db.py [-h] [--db_name DB_NAME] [--db_type {LMDB,HDF5,FILE,PKL}]

optional arguments:
  -h, --help            show this help message and exit
  --db_name DB_NAME     The database to store extracted frames
  --db_type {LMDB,HDF5,FILE,PKL}
                        Type of the database
```

#### Note
* Opening images from string buffer: `img = Image.open(BytesIO(v))`
* Reading string from HDF5 db: `s = np.asarray(db_vid[fid]).tostring()`

#### Sample usage
`python test_db.py --db_name frames-1.lmdb --db_type LMDB`

The script outputs the number of frames in the database and their sizes. As well as showing the last frame in the db and the time to iterate over whole database.

## Dependencies
* Python 3.7
* FFmpeg: Install on [Ubuntu](https://tecadmin.net/install-ffmpeg-on-linux/). Other [platforms](https://www.google.com/).
* Python libraries: `pip install -r requirements.txt`, 


## Common issues
* `RuntimeError: Unable to create link (name already exists)`

   This is caused by writing duplicate frames to a non-empty HDF5 database.
