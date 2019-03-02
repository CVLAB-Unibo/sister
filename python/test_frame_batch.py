#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import glob
from sister.sister import Utilities, Camera
import subprocess
import argparse

tags_scale_map = {
    "classical": 256.,
    "mcnn": 1.
}

parser = argparse.ArgumentParser("Test a Circular Frame")
parser.add_argument("--camera_file", help="Camera parameters filename", type=str, default='/home/daniele/work/workspace_python/sister/data/cameras/usb_camera.xml')
parser.add_argument("--path", help="Datasete path", type=str, required=True)
parser.add_argument("--tag", help="Tag ", type=str, required=True)
parser.add_argument("--level", help="Subfolders depth", type=int, default=1)
args = parser.parse_args()


# Camera
camera = Camera(filename=args.camera_file)

path = args.path
for i in range(args.level):
    path = os.path.join(path,"*")

subfolders = glob.glob(path)
print(subfolders)

command = "python test_frame.py --path {} --tag {} --visualization_type pcd --filter_level 0 --export True --plane_level 100"
for s in subfolders:
    ccommand = command.format(
        s,
        args.tag
    )
    print(ccommand)
    subprocess.call(ccommand.split(" "))
