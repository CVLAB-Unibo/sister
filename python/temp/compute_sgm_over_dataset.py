import glob, os
import subprocess, sys

smatch_exe = '/home/daniele/work/workspace_python/sister/smatch/build/example'
max_disparity = 192
dataset_path = '/home/daniele/data/datasets/sister/v1/quality_fusion_scenes/bunch_of_nuts'
output_subfolder = 'output'

subfolders = glob.glob(os.path.join(dataset_path,"*"))
subfolders  = [x for x in subfolders if os.path.isdir(x)]
print(subfolders)

for s in subfolders:
    if not s.endswith("/"):
        s+="/"
    output_folder = os.path.join(s, output_subfolder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    command = "{} {} {}".format(
        smatch_exe,
        s,
        max_disparity
    )
    print(command)
    subprocess.call(command.split(" "))

