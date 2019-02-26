import os, glob, subprocess, argparse

leafsize_map = {
    'component_0J': 0.002,
    'component_1B': 0.002,
    'component_1G': 0.002,
    'hexa_nut': 0.0001,
    'hexa_screw': 0.0001,
    'washer': 0.0001,
    'arduino': 0.0001,
    'nodemcu': 0.0001,
}

selected = 'plane'

parser = argparse.ArgumentParser("Mesh 2 Clouds")
parser.add_argument("--leaf_size", help="Subsampling leaf size", default=0.002)
args = parser.parse_args()

models = sorted(glob.glob(os.path.join('models', '*')))
names = [os.path.basename(x) for x in models]

leafsize = args.leaf_size
mesh2pcd = "pcl_mesh2pcd {} {} -leaf_size {}"
pcd2ply = "pcl_pcd2ply {} {}"

for i in range(len(models)):
    print(names[i])
    if  names[i] == selected:
        model_path = os.path.join(models[i], names[i] + ".ply")
        cloud_path = os.path.join('clouds', names[i])
        if not os.path.exists(cloud_path):
            os.makedirs(cloud_path)
        cloud_path_ply = os.path.join('clouds', names[i], names[i] + ".ply")
        cloud_path_pcd = os.path.join('clouds', names[i], names[i] + ".pcd")
        print(model_path, cloud_path_pcd)

        command = mesh2pcd.format(model_path, cloud_path_pcd, leafsize)
        command2 = pcd2ply.format(cloud_path_pcd, cloud_path_ply)

        subprocess.call(command.split(' '))
        subprocess.call(command2.split(' '))
