### Export a RAW DATASET

Example:
```
 python export_raw_dataset.py --path /home/daniele/data/datasets/sister/v1/raw_datasets/acquisitions_alpha/arduino --output_folder ${OUTPUT_FOLDER} --baselines 002 010 025 050 100 120
```



### Generate **OUTPUTs** in subfolders "output" of each frame

Each output has a unique tag. For example "classical_multiview" is the output generate di with classical SGM over cross configuration.

To generate SGM model-based output:

```
python compute_sgm_over_dataset.py
```

### Create resulting *clouds*

For each OUTPUT, and therefore its associated tag (e.g. "classical_multiview") 
t is necessary to generate an output PointCloud. With the following command, 
a point cloud transformed in the camera frame is generated:

```
python test_frame_batch.py --path /home/daniele/data/datasets/sister/v1/objects_full_scenes/component_0J --tag classical_multiview
```

In "test_frame_batch.py" is there a SCALING_FACTOR_MAP needed to associated
a rescaling factor per TAG (or sub tag). E.g. for each tag containing "classical" 
it is necessary to perform a (1/256) rescale of disparity.

### Generate per-model Cad transformation

It is important to compute the Pose of a Cad Model into the corresponding scene.
(Model RF is centered in the CAD Model, while Model RF in scene has to be computed).

1. Open Meshlab and a reference scene point cloud, generated with previous approach
(be sure to open a scene with minimal reconstruction error)
2. Import as Mesh the Point Cloud of the corresponding model
3. Roto-Translate in order to match the model-in-scene
4. Save the Meshlab Project in a temp file
5. Open the Meshlab Project with a text editor and copy Model new Roto-Translation
6. Save the rototranslation (4 rows and 4 columns ' ' spaced) in the Model Dataset folder as "object_pose.txt"

#### To generate Model Clouds..

Use PCL tools to convert CAD model to Point cloud (given that your cad model in STL format is 
converted in PLY mesh with Meshlab):

E.g.:

```bash
pcl_mesh2pcd ${DATASET_PATH}/models/component_1B/component_1B.ply ${DATASET_PATH}/clouds/component_1B/component_1B.pcd -leaf_size 0.002
```

Then convert it to PLY point cloud:

```bash
pcl_pcd2ply ${DATASET_PATH}/clouds/component_1B/component_1B.pcd ${DATASET_PATH}/clouds/component_1B/component_1B.ply
```
