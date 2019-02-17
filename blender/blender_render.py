import bpy
import argparse
import os
from math import pi
from mathutils import Vector
from mathutils import Euler
import math
import sys

if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser(description="Sister Blender Scene Building Tool")
    parser.add_argument("-sensor_width","--sensor_width", dest='sensor_width', type=float, default=32,  help="sensor width in mm")

    parser.add_argument("-focal_x", "--focal_x", dest='focal_x', type=float, default=528.580921, help="camera focal lenght in mm or pixel")
    parser.add_argument("-focal_y", "--focal_y", dest='focal_y', type=float, default=528.000913, help="camera focal lenght in mm or pixel")

    parser.add_argument('-focal_in_mm', '--focal_in_mm', dest='focal_in_mm', action='store_true', help='consider focal lenght in mm, default in pixel')
    parser.set_defaults(focal_in_mm=False)

    parser.add_argument('-extrinsics', '--extrinsics', dest='extrinsics', type=float, nargs='+', default=[0.017586899921298027, -0.09221349656581879, 0.30974799394607544, 0.017355073243379593, 0.021796815097332, -1.5809646844863892], help='extrinscs in format Tx Ty Tz Rx Ry Rz')

    parser.add_argument('-resolution', '--resolution', dest='resolution', type=int, nargs='+', default=[640, 480], help='resolution of images format: width height')

    parser.add_argument("-output_path","--output_path", type=str, default="./output", help="where output files will be stored")

    parser.add_argument('-save_depth', '--save_depth', dest='save_depth', action='store_true', help='save rgb render')
    parser.set_defaults(save_depth=False)
    parser.add_argument('-save_rgb', '--save_rgb', dest='save_rgb', action='store_true', help='save depth render')
    parser.set_defaults(save_rgb=False)

    args=parser.parse_known_args(argv)[0]

def set_camera_params(focal_mm, sens_w, T, R, name='Camera'):
    cam = bpy.data.cameras[name]
    # Lens
    cam.type = 'PERSP'
    cam.lens_unit = 'MILLIMETERS'
    cam.lens = focal_mm
    cam.draw_size = 0.1
    cam.sensor_width = sens_w

    cam_obj = bpy.data.objects[name]
    cam_obj.location = Vector((T[0],T[1],T[2]))
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = Euler((R[0],R[1],R[2]),'XYZ')

def set_scene_render_params(res_x, res_y, output_path, name_scene="Scene"):
    bpy.data.scenes[name_scene].render.resolution_x = res_x
    bpy.data.scenes[name_scene].render.resolution_y = res_y
    bpy.data.scenes[name_scene].render.image_settings.color_mode='RGB'
    bpy.data.scenes[name_scene].render.resolution_percentage = 100
    bpy.data.scenes[name_scene].render.filepath = os.path.join(output_path,"Image","Image")
    bpy.data.scenes[name_scene].frame_current = 0

def set_node_tree(output_path, save_depth, name_scene="Scene"):
    #init tree
    bpy.data.scenes[name_scene].use_nodes = True
    tree =  bpy.data.scenes[name_scene].node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    #render layers
    rl = tree.nodes.new('CompositorNodeRLayers')      
    rl.location = 0,0

    comp=tree.nodes.new('CompositorNodeComposite')  
    comp.location= 0,200 
    tree.links.new(rl.outputs["Image"],comp.inputs["Image"])
    
    if save_depth:
        of = tree.nodes.new('CompositorNodeOutputFile')      
        of.location = 200,0
        of.format.file_format = 'OPEN_EXR'
        of.name = "FileOutputDepth"
        of.format.color_mode = "RGB"
        tree.links.new(rl.outputs["Depth"],of.inputs["Image"])
        #depthpath
        of.base_path = os.path.join(output_path,"Depth")
        of.file_slots[0].path = ''

width = args.resolution[0]
height = args.resolution[1]
translation_camera = args.extrinsics[:3]
rotation_camera = args.extrinsics[3:]

### suppose same same ratio width/height bewteen sensor and resolution
if args.focal_in_mm:
    focalx_in_pixel = args.focal_x * width / args.sensor_width
    focalx_in_mm = args.focal_x
    focaly_in_pixel = args.focal_y * width / args.sensor_width
    focaly_in_mm = args.focal_y
else:
    focalx_in_pixel = args.focal_x
    focalx_in_mm = args.focal_x * args.sensor_width / width
    focaly_in_pixel = args.focal_y
    focaly_in_mm = args.focal_y * args.sensor_width / width

set_scene_render_params(width, height, args.output_path)
set_camera_params(focalx_in_mm, args.sensor_width, translation_camera, rotation_camera)
set_node_tree(args.output_path, args.save_depth)

bpy.ops.render.render(write_still=args.save_rgb)