import argparse
import os
import glob
import open3d as o3d
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def clean_mesh(mesh, connected_thr=1000):
    print('Clustering connected triangles')
    label_per_tri, num_tri_per_label, _ = mesh.cluster_connected_triangles()
    label_per_tri = np.array(label_per_tri)
    num_tri_per_label = np.array(num_tri_per_label)
    labels = np.arange(0, num_tri_per_label.shape[0])

    print('Generating mask')
    valid_labels = labels[num_tri_per_label >= connected_thr]
    valid_mask = np.zeros(label_per_tri.shape)
    for i, label in enumerate(valid_labels):
        valid_mask = np.logical_or(valid_mask, label_per_tri == label)

    print('Removing duplicate')
    mesh.remove_triangles_by_mask(np.logical_not(valid_mask))
    mesh.remove_unreferenced_vertices()
    return mesh

def interpolate_trajectory(traj, sample_step, interp_factor):
    '''
    get a sample ever \sample_step.
    interp_factor * num of inital nodes = final nodes
    '''
    intrinsic = traj.parameters[0].intrinsic

    print('Reading trajectory')
    position_anchors = []
    rotation_anchors = []
    N = len(traj.parameters)
    for i in range(0, N, sample_step):
        extrinsic = traj.parameters[i].extrinsic
        position_anchors.append(extrinsic[:3, 3])
        rotation_anchors.append(extrinsic[:3, :3])
    position_anchors = np.vstack(position_anchors).T
    rotation_anchors = R.from_matrix(rotation_anchors)

    t_fine = np.linspace(0, 1, int(N // interp_factor))

    print('Interpolating rotations')
    slerp = Slerp(np.linspace(0, 1, N // sample_step), rotation_anchors)
    rotation_interp = slerp(t_fine)

    # Interpolate positions
    print('Interpolating translations')
    tck, u = interpolate.splprep(position_anchors, s=2)
    position_interp = interpolate.splev(t_fine, tck)
    position_interp = np.array(position_interp) # (3, N)

    # Write
    print('Writing trajectory')
    params = []
    for i in range(0, position_interp.shape[1]):
        param = o3d.camera.PinholeCameraParameters()
        extrinsic = np.identity(4)
        extrinsic[:3, 3] = position_interp[:, i]
        extrinsic[:3, :3] = rotation_interp[i].as_matrix()

        param.intrinsic = intrinsic
        param.extrinsic = extrinsic
        params.append(param)

    traj_interp = o3d.camera.PinholeCameraTrajectory()
    traj_interp.parameters = params
    return traj_interp


def custom_draw_geometry_with_camera_trajectory(mesh, traj):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.traj = traj
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            pass
            # print("Capture image {:05d}".format(glb.index))
            # depth = vis.capture_depth_float_buffer(False)
            # image = vis.capture_screen_float_buffer(False)
            # plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
            #         np.asarray(depth), dpi = 1)
            # plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
            #         np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.traj.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.traj.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, help='Mesh filename')
    parser.add_argument('--clean_mesh', action='store_true')
    parser.add_argument('--traj', type=str,
                        help=".log trajectory file or irectory that contains several .json files")
    parser.add_argument('--interp_traj', action='store_true')
    parser.add_argument('--sample_step', type=int, default=1)
    parser.add_argument('--interp_factor', type=float, default=0.01)
    args = parser.parse_args()

    # Mesh preprocessing
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if args.clean_mesh:
        mesh = clean_mesh(mesh)
        o3d.io.write_triangle_mesh('filtered_mesh.ply', mesh)

    # Trajectory loading and preprocessing
    if os.path.isdir(args.traj):
        jsons = sorted(glob.glob('{}/*.json'.format(args.traj)))

        params = []
        for json in jsons:
            params.append(o3d.io.read_pinhole_camera_parameters(json))
        traj = o3d.camera.PinholeCameraTrajectory()
        traj.parameters = params
    elif args.traj.endswith('.log') or args.traj.endswith('.json'):
        traj = o3d.io.read_pinhole_camera_trajectory(args.traj)
    else:
        raise Exception('Unsupported traj format: {}', args.traj)

    if args.interp_traj:
        traj = interpolate_trajectory(traj, args.sample_step, args.interp_factor)
        o3d.io.write_pinhole_camera_trajectory('interp_traj.log', traj)

    custom_draw_geometry_with_camera_trajectory(mesh, traj)
