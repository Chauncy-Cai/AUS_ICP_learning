import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
parser = argparse.ArgumentParser(description='visualizer')
parser.add_argument('--load', action="store_true", default=False)
parser.add_argument('--generate', action="store_true", default=False)
parser.add_argument('--period', type=int, default=100)
args = parser.parse_args()
if args.generate:
    print('generating point clouds.')
    args.load = True

frame = 0

def AngularDistance(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R = R1.dot(R2.T)
    theta = np.arccos(np.clip((np.sum(np.diag(R)) - 1.0)/2.0, -1, 1))
    theta = theta / np.pi * 180.0
    terr = np.linalg.norm(T1[:3, 3] - T2[:3, 3], 2)
    if (theta > 3) or (terr >= 0.3):
        return True
    else:
        return False

o3d.set_verbosity_level(o3d.VerbosityLevel.Debug)
if not args.load:
    pose_dict = {'intrinsics': [], 'extrinsics': []}

if args.load:
    print('reading poses.')
if args.load:
    import glob
    pose_files = glob.glob('stream/*.pose')
    num_frames = len(pose_files)
    frame_trace = [i for i in range(num_frames) if i % args.period == 0]
    #frame_trace += [i for i in range(450, 480, 2)]
    #frame_trace += [445]
    frame_trace = sorted(frame_trace)
else:
    num_frames = 100000
pcd_partial = o3d.PointCloud()

last_T = None

def custom_draw_geometry_with_view_tracking(mesh):
    def track_view(vis):
        global frame, poses
        global num_frames, last_T
        #print('frame=%d' % frame)
        #import ipdb; ipdb.set_trace()
        ctr = vis.get_view_control()
        ######intrinsics = o3d.read_pinhole_camera_intrinsic('intrinsics.json')
        #intrinsics = o3d.PinholeCameraIntrinsic(o3d.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)
        ######intrinsics = o3d.PinholeCameraIntrinsic(
        ######                o3d.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        ######intrinsics_mat=np.array([[935.30743608719376, 0.0, 0.0],
        ######                           [0.0, 935.30743608719376, 0.0],
        ######                           [959.5, 539.5, 1.0]])
        ######intrinsics.intrinsic_matrix = intrinsics_mat
        #import ipdb; ipdb.set_trace()
        ######print(intrinsics)
        #pcd = o3d.create_point_cloud_from_rgbd_image(rgbd, intrinsics)
        #o3d.write_point_cloud("stream/%d.ply" % frame, pcd)
        if frame == 0:
            #vis.get_render_option().load_from_json("render.json")
            intrinsics, extrinsics = ctr.convert_to_pinhole_camera_parameters()
            pose = [[-0.8188861922, 0.3889273405, -0.4220911372, -14.6068376600],
                    [-0.1157361687, -0.8321937190, -0.5422718444, 23.0477832143],
                    [-0.5621659395, -0.3952077147, 0.7264849060, 4.1193224787],
                    [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]]
            ctr.convert_from_pinhole_camera_parameters(intrinsics, pose)
        fid = frame % num_frames

        if not args.load:
            intrinsics, extrinsics = ctr.convert_to_pinhole_camera_parameters()
            o3d.write_pinhole_camera_intrinsic("stream/%d.intrinsics.json" % frame, intrinsics)
            with open('stream/%d.pose' % frame, 'w') as fout:
                for i in range(4):
                    for j in range(4):
                        if j > 0:
                            fout.write(' ')
                        fout.write('%.10f' % extrinsics[i, j])
                    fout.write('\n')
            #o3d.write_pinhole_camera_intrinsics(intrinsics, "stream/%d.intrinsics.json" % frame)
            #pose_dict['intrinsics'].append(intrinsics)
            #pose_dict['extrinsics'].append(extrinsics)
            #if frame == 100:
            #    import pickle
            #    with open('stream/pose.p', 'wb') as fout:
            #        pickle.dump(pose_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)
            #    exit()
        else:
            intrinsics = o3d.read_pinhole_camera_intrinsic("stream/%d.intrinsics.json" % fid)
            T = np.loadtxt('stream/%d.pose' % fid)
            ctr.convert_from_pinhole_camera_parameters(intrinsics, T)
            if args.generate:
                """ Generate Point Cloud """
                if (last_T is None) or (fid in frame_trace) or (AngularDistance(T, last_T)):
                    print('%d/%d' % (fid, num_frames))
                    #vis.update_geometry()
                    #vis.poll_events()
                    #vis.update_renderer()
                    depth = vis.capture_depth_float_buffer(False)
                    depth = np.array(depth)
                    #print(depth.max(), depth.min())
                    idx = np.where(depth > 30)
                    depth[idx] = 0
                    depth = o3d.Image(depth)
                    image = vis.capture_screen_float_buffer(False)
                    image = o3d.Image(np.array(np.array(image)*255).astype(np.uint8))
                    rgbd = o3d.create_rgbd_image_from_color_and_depth(
                                image, depth, convert_rgb_to_intensity = False)
                    rgbd.depth = o3d.Image(np.array(rgbd.depth)*1000)
                    pcd = o3d.create_point_cloud_from_rgbd_image(rgbd, intrinsics)
                    #pcd = o3d.voxel_down_sample(pcd, voxel_size=0.05)
                    #pcd.transform(np.linalg.inv(T))
                    o3d.write_point_cloud("stream/%d.ply" % fid, pcd, write_ascii=True)
                    cv2.imwrite("stream/%d.png" % fid, np.array(image))
                    last_T = T
                if fid == num_frames - 1:
                    exit()
        frame += 1

    o3d.draw_geometries_with_animation_callback([mesh, pcd_partial], track_view)


def main():
    mesh = o3d.read_triangle_mesh("ground_truth_surface/all.ply")
    custom_draw_geometry_with_view_tracking(mesh)

if __name__ == "__main__":
    main()
