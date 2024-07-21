import os
from os.path import join as pjoin
import glob

def dump_keypoints(rgb_folder):
    import sys
    import json
    import cv2
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import torch
    from torchvision import transforms
    sys.path.insert(0, "/home/sangjun/Desktop/juntae/juntae/foot_keypoint")
    from model import KeypointResNet
    num_points = 12
    model = KeypointResNet(num_points).to("cuda")
    #checkpoint = "/home/sangjun/Desktop/juntae/juntae/foot_keypoint/runs/foot_keypoint_experiment_20240707-103930/best_model.pth"
    #checkpoint = "/home/sangjun/Desktop/juntae/juntae/foot_keypoint/runs/foot_keypoint_experiment_20240707-162859/best_model.pth"
    checkpoint = "/home/sangjun/Desktop/juntae/juntae/foot_keypoint/runs/foot_keypoint_experiment_20240712-012705/best_model.pth"
    model = load_checkpoint(checkpoint, model)
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def denormalize_keypoints(keypoints, width, height):
        """
        Denormalizes the normalized keypoints back to the original image dimensions.

        :param keypoints: List of normalized keypoints.
        :param width: Width of the transformed image.
        :param height: Height of the transformed image.
        :return: List of denormalized keypoints.
        """
        denormalized_keypoints = [
            k * width if i % 2 == 0 else k * height
            for i, k in enumerate(keypoints)
        ]
        return denormalized_keypoints



    input_width, input_height = 480, 640
    mean, std= np.array([[0.485, 0.456, 0.406]]), np.array([0.229, 0.224, 0.225])
    
    os.makedirs(pjoin(os.path.dirname(rgb_folder), "kp_debug"), exist_ok=True)

    res = {"kp_labels": ["big toe", "2nd toe", "3rd toe", "4th toe", "little toe", "heel", "outer extrema", "inner extrema", "lower heel", "arch1", "arch2", "arch3"], "annotations": {}}

    with torch.no_grad():
        for image_path in glob.glob(pjoin(rgb_folder, "*")):
            image = Image.open(image_path).convert('RGB')
            w, h = image.size
            image_torch = val_transform(image)
            image_torch = image_torch.unsqueeze(0).to("cuda")
    
            all_outputs = model(image_torch)
            all_outputs = all_outputs.detach().cpu()
            pred_coords = all_outputs[..., :2]
            log_var = all_outputs[..., 2:]
            var = np.exp(log_var)
            confidence_interval = 1.96 * np.sqrt(var)
            keypoint = denormalize_keypoints(pred_coords, 224, 224)
            keypoint = keypoint[0].reshape(-1, 2).detach().numpy().tolist()
            keypoint_arr = np.array(keypoint)
            print(image_path, keypoint_arr.max(), keypoint_arr.min())
            variance = denormalize_keypoints(confidence_interval, 224, 224)
            variance = variance[0].reshape(-1, 2).detach().numpy().tolist()
    
            res['annotations'][os.path.basename(image_path)] = {
                    'kps': keypoint,
                    'vis':[0.9982098340988159, 0.998548686504364, 0.998916506767273, 0.9991087317466736,
                        0.9961655139923096, 0.9999371767044067, 0.9961687922477722, 0.999933123588562,
                        0.9999364614486694, 0.9999363422393799, 0.9999381303787231, 0.9999349117279053],
                    'variance': variance,
                        }


            # img = cv2.imread(image_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (224, 224))
            # img = img/255.0
            # mean=[0.485, 0.456, 0.406]
            # std=[0.229, 0.224, 0.225]
            # 
            # img[..., 0] -= mean[0]
            # img[..., 1] -= mean[1]
            # img[..., 2] -= mean[2]
            # 
            # img[..., 0] /= std[0]
            # img[..., 1] /= std[1]
            # img[..., 2] /= std[2]
            # print(np.min(img), np.max(img))
            # print(image_path)
            # #img = Image.open(image_path).convert("RGB").resize(size=(input_width, input_height), resample=Image.BILINEAR)
            # #img = np.array(img).astype(np.float32) / 255.0
            # #img = (img - mean) / std
            # img = torch.from_numpy(img).permute(2, 0, 1).to("cuda").float().unsqueeze(0)
            # output = model(img)
            # output = output.detach().cpu().squeeze().numpy()
            # print(output)
            # #res["annotations"][os.path.basename(image_path)] = {"kps": [[float(output[2 * i]) * input_width, float(output[2 * i + 1]) * input_height] for i in range(num_points)],
            # res["annotations"][os.path.basename(image_path)] = {"kps": [[float(output[i][0]) * 224, float(output[i][1]) * 224] for i in range(num_points)],
            #                                                     "vis": [0.99 for i in range(num_points)],
            #                                                     "variance": [[5, 5] for i in range(num_points)]}
            image = mpimg.imread(image_path)
            pts = np.array(res["annotations"][os.path.basename(image_path)]["kps"])
            plt.imshow(image)
            #plt.plot(640, 570, "og", markersize=10)  # og:shorthand for green circle
            plt.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=200)
            target_path = pjoin(os.path.dirname(rgb_folder), "kp_debug", os.path.basename(image_path))
            plt.savefig(target_path)
            plt.close()
    print(res)
    save_path = pjoin(os.path.dirname(rgb_folder), "keypoints.json")
    print(save_path)
    with open(save_path, "w") as fout:
        json.dump(res, fout)

    del sys.path[0]

def load_checkpoint(fpath, model):
    import torch
    ckpt = torch.load(fpath, map_location='cpu')
    if "model" in ckpt:
        ckpt = ckpt["model"]

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

def dump_norm(rgb_folder):
    import sys
    import cv2
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    sys.path.insert(0, "/home/sangjun/soon/surface_normal_uncertainty")
    from models.NNET import NNET

    dump_path = pjoin(os.path.dirname(rgb_folder), "norm")

    class Args:
        architecture = "GN"
        pretrained = True
        sampling_ratio = 0.4
        importance_ratio = 0.7
        input_height = 640
        input_width = 480
    args = Args()
    model = NNET(args).to("cuda")
    checkpoint = "/home/sangjun/soon/surface_normal_uncertainty/experiments/exp02_test/models/checkpoint_iter_0000200000.pt"
    model = load_checkpoint(checkpoint, model)
    model.eval()

    alpha_max = 60
    kappa_max = 30

    os.makedirs(pjoin(os.path.dirname(rgb_folder), "norm"), exist_ok=True)
    os.makedirs(pjoin(os.path.dirname(rgb_folder), "norm_unc"), exist_ok=True)

    mean, std= np.array([[0.485, 0.456, 0.406]]), np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for image_path in glob.glob(pjoin(rgb_folder, "*")):
            print(image_path)
            img = Image.open(image_path).convert("RGB").resize(size=(args.input_width, args.input_height), resample=Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = torch.from_numpy(img).permute(2, 0, 1).to("cuda").float().unsqueeze(0)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

            target_path = pjoin(os.path.dirname(rgb_folder), "norm", os.path.basename(image_path))
            print(target_path)
            plt.imsave(target_path, pred_norm_rgb[0, :, :, :])

            target_path = pjoin(os.path.dirname(rgb_folder), "norm_unc", os.path.basename(image_path))
            print(target_path)
            #plt.imsave(target_path, pred_kappa[0, :, :, 0], vmin=0.0, vmax=kappa_max, cmap='gray')
            cv2.imwrite(target_path, pred_kappa[0, :, :, 0])

            #pred_alpha = utils.kappa_to_alpha(pred_kappa)
            #target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
            #plt.imsave(target_path, pred_alpha[0, :, :, 0], vmin=0.0, vmax=alpha_max, cmap='jet')
        

    del sys.path[0]

def dump_obj_to_png(path):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    def frustum(left, right, bottom, top, znear, zfar):
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = +2.0 * znear / (right - left)
        M[1, 1] = +2.0 * znear / (top - bottom)
        M[2, 2] = -(zfar + znear) / (zfar - znear)
        M[0, 2] = (right + left) / (right - left)
        M[2, 1] = (top + bottom) / (top - bottom)
        M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
        M[3, 2] = -1.0
        return M
    def perspective(fovy, aspect, znear, zfar):
        h = np.tan(0.5*np.radians(fovy)) * znear
        w = h * aspect
        return frustum(-w, w, -h, h, znear, zfar)
    def translate(x, y, z):
        return np.array([[1, 0, 0, x], [0, 1, 0, y],
                         [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
    def xrotate(theta):
        t = np.pi * theta / 180
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                         [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)
    def yrotate(theta):
        t = np.pi * theta / 180
        c, s = np.cos(t), np.sin(t)
        return  np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                          [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)
    V, F = [], []
    with open(path) as f:
        for line in f.readlines():
            if line.startswith('#'):  continue
            values = line.split()
            if not values:            continue
            if values[0] == 'v':      V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f' :   F.append([int(x) for x in values[1:4]])
    V, F = np.array(V), np.array(F)-1
    V = (V-(V.max(0)+V.min(0))/2) / max(V.max(0)-V.min(0))
    MVP = perspective(25,1,1,100) @ translate(0,0,-3.5) @ xrotate(20) @ yrotate(45)
    V = np.c_[V, np.ones(len(V))]  @ MVP.T
    V /= V[:,3].reshape(-1,1)
    V = V[F]
    T =  V[:,:,:2]
    Z = -V[:,:,2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z-zmin)/(zmax-zmin)
    C = plt.get_cmap("magma")(Z)
    I = np.argsort(Z)
    T, C = T[I,:], C[I,:]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
    collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
    ax.add_collection(collection)
    #plt.show()
    plt.savefig(path + ".png")

def dump_colmap_json(rgb_folder):
    import os
    import json
    import numpy as np
    def read_cameras_txt(file_path):
        cameras = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # assume that feature_extractor runs with num_cameras=1
            # 1 SIMPLE_RADIAL 480 640 490.68061424515838 240 320 0.012445952264788228
            # "camera": {"width": 480, "height": 640, "f": 490.8561322243916, "cx": 240.0, "cy": 320.0, "k": 0.012663188101326417}
            for line in lines:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                params = list(map(float, parts[4:]))
                cameras[camera_id] = params
                res = {
                    "width": int(parts[2]),
                    "height": int(parts[3]),
                    "f": float(parts[4]),
                    "cx": int(parts[5]),
                    "cy": int(parts[6]),
                    "k": float(parts[7]),
                }
                return res
    def read_images_txt(file_path):
        images = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    continue
                line = line.strip()
                if len(line) > 0:
                    parts = line.strip().split()
                    try:
                        image_id = int(parts[0])
                    except:
                        continue
                    qvec = list(map(float, parts[1:5]))
                    tvec = list(map(float, parts[5:8]))
                    camera_id = int(parts[8])
                    image_name = parts[9]
                    images[image_id] = {
                        'qvec': qvec,
                        'tvec': tvec,
                        'camera_id': camera_id,
                        'image_name': image_name
                    }
        return images
    def qvec2rotmat(qvec):
        return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
    def convert_to_json(cameras_file, images_file, output_file):
        camera = read_cameras_txt(cameras_file)
        images = read_images_txt(images_file)
        res = {
            "camera": camera,
            "images": []
        }
        data = []
        for image_id, image in images.items():
            qvec = image['qvec']
            tvec = image['tvec']
            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(qvec)
            w2c[:3, 3] = tvec
            c2w = np.linalg.inv(w2c)

            R, T = c2w[:3, :3], c2w[:3, 3:]
            R = np.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

            new_c2w = np.concatenate([R, T], 1)
            w2c = np.linalg.inv(np.concatenate((new_c2w, np.array([[0,0,0,1]])), 0))
            R, T = w2c[:3, :3], w2c[:3, 3] # convert R to row-major matrix
            R, T = R.tolist(), T.tolist() # data.py transpose R
            #R = qvec2rotmat(qvec).tolist()
            C = tvec
            #T = (-np.array(R).T @ np.array(C)).tolist()
            res["images"].append({
                "image_id": image_id,
                "pth": image['image_name'],
                "R": R,
                "C": C,
                "T": T
            })
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)

    prj_folder = os.path.dirname(rgb_folder)
    cameras_path = pjoin(prj_folder, "colmap/sparse_text/0/cameras.txt")
    images_path = pjoin(prj_folder, "colmap/sparse_text/0/images.txt")
    #cameras_path = pjoin(prj_folder, "colmap/sparse/0/cameras.bin")
    #images_path = pjoin(prj_folder, "colmap/dense/sparse/images.bin")
    #from read_write_model import read_cameras_binary, read_images_binary
    #cameras = read_cameras_binary(cameras_path)
    #images = read_images_binary(images_path)
    #print(cameras)
    #print(images)
    output_path = pjoin(prj_folder, "colmap.json")
    convert_to_json(cameras_path, images_path, output_path)
    print(f"JSON data saved to {output_path}")


def dump_skin_color():
    pass

def dump(rgb_folder):
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    input_width, input_height = 480, 640
    os.makedirs(pjoin(os.path.dirname(rgb_folder), "rgb"), exist_ok=True)
    for image_path in glob.glob(pjoin(rgb_folder, "*")):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (input_width, input_height))
        print(image_path)
        target_path = pjoin(os.path.dirname(rgb_folder), "rgb", os.path.basename(image_path))
        cv2.imwrite(target_path, resized_image)



def test():
    #rgb_folder = "data/scans/test_image/rgb"
    #rgb_folder = "data/scans/test_image/rgb"
    rgb_folder = "data/scans/test4/rgb"
    dump_colmap_json(rgb_folder)
    dump_norm(rgb_folder)
    dump_keypoints(rgb_folder)
def test1():
    rgb_folder = "data/scans/0048_a/rgb"
    dump_norm(rgb_folder)
def test2():
    rgb_folder = "data/scans/0048_b/rgb"
    dump_keypoints(rgb_folder)
def test3():
    rgb_folder = "data/scans/0048_c/rgb"
    dump_colmap_json(rgb_folder)
def test4():
    dump_obj_to_png("exp/0048/fit_00.obj")
    dump_obj_to_png("exp/0048/fit_01.obj")
    dump_obj_to_png("exp/0048_a/fit_00.obj")
    dump_obj_to_png("exp/0048_a/fit_01.obj")
    dump_obj_to_png("exp/0048_b/fit_00.obj")
    dump_obj_to_png("exp/0048_b/fit_01.obj")
    dump_obj_to_png("exp/0048_c/fit_00.obj")
    dump_obj_to_png("exp/0048_c/fit_01.obj")



if __name__ == "__main__":
    test()
    #dump("data/scans/test3/rgb_raw")
