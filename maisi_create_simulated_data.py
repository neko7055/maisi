import os
import logging
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.data import create_test_image_3d
from nibabel.affines import from_matvec


def get_random_affine(
        rotation_range=(-30, 30),  # 角度
        scale_range=(0.8, 1.2),  # 縮放比例
        translation_range=(-10, 10)  # 平移 (mm)
):
    """生成一個隨機的 3D 仿射矩陣"""

    # 1. 隨機旋轉 (分別繞 X, Y, Z 軸)
    def _rot_mat(axis, angle_deg):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rx = np.random.uniform(*rotation_range)
    ry = np.random.uniform(*rotation_range)
    rz = np.random.uniform(*rotation_range)

    rot_matrix = _rot_mat('x', rx) @ _rot_mat('y', ry) @ _rot_mat('z', rz)

    # 2. 隨機縮放 (Scaling)
    sx = np.random.uniform(*scale_range)
    sy = np.random.uniform(*scale_range)
    sz = np.random.uniform(*scale_range)
    scale_matrix = np.diag([sx, sy, sz])

    # 3. 組合線性變換部分 (旋轉 + 縮放)
    # 注意乘法順序影響結果，通常先縮放再旋轉
    linear_part = rot_matrix @ scale_matrix

    # 4. 隨機平移 (Translation)
    translation = np.random.uniform(*translation_range, size=3)

    # 5. 組合為 4x4 仿射矩陣
    # 使用 nibabel 工具函數將 3x3 矩陣與 1x3 向量組合成 4x4
    affine = from_matvec(linear_part, translation)

    return affine

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    logger = logging.getLogger("create_simulated_data")

    number_of_sim_data = 10

    sim_datalist = {"training": [{"src_image": f"train_src_image_{i:04d}.nii.gz",
                                  "tar_image": f"train_tar_image_{i:04d}.nii.gz",
                                  "modality": "ct"} for i in range(number_of_sim_data)],
                    "validation": [{"src_image": f"val_src_image_{i:04d}.nii.gz",
                                    "tar_image": f"val_tar_image_{i:04d}.nii.gz",
                                    "modality": "ct"} for i in range(number_of_sim_data)],
                    "test": [{"src_image": f"test_src_image_{i:04d}.nii.gz",
                              "tar_image": f"test_tar_image_{i:04d}.nii.gz",
                              "modality": "ct"} for i in range(number_of_sim_data)]
                    }

    sim_dim = (512, 512, 512)

    data_dir = "./sim_data_dir_2"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    dataroot_dir = os.path.join(data_dir, "data")
    os.makedirs(dataroot_dir, exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "test"), exist_ok=True)

    datalist_file = os.path.join(data_dir, "sim_datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(sim_datalist, f)

    logger.info("Created training data list.")
    datasave_dir = os.path.join(dataroot_dir, "training")
    for d in tqdm(sim_datalist["training"]):
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["src_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["tar_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)

    logger.info("Created validation data list.")
    datasave_dir = os.path.join(dataroot_dir, "validation")
    for d in tqdm(sim_datalist["validation"]):
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["src_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["tar_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)

    logger.info("Created test data list.")
    datasave_dir = os.path.join(dataroot_dir, "test")
    for d in tqdm(sim_datalist["test"]):
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["src_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(datasave_dir, d["tar_image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)

    logger.info("Generated simulated images.")