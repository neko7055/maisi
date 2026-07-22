import os
import glob
import numpy as np
import nibabel as nib

from totalsegmentator.python_api import totalsegmentator


class ApplyCommonMaskd:
    def __init__(self, keys, threshold=-1000, fill_value=-1000):
        self.keys = keys
        self.threshold = threshold
        self.fill_value = fill_value

    def __call__(self, data):
        d = dict(data)
        mask = np.logical_and.reduce(
            [d[key] > self.threshold for key in self.keys]
        )
        for key in self.keys:
            d[key] = np.where(mask, d[key], self.fill_value)
        return d

if __name__ == "__main__":

    # --- 設定資料夾路徑 ---
    input_folder = "/mnt/ncuma_nas/CENC_CEfixed/test_raw"
    output_folder = os.path.join(input_folder,"./tar_segmented_output")
    os.makedirs(output_folder, exist_ok=True)

    tar_files = glob.glob(os.path.join(input_folder, "*-tar.nii.gz"))
    print(f"找到 {len(tar_files)} 個 -tar.nii.gz 檔案，準備開始處理...\n")

    transform = ApplyCommonMaskd(keys=["src", "tar"], threshold=-1000, fill_value=-1000)

    for tar_path in tar_files:
        base_name = os.path.basename(tar_path).replace("-tar.nii.gz", "")
        src_path = os.path.join(input_folder, f"{base_name}-src.nii.gz")
        save_dir = os.path.join(output_folder, base_name)

        if not os.path.exists(src_path):
            print(f"[{base_name}] 警告：找不到對應的 src 檔案，跳過。")
            continue

        print("-" * 40)
        print(f"正在處理: {base_name}")

        tar_img = nib.load(tar_path)
        tar_img = nib.as_closest_canonical(tar_img)
        src_img = nib.load(src_path)
        src_img = nib.as_closest_canonical(src_img)

        # 檢查維度尺寸是否一致
        if tar_img.shape != src_img.shape:
            print(f"[{base_name}] 錯誤：tar 與 src 尺寸不符 {tar_img.shape} vs {src_img.shape}，跳過！")
            continue

        # 將資料讀取為 Numpy 陣列
        data_dict = {
            "tar": tar_img.get_fdata(),
            "src": src_img.get_fdata()
        }

        # 執行聯合遮罩
        processed_data = transform(data_dict)

        # 【關鍵改變】：在記憶體中建立 NIfTI 物件，不存入硬碟
        in_memory_img = nib.Nifti1Image(
            processed_data["tar"].astype(np.float32),
            tar_img.affine,
            tar_img.header
        )

        try:
            print(f"[{base_name}] 啟動記憶體內 (In-memory) 分割...")

            # 將 in_memory_img 直接傳給 input
            totalsegmentator(
                input=in_memory_img,  # <--- 直接餵入 nibabel 物件
                output=save_dir,  # 結果依然會以檔案形式輸出到這個資料夾
                task="heartchambers_highres",
                fast=False
            )
            print(f"[{base_name}] 分割完成！")

        except Exception as e:
            print(f"[{base_name}] 分割時發生錯誤: {e}")

    print("\n所有影像處理完畢！")