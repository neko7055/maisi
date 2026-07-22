import os
import glob
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
if __name__ == '__main__':
    # 設定輸入與輸出資料夾路徑
    input_folder = "/mnt/hdd_nfs/temp_work_dir/predictions/20260723003619/test"
    output_folder = os.path.join(input_folder,"./segmented_output")

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 找出所有 .nii.gz 檔案
    ct_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))
    print(f"找到 {len(ct_files)} 個 CT 檔案，開始批次分割...")

    for file_path in ct_files:
        # 取得原始檔名 (不含附檔名) 以建立專屬輸出資料夾
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        save_dir = os.path.join(output_folder, base_name)

        img = nib.load(file_path)
        img = nib.as_closest_canonical(img)

        print(f"正在處理: {base_name} ...")

        # 執行分割，指定任務為 heartchambers_highres
        totalsegmentator(
            input=img,  # 將 input_path 改為 input
            output=save_dir,  # 將 output_path 改為 output
            task="heartchambers_highres",
            fast=False
        )

    print("全部分割完成！")