import os
import numpy as np
import nibabel as nib

# 1. 設定器官對應的數字 ID (Label Value)
organ_labels = {
    "heart_myocardium": 1,  # 心肌
    "heart_ventricle_left": 2,  # 左心室
    "heart_ventricle_right": 3,  # 右心室
    "heart_atrium_left": 4,  # 左心房
    "heart_atrium_right": 5,  # 右心房
    "aorta": 6,  # 主動脈
    "pulmonary_artery": 7  # 肺動脈
}
if __name__ == "__main__":
    # --- 請修改以下路徑 ---
    input_folder = "/mnt/ncuma_nas/CENC_CEfixed/test_raw/tar_segmented_output/CVAI-0560"  # 放 7 個 nii.gz 的資料夾
    output_file = os.path.join(input_folder, "heart_seg_all.nii.gz") # 合併後的輸出檔案名稱

    combined_array = None
    reference_affine = None
    reference_header = None

    print("開始讀取並合併器官 Mask...")

    for organ_name, label_id in organ_labels.items():
        file_path = os.path.join(input_folder, f"{organ_name}.nii.gz")

        if os.path.exists(file_path):
            nii = nib.load(file_path)
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata() > 0  # 轉為 True/False 布林值

            # 第一次讀取時初始化畫布與座標資訊
            if combined_array is None:
                combined_array = np.zeros(data.shape, dtype=np.uint8)
                reference_affine = nii.affine
                reference_header = nii.header

            # 將對應的器官寫入指定數值 ID
            combined_array[data] = label_id
            print(f"已合併: {organ_name} -> Label ID: {label_id}")
        else:
            print(f"⚠️ 找不到檔案: {organ_name}.nii.gz，已跳過。")

    # 2. 封裝並轉為 RAS+ 系統儲存
    if combined_array is not None:
        # 建立初步的 NIfTI 物件
        raw_nii = nib.Nifti1Image(combined_array, reference_affine, header=reference_header)

        # 【關鍵關鍵】：將影像強轉為 RAS+ 方向 (Right-Anterior-Superior)
        ras_nii = nib.as_closest_canonical(raw_nii)

        # 驗證轉換後的座標軸方向
        orient_codes = nib.aff2axcodes(ras_nii.affine)
        print(f"\n轉換後的影像座標軸方向: {orient_codes}")  # 預期顯示 ('R', 'A', 'S')

        # 寫入磁碟
        nib.save(ras_nii, output_file)
        print("\n" + "=" * 40)
        print(f"🎉 成功！已按照 RAS+ 座標系統儲存至:\n{os.path.abspath(output_file)}")
        print("=" * 40)
    else:
        print("❌ 失敗：資料夾中未找到任何符合的器官 NIfTI 檔案。")