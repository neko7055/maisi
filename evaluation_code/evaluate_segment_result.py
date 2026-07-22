import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm


# 新增 pred_suffix 參數，預設為 "-tar_pred"
def evaluate_segmentations(gt_base_dir, pred_base_dir, output_csv="evaluation_results.csv", pred_suffix="-tar_pred"):
    results = []

    # 取得 GT 資料夾下的所有病患 ID (例如 XXX)
    gt_ids = [d for d in os.listdir(gt_base_dir) if os.path.isdir(os.path.join(gt_base_dir, d))]

    for gt_id in tqdm(gt_ids, desc="Processing Patients"):
        # 推導對應的 Pred 資料夾名稱 (XXX-tar_pred)
        pred_id = f"{gt_id}{pred_suffix}"
        gt_patient_dir = os.path.join(gt_base_dir, gt_id)
        pred_patient_dir = os.path.join(pred_base_dir, pred_id)

        if not os.path.exists(pred_patient_dir):
            print(f"Warning: 找不到對應的預測資料夾 {pred_patient_dir}，已略過。")
            continue

        # 找出該病患 GT 資料夾下所有的 NIfTI 分割檔 (TotalSegmentator 預設產出)
        gt_files = glob.glob(os.path.join(gt_patient_dir, "*.nii.gz"))

        for gt_file_path in gt_files:
            organ_filename = os.path.basename(gt_file_path)
            pred_file_path = os.path.join(pred_patient_dir, organ_filename)

            if not os.path.exists(pred_file_path):
                # 若預測資料夾中缺少此器官，視為預測全空
                continue

            # 1. 讀取影像 (包含 metadata 如 spacing, origin, direction)
            gt_img = sitk.ReadImage(gt_file_path)
            gt_img = sitk.DICOMOrient(gt_img, 'RAS')
            pred_img = sitk.ReadImage(pred_file_path)
            pred_img = sitk.DICOMOrient(pred_img, 'RAS')

            # 2. 將 Predict 影像對齊 (Resample) 到 GT 影像的物理空間與維度
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(gt_img)  # 設定 GT 為參考基準
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 分割標籤必須用最近鄰插值
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform())  # Identity Transform

            pred_aligned = resampler.Execute(pred_img)

            # 3. 轉換為 Numpy 陣列進行矩陣運算 (Z, Y, X 順序)
            gt_arr = sitk.GetArrayFromImage(gt_img) > 0
            pred_arr = sitk.GetArrayFromImage(pred_aligned) > 0

            # 取得 Voxel Spacing (注意：SimpleITK 是 X,Y,Z，Numpy 是 Z,Y,X，需反轉)
            spacing = gt_img.GetSpacing()[::-1]

            # 4. 計算指標 (記得接收的值變成 hd95)
            dice, iou, hd95, asd = calculate_metrics_for_arrays(gt_arr, pred_arr, spacing)

            results.append({
                "Patient_ID": gt_id,
                "Organ": organ_filename.replace(".nii.gz", ""),
                "Dice": dice,
                "IoU": iou,
                "HD95_mm": hd95,  # 欄位名稱更新
                "Average_Surface_Distance_mm": asd
            })

    # 輸出為 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n評估完成！結果已儲存至 {output_csv}")
    return df


def calculate_metrics_for_arrays(gt_arr, pred_arr, spacing):
    """計算 Dice, IoU, HD95, ASD (含邊界防護)"""
    gt_sum = gt_arr.sum()
    pred_sum = pred_arr.sum()

    # 邊界防護：如果兩者皆為空
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 0.0, 0.0

    # 邊界防護：如果其中一方為空 (無法計算距離)
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, np.nan, np.nan

    # 交集與聯集
    intersection = (gt_arr & pred_arr).sum()
    union = (gt_arr | pred_arr).sum()

    # Dice Coefficient & IoU
    dice = 2.0 * intersection / (gt_sum + pred_sum)
    iou = intersection / union if union > 0 else 0.0

    # 計算 HD95 與 ASD
    try:
        # 改用 hd95 函數
        hd95 = metric.binary.hd95(pred_arr, gt_arr, voxelspacing=spacing)
        asd = metric.binary.assd(pred_arr, gt_arr, voxelspacing=spacing)
    except Exception as e:
        hd95 = np.nan
        asd = np.nan

    return dice, iou, hd95, asd


def print_summary_statistics(df):
    print("\n" + "=" * 40)
    print("           整體評估統計結果摘要")
    print("=" * 40)

    # 更新這裡的欄位名稱
    metrics = ["Dice", "IoU", "HD95_mm", "Average_Surface_Distance_mm"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        valid_data = df[metric].dropna()

        if len(valid_data) == 0:
            print(f"[{metric}] 無有效數據可供統計\n" + "-" * 40)
            continue

        mean_val = valid_data.mean()
        median_val = valid_data.median()

        # 由於欄位名稱從 Hausdorff_Distance 變成 HD95，這裡判斷式需包含 "HD95" 或 "Distance"
        if "Distance" in metric or "HD95" in metric:
            worst_val = valid_data.max()
            worst_desc = "最大值"
        else:
            worst_val = valid_data.min()
            worst_desc = "最小值"

        print(f"指標: {metric}")
        print(f"  ▶ 平均值 (Mean)   : {mean_val:.4f}")
        print(f"  ▶ 中位數 (Median) : {median_val:.4f}")
        print(f"  ▶ 最差值 (Worst)  : {worst_val:.4f} ({worst_desc})")
        print("-" * 40)


def print_organ_statistics(df):
    """印出每個器官所有指標的獨立統計結果 (Mean, Median, Worst)"""
    print("\n" + "=" * 90)
    print("                                各器官獨立評估統計")
    print("=" * 90)

    # 定義所有要統計的指標 (需對應 DataFrame 中的欄位名稱)
    metrics = ["Dice", "IoU", "HD95_mm", "Average_Surface_Distance_mm"]
    summary_list = []

    for organ, group in df.groupby("Organ"):
        organ_stats = {"Organ": organ}

        for metric in metrics:
            if metric not in df.columns:
                continue

            valid_data = group[metric].dropna()

            # 簡化欄位名稱，避免終端機印出時表格過寬折行
            short_metric = metric.replace("_mm", "").replace("Average_Surface_Distance", "ASD")

            if len(valid_data) > 0:
                organ_stats[f"{short_metric}_Mean"] = valid_data.mean()
                organ_stats[f"{short_metric}_Med"] = valid_data.median()

                # 判斷最差值：HD95與ASD(距離)找最大值，Dice與IoU(重疊率)找最小值
                if "Distance" in metric or "HD95" in metric:
                    organ_stats[f"{short_metric}_Worst"] = valid_data.max()
                else:
                    organ_stats[f"{short_metric}_Worst"] = valid_data.min()
            else:
                organ_stats[f"{short_metric}_Mean"] = np.nan
                organ_stats[f"{short_metric}_Med"] = np.nan
                organ_stats[f"{short_metric}_Worst"] = np.nan

        summary_list.append(organ_stats)

    # 轉換成 DataFrame 並將浮點數四捨五入到小數點後 4 位
    summary_df = pd.DataFrame(summary_list).round(4)

    # 使用 to_string(index=False) 避免印出最左側的 DataFrame index
    print(summary_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    GT_DIRECTORY = "/mnt/ncuma_nas/CENC_CEfixed/test_raw/tar_segmented_output"
    PRED_DIRECTORY = "/mnt/hdd_nfs/temp_work_dir/predictions/SRC_GT/test/segmented_output"

    # 這裡可以自訂預測資料夾的尾綴 (例如傳入空字串 "" 如果完全同名)
    CUSTOM_SUFFIX = "-tar_pred"

    results_df = evaluate_segmentations(GT_DIRECTORY, PRED_DIRECTORY, pred_suffix=CUSTOM_SUFFIX)

    print_summary_statistics(results_df)
    print_organ_statistics(results_df)