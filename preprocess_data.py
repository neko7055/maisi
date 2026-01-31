import json
import os
import SimpleITK as sitk
from tqdm import tqdm
import multiprocessing

def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                filename = file.replace(".nii.gz", "").replace(".nii", "")
                file_type = filename.split("-")[-1]
                fileID = "-".join(filename.split("-")[:-1])
                gz_files.append({"filepath": os.path.join(root, file), "fileID": fileID, "type": file_type})
    return gz_files

def resize_image_and_store(src_path, dest_path, new_z = 512):
    img = sitk.ReadImage(src_path, sitk.sitkFloat32)

    orig_size = img.GetSize()  # (x, y, z)
    orig_spacing = img.GetSpacing()  # (sx, sy, sz)
    new_size = (
        orig_size[0],  # X 不變
        orig_size[1],  # Y 不變
        new_z
    )
    new_spacing = (
        orig_spacing[0],  # sx 不變
        orig_spacing[1],  # sy 不變
        orig_spacing[2] * (orig_size[2] - 1) / (new_z - 1)
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)  # CT
    resampler.SetDefaultPixelValue(-1000)

    resampled_img = resampler.Execute(img)
    sitk.WriteImage(resampled_img, dest_path)

# Worker initializer: run once per child process
def _worker_init():
    # Limit SimpleITK threads per worker to avoid oversubscription
    try:
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        # If SimpleITK changed API, ignore silently
        pass

# Worker wrapper: receives tuple args to be picklable
def _worker_resize(args):
    src_path, dest_path, new_z = args
    try:
        # Ensure SimpleITK imported in child (module import at top already imported, but safe)
        # Do resize
        resize_image_and_store(src_path, dest_path, new_z=new_z)
        return (src_path, dest_path, True, "")
    except Exception as e:
        return (src_path, dest_path, False, str(e))

if __name__ == "__main__":
    raw_data_dir = "/work/r12946008/CENC_CEfixed"
    data_split_json = os.path.join(raw_data_dir, "dataset_split.json")
    z=512
    data_dir = "./CVAI_data" + f"_z{z}"
    os.makedirs(data_dir, exist_ok=True)
    dataroot_dir = os.path.join(data_dir, "data")
    os.makedirs(dataroot_dir, exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(dataroot_dir, "test"), exist_ok=True)

    gz_files = list_gz_files(raw_data_dir)

    with open(data_split_json, "r") as file:
        json_data = json.load(file)

    # Build job list: (src_path, dest_path, new_z)
    jobs = []
    new_z = z
    for mode in ["training", "validation", "test"]:
        out_dir = os.path.join(dataroot_dir, mode)
        # ensure existence
        os.makedirs(out_dir, exist_ok=True)
        for fileID in json_data[mode]:
            # find every file that matches this fileID
            matched_files = [f for f in gz_files if f["fileID"] == fileID]
            for f in matched_files:
                src_path = f["filepath"]
                filename = os.path.basename(src_path)
                dest_path = os.path.join(out_dir, filename)
                jobs.append((src_path, dest_path, new_z))
    # Run pool: choose processes conservatively
    cpu_count = max(1, min(multiprocessing.cpu_count(), len(jobs)))
    processes = cpu_count if cpu_count > 0 else 1

    # Use Pool with initializer to set per-process SITK threads
    with multiprocessing.Pool(processes=processes, initializer=_worker_init) as pool:
        # tqdm over imap_unordered for progress
        results_iter = pool.imap_unordered(_worker_resize, jobs, chunksize=1)
        errors = []
        for res in tqdm(results_iter, total=len(jobs), desc="Resampling"):
            src_path, dest_path, success, msg = res
            if not success:
                errors.append((src_path, msg))

    if errors:
        print(f"Encountered {len(errors)} errors during resizing. Example: {errors[:5]}")

    # Reconstruct data_lists (same format as original)
    data_lists = {"training": [], "validation": [], "test": []}
    for mode in ["training", "validation", "test"]:
        datasave_dir = os.path.join(dataroot_dir, mode)
        for fileID in json_data[mode]:
            matched_files = filter(lambda x: x["fileID"] == fileID, gz_files)
            data_list = {"src_image": "", "tar_image": "", "modality": "ct"}
            for f in matched_files:
                filename = os.path.basename(f["filepath"])
                # the resized file now lives in datasave_dir/filename
                data_list[f["type"] + "_image"] = filename
            data_lists[mode].append(data_list)

    datalist_file = os.path.join(data_dir, "datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(data_lists, f, indent=2)

    print("Done.")