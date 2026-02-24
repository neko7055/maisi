import csv
import sys
from collections import defaultdict


def main(file_path):
    max_memory = defaultdict(int)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳過標題行

            for row in reader:
                if len(row) < 4:
                    continue

                # 取得 GPU index 並去除空白
                gpu_index = row[1].strip()

                # 取得記憶體數值，去除 'MiB' 與空白後轉為整數
                mem_str = row[3].strip().replace('MiB', '').strip()
                try:
                    mem_val = int(mem_str)
                    if mem_val > max_memory[gpu_index]:
                        max_memory[gpu_index] = mem_val
                except ValueError:
                    continue

        print("=== GPU 最大記憶體使用量統計 ===")
        # 依照 GPU index 排序輸出
        for gpu_index in sorted(max_memory.keys(), key=lambda x: int(x)):
            print(f"GPU {gpu_index}: {max_memory[gpu_index]} MiB")

    except FileNotFoundError:
        print(f"找不到檔案: {file_path}")


if __name__ == "__main__":
    # 允許從命令列傳入檔案路徑，預設為 gpu_log.csv
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'gpu_log.csv'
    main(file_path)
