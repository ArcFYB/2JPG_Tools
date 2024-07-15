import os

def delete_block_jpg_files(directory):
    # 列出目录中的所有文件
    files = os.listdir(directory)

    # 遍历文件并删除以"block"开头的JPG文件
    for file in files:
        if file.startswith("block") and file.endswith(".jpg"):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

if __name__ == "__main__":
    directory = '/home/fiko'  # 文件夹路径
    delete_block_jpg_files(directory)
