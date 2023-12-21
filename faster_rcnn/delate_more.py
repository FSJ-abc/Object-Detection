import os

def process_folders(folder_a, folder_b):
    # 获取a文件夹中的所有jpg文件
    a_files = [f for f in os.listdir(folder_a) if f.endswith(".jpg")]

    # 获取b文件夹中的所有xml文件
    b_files = [f for f in os.listdir(folder_b) if f.endswith(".xml")]

    # 检查b文件夹中多余的xml文件并删除
    for xml_file in b_files:
        jpg_file = xml_file.replace(".xml", ".jpg")
        if jpg_file not in a_files:
            file_to_remove = os.path.join(folder_b, xml_file)
            os.remove(file_to_remove)
            print(f"Removed {file_to_remove}")

if __name__ == "__main__":
    folder_a = r"E:\transfer learning\myself\faster_rcnn\VOCdevkit\VOC2012\JPEGImages"  # 替换为实际的a文件夹路径
    folder_b = r"E:\transfer learning\myself\faster_rcnn\VOCdevkit\VOC2012\Annotations"  # 替换为实际的b文件夹路径

    process_folders(folder_a, folder_b)
