import os
import random

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    train_txt_path = "./VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    val_txt_path = "./VOCdevkit/VOC2012/ImageSets/Main/val.txt"

    # 尝试删除现有的文件
    try:
        os.remove(train_txt_path)
        os.remove(val_txt_path)
        print(f"Existing files '{train_txt_path}' and '{val_txt_path}' have been deleted.")
    except FileNotFoundError:
        pass  # 如果文件不存在，继续执行

    # 创建新的文件
    with open(train_txt_path, "w") as train_f, open(val_txt_path, "w") as eval_f:
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))

if __name__ == '__main__':
    main()
