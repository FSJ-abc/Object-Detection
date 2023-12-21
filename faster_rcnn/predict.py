import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs
from torchvision.models.detection import FasterRCNN

def count_files_in_folder(folder_path):
    try:
        # 获取指定文件夹下的所有文件
        files = os.listdir(folder_path)

        # 使用 len() 函数获取文件数量
        num_files = len(files)

        print(f"文件夹 {folder_path} 中的文件数量为: {num_files}")
        return num_files

    except Exception as e:
        print(f"发生错误: {e}")
        return 0



def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    # 设置 RPN 阶段的分数阈值
    model.rpn.score_thresh = 0.5  # 替换为你想要的分数阈值
    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=2)

    # load trained weights
    weights_path = "./save_weights/resNetFpn-model-14.pth"
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict, False)
    model.to(device)

    # read class indices
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # specify folder path to count files
    folder_path = r"C:\Users\fsj\Desktop\new_img\No.4\lineimg\cropped_images\result"
    output_file = "/path"
    # call function to count files and get the count
    file_count = count_files_in_folder(folder_path)

    # check if file_count is None or 0
    if file_count is None or file_count == 0:
        print("No files found in the folder.")
        return

    # loop through each file
    for num in range(1, file_count + 1):
        # load image using os.path.join
        img_path = os.path.join(folder_path, f"img{str(num).zfill(3)}.jpg")

        # check if the file exists
        if not os.path.exists(img_path):
            print(f"Image file {img_path} not found. Skipping...")
            continue

        original_img = Image.open(img_path)

        # convert PIL image to tensor, without normalizing image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # set to evaluation mode
        with torch.no_grad():
            # initialize
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("Inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            #手动添加非极大值抑制
            keep = torchvision.ops.nms(torch.tensor(predict_boxes), torch.tensor(predict_scores), iou_threshold=0.3)

            predict_boxes = predict_boxes[keep]
            predict_classes = predict_classes[keep]
            predict_scores = predict_scores[keep]

            plot_img = draw_objs(original_img,
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)
            # plt.imshow(plot_img)
            # plt.show()
            # 保存预测的图片结果
            output_file = r"C:\Users\fsj\Desktop\new_img\No.4\lineimg\cropped_images/result/result"
            output_file_path = os.path.join(output_file, f"img_result_{num}.jpg")
            plot_img.save(output_file_path)
            print(f"Result saved: {output_file_path}")

if __name__ == '__main__':
    main()
