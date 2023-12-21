import os
import sys
import json
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from torchvision.models import mobilenet_v2
from train_utils import train_eval_utils as utils
from backbone import resnet50_fpn_backbone, MobileNetV2

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # # 注意，这里的norm_layer要和训练脚本中保持一致
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def transfer_learning_mobilenetv2(net, train_loader, validate_loader, device, num_classes=2, epochs=3, my_lr=0.0001):
    # # Load pretrain weights
    # model_weight_path = r".\save_weights\pretrain.pth"
    # assert os.path.exists(model_weight_path), "File {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # load train weights
    weights_path = "./save_weights/mobile-model-24.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    net.load_state_dict(weights_dict,False)

    # # Change classifier layer structure for fine-tuning
    # in_channel = net.classifier[1].in_features
    # net.classifier[1] = nn.Linear(in_channel, num_classes)
    net.to(device)
# 设置保存结果的文件名
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S_tr"))
#设置保存权重的文件夹，如果不存在就创建一个
    if not os.path.exists("save_weights_tr"):
        os.makedirs("save_weights_tr")


    # # Define loss function
    # loss_function = nn.CrossEntropyLoss()

    train_loss = []
    learning_rate = []
    val_map = []

    amp = False  # 是否使用混合精度训练，需要GPU支持

    scaler = torch.cuda.amp.GradScaler() if amp else None
    save_path = r".\save_weights\mobile-model-24.pth"
    net.train()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in net.backbone.parameters():
        param.requires_grad = False


    # define optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, my_lr,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    for epoch in range(epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(net, optimizer, train_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(int, validate_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        # 仅保存最后5个epoch的权重
        if epoch in range(epochs)[-5:]:
            save_files = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

    #
    # for epoch in range(epochs):
    #     # Train
    #     net.train()
    #     running_loss = 0.0
    #     train_bar = tqdm(train_loader, file=sys.stdout)
    #     for step, data in enumerate(train_bar):
    #         images, labels = data
    #         optimizer.zero_grad()
    #         logits = net(images.to(device))
    #         loss = loss_function(logits, labels.to(device))
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print statistics
    #         running_loss += loss.item()
    #         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
    #                                                                  epochs,
    #                                                                  loss)
    #
    #     # Validate
    #     net.eval()
    #     acc = 0.0  # Accumulate accurate number / epoch
    #     with torch.no_grad():
    #         val_bar = tqdm(validate_loader, file=sys.stdout)
    #         for val_data in val_bar:
    #             val_images, val_labels = val_data
    #             outputs = net(val_images.to(device))
    #             predict_y = torch.max(outputs, dim=1)[1]
    #             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    #
    #             val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
    #                                                        epochs)
    #
    #     val_accurate = acc / len(validate_loader.dataset)
    #     print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #           (epoch + 1, running_loss / train_steps, val_accurate))
    #
    #     if val_accurate > best_acc:
    #         best_acc = val_accurate
    #         torch.save(net.state_dict(), save_path)
    #
    # print('Finished Transfer Learning')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #定义数据转换
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    #获取数据集
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # 获取数据根路径
    image_path = os.path.join(data_root, "my_data")  # 设置图像数据集路径
    assert os.path.exists(image_path), "{} 路径不存在.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    my_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in my_list.items())
    # 将字典写入 JSON 文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 设置数据加载器的工作进程数量
    print('Using {} dataloader workers every process'.format(nw))

    # 创建并加载数据集和数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last=True)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # net = mobilenet_v2(pretrained=True)
    net = create_model(num_classes=2)
    transfer_learning_mobilenetv2(net, train_loader, validate_loader, device)

if __name__ == '__main__':
    main()