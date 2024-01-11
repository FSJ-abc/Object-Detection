import math
import random

import transforms

import PIL
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
# random_mirror = True
#
#
# def shear_x_transform(img, v):
#     assert -0.3 <= v <= 0.3
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     # 对图像进行 ShearX 变换
#     img = img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
#     # 调整目标的坐标信息
#     bbox = target["boxes"]
#     width = img.size[0]
#     # 调整 x 轴坐标
#     bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
#     target["boxes"] = bbox
#     return img, target
#
# def shear_y_transform(img, v):
#     assert -0.3 <= v <= 0.3
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     # 对图像进行 ShearY 变换
#     img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
#     # 调整目标的坐标信息
#     bbox = target["boxes"]
#     height = img.size[1]
#     # 调整 y 轴坐标
#     bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
#     target["boxes"] = bbox
#     return img, target
#
# def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert -0.45 <= v <= 0.45
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     v = v * img.size[0]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
#
#
# def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert -0.45 <= v <= 0.45
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     v = v * img.size[1]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
#
#
# def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert 0 <= v <= 10
#     if random.random() > 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
#
#
# def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert 0 <= v <= 10
#     if random.random() > 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
#
#
# def Rotate(img, v):  # [-30, 30]
#     assert -30 <= v <= 30
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     return img.rotate(v)
#
#
# def AutoContrast(img, _):
#     return PIL.ImageOps.autocontrast(img)
#
#
# def Invert(img, _):
#     return PIL.ImageOps.invert(img)
#
#
# def Equalize(img, _):
#     return PIL.ImageOps.equalize(img)
#
#
# def Flip(img, _):  # not from the paper
#     return PIL.ImageOps.mirror(img)
#
#
# def Solarize(img, v):  # [0, 256]
#     assert 0 <= v <= 256
#     return PIL.ImageOps.solarize(img, v)
#
#
# def Posterize(img, v):  # [4, 8]
#     assert 4 <= v <= 8
#     v = int(v)
#     return PIL.ImageOps.posterize(img, v)
#
#
# def Posterize2(img, v):  # [0, 4]
#     assert 0 <= v <= 4
#     v = int(v)
#     return PIL.ImageOps.posterize(img, v)
#
#
# def Contrast(img, v):  # [0.1,1.9]
#     assert 0.1 <= v <= 1.9
#     return PIL.ImageEnhance.Contrast(img).enhance(v)
#
#
# def Color(img, v):  # [0.1,1.9]
#     assert 0.1 <= v <= 1.9
#     return PIL.ImageEnhance.Color(img).enhance(v)
#
#
# def Brightness(img, v):  # [0.1,1.9]
#     assert 0.1 <= v <= 1.9
#     return PIL.ImageEnhance.Brightness(img).enhance(v)
#
#
# def Sharpness(img, v):  # [0.1,1.9]
#     assert 0.1 <= v <= 1.9
#     return PIL.ImageEnhance.Sharpness(img).enhance(v)
#
#
# def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
#     assert 0.0 <= v <= 0.2
#     if v <= 0.:
#         return img
#
#     v = v * img.size[0]
#     return CutoutAbs(img, v)
#
#
# def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
#     # assert 0 <= v <= 20
#     if v < 0:
#         return img
#     w, h = img.size
#     x0 = np.random.uniform(w)
#     y0 = np.random.uniform(h)
#
#     x0 = int(max(0, x0 - v / 2.))
#     y0 = int(max(0, y0 - v / 2.))
#     x1 = min(w, x0 + v)
#     y1 = min(h, y0 + v)
#
#     xy = (x0, y0, x1, y1)
#     color = (125, 123, 114)
#     # color = (0, 0, 0)
#     img = img.copy()
#     PIL.ImageDraw.Draw(img).rectangle(xy, color)
#     return img
def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (transforms.ShearX, -0.3, 0.3),  # 0
        (transforms.ShearY, -0.3, 0.3),  # 1
        (transforms.TranslateX, -0.45, 0.45),  # 2
        (transforms.TranslateY, -0.45, 0.45),  # 3
        (transforms.RotateWithBBox, -30, 30),  # 4
        (transforms.AutoContrastWithBBox, 0, 1),  # 5
        (transforms.InvertWithBBox, 0, 1),  # 6
        (transforms.EqualizeWithBBox, 0, 1),  # 7
        (transforms.RandomSolarize, 0, 256),  # 8
        (transforms.RandomPosterize, 4, 8),  # 9
        (transforms.ContrastTransform, 0.1, 1.9),  # 10
        (transforms.ColorTransform, 0.1, 1.9),  # 11
        (transforms.BrightnessTransform, 0.1, 1.9),  # 12
        (transforms.SharpnessTransform, 0.1, 1.9),  # 13
        (transforms.CutoutTransform, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (transforms.CutoutAbsTransform, 0, 20),  # compatible with auto-augment
            (transforms.Posterize2Transform, 0, 4),  # 9
            (transforms.TranslateXAbs, 0, 10),  # 9
            (transforms.TranslateYAbs, 0, 10),  # 9
        ]
    return l
# 将数据增强函数的名称映射到对应的元组，该元组包含了增强函数本身和其强度范围的下限（v1）和上限（v2）
augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

# 通过给定的增强函数名称 name，从预先构建好的字典 augment_dict 中获取相应的元组
def get_augment(name):
    return augment_dict[name]

# 应用指定名称的数据增强函数到给定的图像上。
def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)
# 这段代码定义了一个数据增强类，通过随机选择一种增强策略，并按照该策略的具体操作对输入图像进行增强。
class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img
# 随机区域裁剪
class EfficientNetRandomCropWithBoxes:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        """
        初始化随机裁剪的参数和备用的中心裁剪。

        Args:
            imgsize (int): 裁剪后的图像尺寸。
            min_covered (float): 最小覆盖比例，用于控制裁剪区域与原图的相似性。
            aspect_ratio_range (tuple): 高宽比例范围。
            area_range (tuple): 裁剪区域面积占原图的比例范围。
            max_attempts (int): 最大尝试次数，以获取有效的裁剪。

        """
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.imgsize = imgsize
        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCropWithBoxes(imgsize)

    def __call__(self, img, boxes):
        """
        对图像和标记框进行随机裁剪。

        Args:
            img (PIL Image): 待裁剪的图像。
            boxes (list): 包含标记框左上角和右下角坐标的列表。

        Returns:
            PIL Image: 裁剪后的图像。
            list: 调整后的标记框。

        """
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue

            if width > original_width or height > original_height:
                continue

            if area < self.min_covered * (original_width * original_height):
                continue

            if width == original_width and height == original_height:
                # 如果裁剪区域与原图相同，使用备用的中心裁剪
                return self._fallback(img), boxes

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)

            # 调整标记框的坐标，使其适应裁剪后的图像
            adjusted_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                xmin = max(0, xmin - x)
                ymin = max(0, ymin - y)
                xmax = min(width, xmax - x)
                ymax = min(height, ymax - y)
                if xmin < width and ymin < height and xmax > 0 and ymax > 0:
                    adjusted_boxes.append([xmin, ymin, xmax, ymax])

            if adjusted_boxes:
                return img.crop((x, y, x + width, y + height)), adjusted_boxes

        # 如果所有尝试都失败，返回备用的中心裁剪
        return self._fallback(img), boxes

# 随机从中心裁剪
class EfficientNetCenterCropWithBoxes:
    def __init__(self, imgsize):
        """
        初始化中心裁剪的参数。

        Args:
            imgsize (int): 裁剪后的图像尺寸。

        """
        self.imgsize = imgsize

    def __call__(self, img, boxes):
        """
        对图像和标记框进行中心裁剪。

        Args:
            img (PIL Image): 待裁剪的图像。
            boxes (list): 包含标记框左上角和右下角坐标的列表。

        Returns:
            PIL Image: 裁剪后的图像。
            list: 调整后的标记框。

        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))

        # 调整标记框的坐标，使其适应裁剪后的图像
        adjusted_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin = max(0, xmin - crop_left)
            ymin = max(0, ymin - crop_top)
            xmax = min(crop_width, xmax - crop_left)
            ymax = min(crop_height, ymax - crop_top)
            if xmin < crop_width and ymin < crop_height and xmax > 0 and ymax > 0:
                adjusted_boxes.append([xmin, ymin, xmax, ymax])

        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)), adjusted_boxes

def policy_decoder(augment, num_policy, num_op):
    op_list = augment_list(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies