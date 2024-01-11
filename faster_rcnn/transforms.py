import random
import PIL
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target
class RandomVerticalFlip(object):
    """随机竖直翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)  # 竖直翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target
class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class CutoutTransform(object):
    """随机cutout图像以及调整bboxes"""

    def __init__(self, max_cutout_percentage=0.2):
        self.max_cutout_percentage = max_cutout_percentage

    def __call__(self, image, target):
        # 生成 cutout 的概率
        cutout_prob = random.random()
        if cutout_prob < self.max_cutout_percentage:
            # 计算 cutout 的大小
            cutout_size = int(self.max_cutout_percentage * image.size[0])
            # 随机生成 cutout 区域的位置
            x0 = int(random.uniform(0, image.size[0] - cutout_size))
            y0 = int(random.uniform(0, image.size[1] - cutout_size))
            x1 = x0 + cutout_size
            y1 = y0 + cutout_size
            # 对图像进行 cutout 操作
            image = self.cutout(image, x0, y0, x1, y1)
            # 调整对应的 bbox 坐标信息
            target["boxes"] = self.adjust_bbox(target["boxes"], x0, y0, x1, y1)
        return image, target
    def cutout(self, img, x0, y0, x1, y1):
        # 复制图像
        img = img.copy()
        # 对图像进行 cutout
        PIL.ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], (0, 0, 0))
        return img
    def adjust_bbox(self, bbox, x0, y0, x1, y1):
        # 调整 bbox 的坐标信息
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], x0, x1)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], y0, y1)
        return bbox


class SharpnessTransform(object):
    """调整图像锐度以及保持 bboxes 不变"""

    def __init__(self, max_sharpness_factor=1.9):
        self.max_sharpness_factor = max_sharpness_factor

    def __call__(self, image, target):
        # 生成锐度调整因子
        sharpness_factor = random.uniform(0.1, self.max_sharpness_factor)

        # 对图像进行锐度调整
        image = self.adjust_sharpness(image, sharpness_factor)

        return image, target

    def adjust_sharpness(self, img, sharpness_factor):
        # 复制图像
        img = img.copy()
        # 对图像进行锐度调整
        img = PIL.ImageEnhance.Sharpness(img).enhance(sharpness_factor)
        return img

class CutoutAbsTransform(object):
    """在图像上随机生成矩形区域并填充指定颜色，保持 bboxes 不变"""
    def __init__(self, max_cutout_percentage=0.2, fill_color=(125, 123, 114)):
        self.max_cutout_percentage = max_cutout_percentage
        self.fill_color = fill_color

    def __call__(self, img, target):
        # 生成 Cutout 操作的百分比
        cutout_percentage = random.uniform(0, self.max_cutout_percentage)

        # 对图像进行 Cutout
        img = self.cutout_abs(img, cutout_percentage)

        return img, target

    def cutout_abs(self, img, cutout_percentage):
        # 复制图像
        img = img.copy()

        # 获取图像尺寸
        w, h = img.size

        # 生成随机矩形区域
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)
        x0 = int(max(0, x0 - cutout_percentage * w / 2.))
        y0 = int(max(0, y0 - cutout_percentage * h / 2.))
        x1 = min(w, x0 + cutout_percentage * w)
        y1 = min(h, y0 + cutout_percentage * h)

        xy = (x0, y0, x1, y1)

        # 在图像上画矩形并填充颜色
        PIL.ImageDraw.Draw(img).rectangle(xy, fill=self.fill_color)

        return img

class BrightnessTransform(object):
    """调整图像的亮度，保持 bboxes 不变"""
    def __init__(self, brightness_range=(0.1, 1.9)):
        self.brightness_range = brightness_range

    def __call__(self, img, target):
        # 生成亮度调整的参数
        brightness_factor = random.uniform(*self.brightness_range)

        # 对图像进行亮度调整
        img = self.adjust_brightness(img, brightness_factor)

        return img, target

    def adjust_brightness(self, img, brightness_factor):
        # 调整图像亮度
        enhancer = PIL.ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        return img

class ColorTransform(object):
    """调整图像的颜色，保持 bboxes 不变"""
    def __init__(self, color_range=(0.1, 1.9)):
        self.color_range = color_range

    def __call__(self, img, target):
        # 生成颜色调整的参数
        color_factor = random.uniform(*self.color_range)

        # 对图像进行颜色调整
        img = self.adjust_color(img, color_factor)

        return img, target

    def adjust_color(self, img, color_factor):
        # 调整图像颜色
        enhancer = PIL.ImageEnhance.Color(img)
        img = enhancer.enhance(color_factor)

        return img

import PIL.ImageEnhance

class ContrastTransform(object):
    """调整图像的对比度，保持 bboxes 不变"""
    def __init__(self, contrast_range=(0.1, 1.9)):
        self.contrast_range = contrast_range

    def __call__(self, img, target):
        # 生成对比度调整的参数
        contrast_factor = random.uniform(*self.contrast_range)

        # 对图像进行对比度调整
        img = self.adjust_contrast(img, contrast_factor)

        return img, target

    def adjust_contrast(self, img, contrast_factor):
        # 调整图像对比度
        enhancer = PIL.ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class Posterize2Transform(object):
    """Posterize2 变换，保持 bboxes 不变"""
    def __init__(self, posterize_range=(0, 4)):
        self.posterize_range = posterize_range

    def __call__(self, img, target):
        # 生成 Posterize2 调整的参数
        posterize_value = int(random.uniform(*self.posterize_range))

        # 对图像进行 Posterize2 调整
        img = self.posterize2(img, posterize_value)

        return img, target

    def posterize2(self, img, posterize_value):
        # 调整图像的 Posterize2
        img = PIL.ImageOps.posterize(img, posterize_value)

        return img

class RandomPosterize(object):
    """随机 Posterize 图像以及 bboxes"""
    def __init__(self, min_bits=4, max_bits=8, prob=0.5):
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            bits = random.randint(self.min_bits, self.max_bits)
            image = self.posterize(image, bits)
        return image, target

    @staticmethod
    def posterize(img, bits):
        """
        Posterize the image.

        Args:
            img (PIL.Image): Input image.
            bits (int): Number of bits to keep.

        Returns:
            PIL.Image: Posterized image.
        """
        assert 0 <= bits <= 8
        if bits == 8:
            return img

        img = img.convert("P", palette=Image.ADAPTIVE, colors=2**bits)
        return img

class RandomSolarize(object):
    """随机 Solarize 图像以及 bboxes"""
    def __init__(self, min_threshold=0, max_threshold=256, prob=0.5):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            threshold = random.randint(self.min_threshold, self.max_threshold)
            image = self.solarize(image, threshold)
        return image, target

    @staticmethod
    def solarize(img, threshold):
        """
        Solarize the image.

        Args:
            img (PIL.Image): Input image.
            threshold (int): Threshold value.

        Returns:
            PIL.Image: Solarized image.
        """
        assert 0 <= threshold <= 256
        return PIL.ImageOps.solarize(img, threshold)

class EqualizeWithBBox(object):
    def __call__(self, image, target):
        # 将图像和边界框转换为 PIL.Image 和 torchvision 格式
        pil_image = F.to_pil_image(image)
        bbox = target["boxes"]

        # 应用 Equalize 变换
        equalized_image = self.equalize(pil_image)

        # 将经过均衡化的图像转换回 torch 张量
        equalized_image_tensor = F.to_tensor(equalized_image)

        # 使用均衡化后的图像更新目标
        target["image"] = equalized_image_tensor

        return target

    def equalize(self, img):
        # 应用 Equalize 变换
        equalized_img = PIL.ImageOps.equalize(img)
        return equalized_img

class InvertWithBBox(object):
    def __call__(self, image, target):
        # 将图像和边界框转换为 PIL.Image 和 torchvision 格式
        pil_image = F.to_pil_image(image)
        bbox = target["boxes"]

        # 应用 Invert 变换
        inverted_image = self.invert(pil_image)

        # 将经过反转的图像转换回 torch 张量
        inverted_image_tensor = F.to_tensor(inverted_image)

        # 使用反转后的图像更新目标
        target["image"] = inverted_image_tensor

        return target

    def invert(self, img):
        # 应用 Invert 变换
        inverted_img = PIL.ImageOps.invert(img)
        return inverted_img

class AutoContrastWithBBox(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            # Apply AutoContrast to the image
            image = F.autocontrast(image)

            # Adjust bounding box coordinates accordingly
            if 'boxes' in target:
                width, height = image.size
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]

        return image, target

class RotateWithBBox(object):
    def __init__(self, degrees=(-30, 30), prob=1.0):
        self.degrees = degrees
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            # Randomly select rotation degree within the specified range
            angle = np.random.uniform(self.degrees[0], self.degrees[1])

            # Apply rotation to the image
            image = F.rotate(image, angle)

            # Adjust bounding box coordinates accordingly
            if 'boxes' in target:
                # Convert angle to radians
                radians = np.radians(angle)
                # Get image center coordinates
                center_x, center_y = image.width / 2, image.height / 2
                # Get rotation matrix
                rotation_matrix = torch.tensor([
                    [np.cos(radians), -np.sin(radians)],
                    [np.sin(radians), np.cos(radians)]
                ])
                # Get bounding box center coordinates
                bbox_centers = (target['boxes'][:, :2] + target['boxes'][:, 2:]) / 2
                # Translate bounding box centers to be relative to the image center
                bbox_centers -= torch.tensor([[center_x, center_y]])
                # Rotate bounding box centers
                rotated_bbox_centers = bbox_centers @ rotation_matrix.t()
                # Translate bounding box centers back to the original coordinates
                rotated_bbox_centers += torch.tensor([[center_x, center_y]])
                # Get rotated bounding boxes
                rotated_boxes = torch.cat([
                    rotated_bbox_centers - target['boxes'][:, 2:] / 2,
                    rotated_bbox_centers + target['boxes'][:, 2:] / 2
                ], dim=1)
                # Clip bounding boxes to be within image boundaries
                rotated_boxes[:, 0::2] = rotated_boxes[:, 0::2].clamp(0, image.width)
                rotated_boxes[:, 1::2] = rotated_boxes[:, 1::2].clamp(0, image.height)
                target['boxes'] = rotated_boxes

        return image, target


class TranslateYAbs(object):
    """随机竖直平移图像以及bboxes"""

    def __init__(self, max_translate_y=10, prob=0.5):
        self.max_translate_y = max_translate_y
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的竖直平移
            translate_y = random.uniform(0, self.max_translate_y)
            image = self.translate_y_transform(image, translate_y)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            height = image.size[1]
            bbox[:, [1, 3]] += int(translate_y)
            bbox[:, [1, 3]] = torch.clamp(bbox[:, [1, 3]], 0, height)  # 保证坐标在图像内
            target["boxes"] = bbox

        return image, target

    def translate_y_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateXAbs(object):
    """随机水平平移图像以及bboxes"""

    def __init__(self, max_translate_x=10, prob=0.5):
        self.max_translate_x = max_translate_x
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的水平平移
            translate_x = random.uniform(0, self.max_translate_x)
            image = self.translate_x_transform(image, translate_x)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            width = image.size[0]
            bbox[:, [0, 2]] += int(translate_x)
            bbox[:, [0, 2]] = torch.clamp(bbox[:, [0, 2]], 0, width)  # 保证坐标在图像内
            target["boxes"] = bbox

        return image, target

    def translate_x_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY(object):
    """随机竖直平移图像以及bboxes"""

    def __init__(self, max_translate_y=0.45, prob=0.5):
        self.max_translate_y = max_translate_y
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的竖直平移
            translate_y = random.uniform(-self.max_translate_y, self.max_translate_y)
            image = self.translate_y_transform(image, translate_y)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            height = image.size[1]
            bbox[:, [1, 3]] += int(translate_y)
            bbox[:, [1, 3]] = torch.clamp(bbox[:, [1, 3]], 0, height)  # 保证坐标在图像内
            target["boxes"] = bbox

        return image, target

    def translate_y_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateX(object):
    """随机水平平移图像以及bboxes"""

    def __init__(self, max_translate_x=0.45, prob=0.5):
        self.max_translate_x = max_translate_x
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的水平平移
            translate_x = random.uniform(-self.max_translate_x, self.max_translate_x)
            image = self.translate_x_transform(image, translate_x)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            width = image.size[0]
            bbox[:, [0, 2]] += int(translate_x)
            bbox[:, [0, 2]] = torch.clamp(bbox[:, [0, 2]], 0, width)  # 保证坐标在图像内
            target["boxes"] = bbox

        return image, target

    def translate_x_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


class ShearX(object):
    """随机横向错切图像以及bboxes"""

    def __init__(self, max_shear_x=0.3, prob=0.5):
        self.max_shear_x = max_shear_x
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的横向错切
            shear_x = random.uniform(-self.max_shear_x, self.max_shear_x)
            image = self.shear_x_transform(image, shear_x)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            width = image.size[0]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

        return image, target

    def shear_x_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


class ShearY(object):
    """随机纵向错切图像以及bboxes"""

    def __init__(self, max_shear_y=0.3, prob=0.5):
        self.max_shear_y = max_shear_y
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # 图像的纵向错切
            shear_y = random.uniform(-self.max_shear_y, self.max_shear_y)
            image = self.shear_y_transform(image, shear_y)

            # 调整目标的坐标信息
            bbox = target["boxes"]
            height = image.size[1]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox

        return image, target

    def shear_y_transform(self, img, v):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))