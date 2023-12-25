# import random
# import PIL
# from PIL import Image
# import numpy as np
# import torch
# import torchvision
# from torchvision.transforms import functional as F
#
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = [ToPILImage()] + transforms  # 将 ToPILImage 添加到开头
#
#     def __call__(self, image, target):
#         for t in self.transforms:
#             image, target = t(image, target)
#         return image, target
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string
#
#
#
# class Resize(object):
#     def __init__(self, min_size, max_size):
#         if not isinstance(min_size, (list, tuple)):
#             min_size = (min_size,)
#         self.min_size = min_size
#         self.max_size = max_size
#
#     def get_size(self, image_size):
#         w, h = image_size
#         size = random.choice(self.min_size)
#         max_size = self.max_size
#
#         if callable(image_size):
#             image_size = image_size()
#         elif isinstance(image_size, (list, tuple)):
#             image_size = tuple(image_size)
#         else:
#             raise ValueError("Unsupported image_size format")
#
#         if max_size is not None:
#             min_original_size = float(min((w, h)))
#             max_original_size = float(max((w, h)))
#             if max_original_size / min_original_size * size > max_size:
#                 size = int(round(max_size * min_original_size / max_original_size))
#
#         if (w <= h and w == size) or (h <= w and h == size):
#             return (h, w)
#
#         if w < h:
#             ow = size
#             oh = int(size * h / w)
#         else:
#             oh = size
#             ow = int(size * w / h)
#
#         return (oh, ow)
#
#     def __call__(self, image, target=None):
#         size = self.get_size(image.size)
#         image = F.resize(image, size)
#         if target is None:
#             return image
#         target = target.resize(image.size)
#         return image, target
#
#
# class RandomHorizontalFlip(object):
#     def __init__(self, prob=0.5):
#         self.prob = prob
#
#     def __call__(self, image, target):
#         if 'boxes' in target and random.random() < self.prob:
#             print(f"Before: Type of target['boxes']: {type(target['boxes'])}")
#             print(f"Before: Type of image: {type(image)}")
#
#             image = F.hflip(image)
#             target['boxes'][:, [0, 2]] = image.size[0] - target['boxes'][:, [2, 0]].clone()
#
#             print(f"After: Type of target['boxes']: {type(target['boxes'])}")
#             print(f"After: Type of image: {type(image)}")
#         return image, target
#
#
# class RandomVerticalFlip(object):
#     def __init__(self, prob=0.5):
#         self.prob = prob
#
#     def __call__(self, image, target):
#         if 'boxes' in target and random.random() < self.prob:
#             print(f"Type of image: {type(image)}")
#             image = F.vflip(image)
#             target['boxes'][:, [1, 3]] = image.size[1] - target['boxes'][:, [3, 1]]
#         return image, target
#
#
# class ColorJitter(object):
#     def __init__(self,
#                  brightness=None,
#                  contrast=None,
#                  saturation=None,
#                  hue=None,
#                  ):
#         self.color_jitter = torchvision.transforms.ColorJitter(
#             brightness=brightness,
#             contrast=contrast,
#             saturation=saturation,
#             hue=hue,)
#
#     def __call__(self, image, target):
#         image = self.color_jitter(image)
#         image = F.to_tensor(image)
#
#         if target is None:
#             return image, target
#
#         if 'boxes' in target:
#             target['boxes'] = target['boxes'].clone().detach().to(torch.float32)
#         if 'masks' in target:
#             target['masks'] = target['masks'].clone().detach().to(torch.float32)
#
#         return image, target
#
# class ToTensor(object):
#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         return image, target
#
# class ToPILImage(object):
#     def __call__(self, image, target):
#         if not isinstance(image, torch.Tensor):
#             # 如果输入不是张量，假设它是 PIL.Image.Image 对象，可以直接返回
#             return image, target
#
#         # Convert the torch tensor to PIL Image
#         pil_image = F.to_pil_image(image)
#
#         # Convert target['boxes'] to ndarray before modifying
#         target_boxes = target['boxes'].detach().numpy() if 'boxes' in target else None
#
#         # Apply the PIL Image transformation
#         pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
#         target_boxes[:, [0, 2]] = pil_image.width - target_boxes[:, [2, 0]]
#
#         # Convert the modified target_boxes back to torch tensor
#         target['boxes'] = torch.tensor(target_boxes) if 'boxes' in target else None
#
#         return pil_image, target
#
#
#
#
#
# class Normalize(object):
#     def __init__(self, mean, std, to_bgr255=True):
#         self.mean = mean
#         self.std = std
#         self.to_bgr255 = to_bgr255
#
#     def __call__(self, image, target=None):
#         if self.to_bgr255:
#             image = image[[2, 1, 0]] * 255
#         image = F.normalize(image, mean=self.mean, std=self.std)
#         if target is None:
#             return image
#         return image, target
