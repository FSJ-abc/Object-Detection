import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import cv2
import matplotlib
# from method import *
# import timm

def data_input(txt):
    if txt.endswith(('.bin', '.txt')):
        # data_mat = 0
        binData = np.fromfile(txt, dtype='float')  # 以二进制方式读取bin文件并存储
        binData_shape0 = binData.shape[0]
        # print(binData.shape[0])
        # if np.mod(binData.shape[0], 1206) == 0:  # 判断是否能整除
        #     print('S2-1206')
        #     print(txt)
        #     AnalysisData = np.reshape(binData, (-1, 1206))
        #     my_mat = txt[:-4] + '.mat'
        #     if AnalysisData.size != 0:
        #       #  scio.savemat(os.path.join(path_save, my_mat), {'A': AnalysisData})
        #         data.append(AnalysisData)
        #     c = 1
        if np.mod(binData.shape[0], 1046) == 0:
            print('S1-1046')
            AnalysisData = np.reshape(binData, (-1, 1046))

        elif np.mod(binData.shape[0], 709) == 0:
            print('S2-709-323C')
            AnalysisData = np.reshape(binData, (-1, 709))

        elif np.mod(binData.shape[0], 621) == 0:
            print('323A-621')
            AnalysisData = np.reshape(binData, (-1, 621))

        elif np.mod(binData.shape[0], 561) == 0:
            print('S2-561')
            AnalysisData = np.reshape(binData, (-1, 561))

        elif np.mod(binData.shape[0], 486) == 0:
            print('S1-486')
            AnalysisData = np.reshape(binData, (-1, 486))

        elif np.mod(binData.shape[0], 606) == 0:
            print('S1-606')
            AnalysisData = np.reshape(binData, (-1, 606))

        elif np.mod(binData.shape[0], 699) == 0:
            print('S2-699')
            AnalysisData = np.reshape(binData, (-1, 699))

        elif np.mod(binData.shape[0], 1109) == 0:
            print('S2-1109')
            AnalysisData = np.reshape(binData, (-1, 1109))

        elif np.mod(binData.shape[0], 821) == 0:
            print('S2-821')
            AnalysisData = np.reshape(binData, (-1, 821))

        elif np.mod(binData.shape[0], 669) == 0:
            print('S2-669')
            AnalysisData = np.reshape(binData, (-1, 669))

        elif np.mod(binData.shape[0], 1205) == 0:
            print('S2-1205')
            AnalysisData = np.reshape(binData, (-1, 1205))


        elif np.mod(binData.shape[0], 469) == 0:
            print('S2-469')
            AnalysisData = np.reshape(binData, (-1, 469))

    return AnalysisData


def data_extract(DetectorSize):
    # AxialCol = 0
    # DetectorSize = int(input('输入内检测器尺寸：8/10/12/14/16/32/212'))
    if DetectorSize == 8:
        SourceData = scio.loadmat('./8寸AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == '12S1':
        SourceData = scio.loadmat('12寸AxialColS1.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == '323A':
        SourceData = scio.loadmat('323AAxialColS2.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 10:
        SourceData = scio.loadmat('10寸AxialColS2.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 82:
        SourceData = scio.loadmat('./8寸AxialCol2.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 12:
        SourceData = scio.loadmat('./12寸AxialColS2.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 14:
        SourceData = scio.loadmat('./14寸AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 16:
        SourceData = scio.loadmat('./16寸AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 32:
        SourceData = scio.loadmat('./32寸AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 3231:
        SourceData = scio.loadmat('./32寸SS1AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
        print(len(SourceData['AxialCol'][0]))
    elif DetectorSize == 212:
        SourceData = scio.loadmat('./S212寸AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 162:
        SourceData = scio.loadmat('./16寸S2数据.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 457:
        SourceData = scio.loadmat('./457S1AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 322:
        SourceData = scio.loadmat('./AxialCol32S2.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 219:
        SourceData = scio.loadmat('./219AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 1016:
        SourceData = scio.loadmat('./1016AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 508:
        SourceData = scio.loadmat('./508AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 762:
        SourceData = scio.loadmat('./762AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 610:
        SourceData = scio.loadmat('./610AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 5082:
        SourceData = scio.loadmat('./5082AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    elif DetectorSize == 4572:
        SourceData = scio.loadmat('./457S2AxialCol.mat')
        AxialCol = np.array(SourceData['AxialCol'])
    else:
        AxialCol = 0
        print('请输入正确的传感器尺寸')

    return AxialCol


def extract(data, AxialCol):    
    extractdata = data[:, AxialCol[0] - 1]

    return extractdata


def f_interpolation_v1(mfDataDisplay):
    """
    :param mfDataDisplay: 待基值矫正的漏磁数据
    :return: 基值矫正后的数据
    """
    data = np.array(mfDataDisplay)
    mid = []
    for i in range(data.shape[1]):
        mid.append(np.median(data[:, i]))  # 计算中位数

    data = data - np.ones(shape=(data.shape[0], 1)) * mid + np.ones(shape=(data.shape[0], data.shape[1])) * np.mean(mid)

    return data


def process_array(arr,base_value,quotient):
    """
    :param arr: 原始轴向数据
    :param base_value: 基值倍率 数越大，滤波越明显
    :param quotient: 基值/原数据 的商 数越大，滤波越明显
    """
    target_array = np.zeros_like(arr,dtype=np.float64)
    np.copyto(target_array, arr)
    # 计算每列的中位数
    column_medians = np.median(arr, axis=0)
    column_medians2=[x * base_value for x in column_medians]
    # 遍历每列
    for i in range(arr.shape[1]):
        column = arr[:, i]  # 获取当前列的数据
        a=target_array[:, i]
        max_value = np.max(a)
        max_index = np.argmax(a)
        print(max_value)
        print(max_index)
        # 找到中位数所在的索引
        # median_index = np.where(column == column_medians[i])[0][0]

        # 遍历当前列的每个元素
        for j in range(arr.shape[0]):
            # 判断当前元素是否小于等于中位数
            if column[j] <= column_medians2[i]:
                # 将小于等于中位数的元素替换为平均数
                column[j] = column_medians[i]
        for j in range(arr.shape[0]):
            if a[j] > column_medians2[i]:
                b = column_medians[i] / column[j]
                # print(b)
                if b > quotient:
                    column[j] = column_medians2[i]
                else:
                    if j >= 5 and j < arr.shape[0] - 6:  # 判断是否存在足够的上方和下方元素
                        column[max(0, j - 5): j + 6] = a[max(0, j - 5): j + 6]  # 将上下5个共11个元素保持原值不变
                    elif j >= 5:  # 如果下方元素不足，则尽量替换更多
                        column[max(0, j - 5): j + 1] = a[max(0, j - 5): j + 1]  # 将上方5个和剩余下方元素保持原值不变
                    else:  # 如果上方元素不足，则尽量替换更多
                        column[j: j + 6] = a[j: j + 6]  # 将下方6个和剩余上方元素保持原值不变


    return arr

# 遍历数组，取出30%大的数据并缩减10%
def data_reduction(array):
    flattened_arr = np.sort(array, axis=None)[::-1]
    num_elements = int(len(flattened_arr) * 0.06)
    extracted_data = flattened_arr[:num_elements]
    median_value = np.median(extracted_data)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > median_value:
                array[i, j] = median_value*1.5

    return array



def filtering(Axialdata, ksize2):
    """
    :param Axialdata: 原始轴向数据
    :param ksize1: 高斯滤波的模板大小,类型为tuple
    :param ksize2: 均值滤波的模板大小,类型为tuple
    """
    # fdata = cv2.GaussianBlur(Axialdata, ksize2, 1)#高斯滤波
    fdata1 = cv2.blur(Axialdata, ksize2)  # 滤波
    # data_f = Axialdata - fdata1
    # data_flog = np.log(data_f-data_f.min()+1)

    return fdata1

def filtering2(Axialdata, ksize2):
    fdata1 = cv2.blur(Axialdata, ksize2)  # 使用均值滤波对输入图像进行滤波
    data_f = cv2.subtract(Axialdata, fdata1)  # 逐像素相减
    # 对结果进行限制，使其在一定范围内
    min_value = 1  # 最小值
    max_value = 100  # 最大值
    data_f = cv2.max(min_value, cv2.min(data_f, max_value))
    return data_f  # 返回滤波后的结果



def bilateral_filter_gray(image, d, sigma_color, sigma_space):
    """
    :param image: 输入灰度图像
    :param d: 滤波器直径
    :param sigma_color: 颜色空间的标准方差
    :param sigma_space: 坐标空间的标准方差
    :return: 滤波后的图像
    """
    image = image.astype(np.float32)
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    data_f = image - filtered_image*0.5
    return data_f

def power_mapping(data, gamma):
    return np.power(data, gamma)


def colormap(data, gamma):
    data = np.array(data)
    averdata = np.mean(data)
    stddata = np.std(data)
    maxvalue = averdata + 5 * stddata
    minvalue = averdata - 2 * stddata

    data = (data - minvalue) / (maxvalue - minvalue)

    # Apply power mapping
    data = power_mapping(data, gamma)

    data = np.ceil(data * 255)
    data[data > 255] = 255
    data[data < 0] = 0
    finaldata = np.transpose(data)
    finaldata = finaldata.astype(np.uint8)
    return finaldata


def colormap2(data, map_scale, gamma):
    data = np.array(data)
    averdata = np.mean(data)
    stddata = np.std(data)
    maxvalue = averdata + 5 * stddata
    minvalue = averdata - 2 * stddata

    data = (data - minvalue) / (maxvalue - minvalue)

    # 应用幂函数映射
    data = power_mapping(data, gamma)

    data = np.ceil(data * 1023)
    data[data > 1023] = 1023
    data[data < 0] = 0
    # finaldata = np.transpose(data)
    finaldata = data.astype(np.int16)
    out = map_scale[finaldata, :]
    return out






def count_white_pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    white_pixel_count = np.count_nonzero(binary_image)
    return white_pixel_count

# def segment_images_by_gray_value(image_path, bin_size, output_folder):
#     """
#     对图像进行按灰度值分割为255/x张（x为所选灰度值）并返回白色点倒数第二多的图片
#     :param image_path: 输入原图像
#     :param bin_size: 想要分割的灰度值
#     :param output_folder: 输出路径
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     num_bins = 256
#     bins = [(i, i + bin_size - 1) for i in range(0, num_bins, bin_size)]
#     segmented_images = []
#     white_pixel_counts = []
#     for i, (lower, upper) in enumerate(bins):
#         mask = cv2.inRange(gray_image, lower, upper)
#         segmented_image = cv2.bitwise_and(image, image, mask=mask)
#         segmented_images.append(segmented_image)
#         white_pixel_count = count_white_pixels(segmented_image)
#         white_pixel_counts.append(white_pixel_count)
#         output_path = os.path.join(output_folder, f'segmented_image_{i}.jpg')
#         cv2.imwrite(output_path, segmented_image)
#
#     sorted_counts = sorted(white_pixel_counts, reverse=True)
#     second_max_index = white_pixel_counts.index(sorted_counts[1])
#     second_max_image_path = os.path.join(output_folder, f'segmented_image_{second_max_index}.jpg')
#
#     return second_max_image_path

def segment_images_by_gray_value(image_path, bin_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_bins = 256
    bins = [(i, i + bin_size - 1) for i in range(0, num_bins, bin_size)]
    segmented_images = []
    white_pixel_counts = []
    densities = []  # 存储白点密度的列表
    for i, (lower, upper) in enumerate(bins):
        mask = cv2.inRange(gray_image, lower, upper)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        segmented_images.append(segmented_image)
        white_pixel_count = count_white_pixels(segmented_image)
        white_pixel_counts.append(white_pixel_count)
        # 计算白点密度
        density = white_pixel_count / (segmented_image.shape[0] * segmented_image.shape[1])
        densities.append(density)

        output_path = os.path.join(output_folder, f'segmented_image_{i}.jpg')
        cv2.imwrite(output_path, segmented_image)

    max_white_pixel_count = max(white_pixel_counts)
    max_index = white_pixel_counts.index(max_white_pixel_count)
    max_image_path = os.path.join(output_folder, f'segmented_image_{max_index}.jpg')

    return max_image_path, densities

def calculate_density(image, region):
    region_image = image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(binary_image)
    density = white_pixel_count / (region[2] * region[3])
    return density

def calculate_density(image, region):
    region_image = image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(binary_image)
    density = white_pixel_count / (region[2] * region[3])
    return density

def calculate_density(image, region):
    region_image = image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(binary_image)
    density = white_pixel_count / (region[2] * region[3])
    return density

def calculate_density(image, region):
    region_image = image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    white_pixel_count = cv2.countNonZero(binary_image)
    density = white_pixel_count / (region[2] * region[3])
    return density, (region[0] + region[2], region[1]), (region[0], region[1] + region[3])


#对按灰度值分割后的图像进行白色最密集的区域检测与删除白色最密集的区域的整行数据
def detect_densest_region(image_path, region_width, region_height):
    """
    :param image_path: 输入原图像
    :param region_width: 检测框长度
    :param region_height: 检测框宽度
    """

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    max_density = 0
    max_loc = (0, 0)

    for y in range(height - region_height + 1):
        for x in range(width - region_width + 1):
            region = (x, y, region_width, region_height)
            density, _, _ = calculate_density(image, region)
            if density > max_density:
                max_density = density
                max_loc = (x, y)

    top_left = max_loc
    bottom_right = (max_loc[0] + region_width, max_loc[1] + region_height)

    # 删除整行数据
    image[top_left[1]:bottom_right[1], :] = 0

    cv2.imwrite('marked_image.jpg', image)
    # cv2.imshow('Processed Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 在处理后的图像上再次进行红框检测
    processed_image = cv2.imread('marked_image.jpg')
    processed_height, processed_width = processed_image.shape[:2]

    max_density = 0
    max_loc = (0, 0)

    for y in range(processed_height - region_height + 1):
        for x in range(processed_width - region_width + 1):
            region = (x, y, region_width, region_height)
            density, _, _ = calculate_density(processed_image, region)
            if density > max_density:
                max_density = density
                max_loc = (x, y)

    top_left_2 = max_loc
    bottom_right_2 = (max_loc[0] + region_width, max_loc[1] + region_height)

    # 删除第二次检测的数据
    processed_image[top_left_2[1]:bottom_right_2[1], :] = 0

    # cv2.imshow('Processed Image with Second Detection', processed_image)
    # cv2.imwrite('marked_image_twice.jpg', processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return top_left, bottom_right,top_left_2,bottom_right_2


#浮雕
def convert_to_embossed_gray(image_path):

    # 获取图像的宽度和高度
    height, width = image_path.shape

    # 创建新的浮雕灰度图像
    embossed_image = np.zeros((height, width), dtype=np.uint8)

    # 定义偏移值
    offset = 128

    # 遍历图像的像素
    for i in range(1, height-1):
        for j in range(1, width-1):
            # 计算当前像素与右边像素的差值
            diff = int(image_path[i, j+1]) - int(image_path[i, j])

            # 将差值加上偏移值，并将结果限制在0到255之间
            embossed_value = np.clip(diff + offset, 0, 255)

            # 将结果灰度值赋给新的浮雕灰度图像的像素
            embossed_image[i, j] = embossed_value

    # 将边缘像素设置为255
    embossed_image[0, :] = 255
    embossed_image[height-1, :] = 255
    embossed_image[:, 0] = 255
    embossed_image[:, width-1] = 255

    # 返回浮雕灰度图像
    return embossed_image


def datatoGray(sourcepath, savepath_kai,savepath_fudiao,axialCol, index=0, start=None, space=None, length=None):
    """
      :param sourcepath: 源数据文件夹路径
      :param savepath: 转换后的漏磁灰度图片保存路径
      :param savepath_kai: 转换后的漏磁彩色图片保存路径
      :param savepath_fudiao: 转换后的漏磁浮雕图片保存路径
      :param index: 起始图片标号，默认为0
      :param start: 数据对应的起始里程
      :param space: 每张图片之间的滑动步长
      :param length: 每张图片的长度可不设置,程序默认为2倍的通道数
      """
    sourcedatas = os.listdir(sourcepath)
    # sourcedatas = sorted(sourcedatas, key=lambda i: int(re.search(r'(\d+)', i).group()))
    mile = space * 0.002
    map = scio.loadmat('./jet.mat')
    map = map['map_scale']
    k = 0
    flag_ = ""
    for sourcedata in sourcedatas:
        if sourcedata.endswith(('.mat', '.txt')):
            k = k + 1
            data = scio.loadmat(os.path.join(sourcepath, sourcedata))
            print(sourcedata + 'image conversion ends\n')
            keylist = list(data.keys())
            AxialCol = data_extract(axialCol)  # 根据不同数量的传感器加载不同的轴向数据索引
            extract_data = extract(data[keylist[-1]], AxialCol)
            mfData = np.array(extract_data[:, 1:])  # 根据实际数据的键值对做对应的修改
            start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
            flag_ = "mat"
        elif sourcedata.endswith(('.bin', '.txt')):
            bin_data = data_input(os.path.join(sourcepath, sourcedata))  # 读取数据，将数据与通道数格式匹配
            AxialCol = data_extract('12S1')  # 根据不同数量的传感器加载不同的轴向数据索引
            data_ = extract(bin_data, AxialCol)
            mfData = data_[:, 1:]

            print(sourcedata + 'image conversion ends-bin', end='\n')
            flag_ = "bin"
        else:
            print("输入数据类型错误")

        length = np.int16(2.5 * (mfData.shape[1] + 36))  # 可以自己设定



        for i in range(0, mfData.shape[0], space):
            if i < mfData.shape[0] - length:
                data = mfData[i:i + length, :]
                data = f_interpolation_v1(mfDataDisplay=data)
                data = process_array(arr=data,base_value=1.01,quotient=1.05)
                data_add = data[:, -36:]
                data = np.hstack((data_add, data))
                # data = bilateral_filter_gray(data, d=5, sigma_color=50, sigma_space=50)
                data = filtering(Axialdata=data, ksize2=(4,2))

                out = colormap(data=data, gamma=1)

                image_gray=out
                ########下列代码为调用灰度浮雕#######
                inverted_image2=convert_to_embossed_gray(image_gray)
                matplotlib.image.imsave(savepath_fudiao + '\\' + str(format(start + index * mile, '.3f')) + '.jpg', inverted_image2,cmap='gray')
                ########上述代码为调用灰度浮雕#######

                ########### 下列代码为转彩色图 #########

                filtered_image = cv2.medianBlur(image_gray, 3)

                color=colormap2(data=filtered_image,map_scale=map,gamma=1)
                matplotlib.image.imsave(savepath_kai + '\\' +str(format(start + index * mile, '.3f')) + '.jpg', color)
                print(str(format(start + index * mile)))
                ########### 上述代码为转彩色图 #########


                index = index + 1
            else:
                data = mfData[i:, :]
                data = f_interpolation_v1(mfDataDisplay=data)
                data = process_array(arr=data,base_value=1.01,quotient=1.05)
                data2 = data[:, -36:]
                data = np.hstack((data2, data))
                data = filtering(Axialdata=data, ksize2=(4,2))
                out = colormap(data=data, gamma=1)
                # plt.imsave(savepath + '\\' + str(format(start + index * mile, '.3f')) + '.jpg', out, cmap='gray')

                image_gray = out
                ########下列代码为调用灰度浮雕#######
                inverted_image2 = convert_to_embossed_gray(image_gray)
                matplotlib.image.imsave(savepath_fudiao + '\\' + str(format(start + index * mile, '.3f')) + '.jpg',
                                        inverted_image2, cmap='gray')
                ########上述代码为调用灰度浮雕#######

                ########### 下列代码为转彩色图 #########

                filtered_image = cv2.medianBlur(image_gray, 3)

                color = colormap2(data=filtered_image, map_scale=map, gamma=1)
                matplotlib.image.imsave(savepath_kai + '\\' + str(format(start + index * mile, '.3f')) + '.jpg', color)
                print(str(format(start + index * mile)))
                ########### 上述代码为转彩色图 #########
                index = 0
                break


if __name__ == '__main__':
    # Example usage

    sourcepath =r'E:\transfer learning\myself\faster_rcnn\111\data-lu'
    # savepath = r'C:\Users\Administrator\Desktop\b\result_gray'
    savepath_kai=r'D:\管道文件\No.3_output'
    savepath_fudiao=r'D:\管道文件\No.3_output_2'
    datatoGray(sourcepath=sourcepath,savepath_kai=savepath_kai,
                savepath_fudiao=savepath_fudiao,index=0, start=0.002, space=100, length=500, axialCol='12S1')
