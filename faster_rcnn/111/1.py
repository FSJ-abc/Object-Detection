"""
author:Sxiangkai
date:2020.7.9
objective:This program mainly realizes the visualization of MFL data
Function description of each function:
    f_interpolation_v1:Base value correction of magnetic flux leakage data
    filtering:Multi resolution image enhancement
    mapscale:To achieve one-to-one correspondence between data and color map, the resolution is set to 0.001
    datatoRgb:main function to activative Image conversion of magnetic flux leakage data
"""
import os
import re
import numpy as np
import scipy.io as scio
import matplotlib
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
# import seaborn as sns
# from f_lineadptive import f_lineadptive
from sklearn.neighbors import NearestNeighbors

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
        mid.append(np.median(data[:, i]))#计算中位数
    data = data-np.ones(shape=(data.shape[0], 1))*mid+np.ones(shape=(data.shape[0],data.shape[1]))*np.mean(mid)

    return data


def filtering(Axialdata,ksize2):
    """
    :param Axialdata: 原始轴向数据
    :param ksize1: 高斯滤波的模板大小,类型为tuple
    :param ksize2: 均值滤波的模板大小,类型为tuple
    """
    # fdata = cv2.GaussianBlur(Axialdata, ksize2, 1)#高斯滤波
    fdata1 = cv2.blur(Axialdata, ksize2)#滤波
    data_f = Axialdata-fdata1
    # data_flog = np.log(data_f-data_f.min()+1)

    return data_f



def dbscan(X, eps, min_samples):
    # 定义核心对象集合和聚类标签
    core_objects = set()
    clusters = []

    # 计算每个样本的邻近点数量
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, indices = neighbors.kneighbors(X)

    # 找到核心对象
    for i, distance in enumerate(distances):
        if len(distance) >= min_samples:
            core_objects.add(i)

    # 开始聚类
    unvisited = set(range(len(X)))
    while len(core_objects) > 0:
        current_core = core_objects.pop()
        cluster = set()
        cluster.add(current_core)
        unvisited.remove(current_core)

        while len(cluster) > 0:
            current_object = cluster.pop()
            neighbors = indices[current_object]
            for neighbor in neighbors:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    cluster.add(neighbor)
                    if neighbor in core_objects:
                        core_objects.remove(neighbor)

        clusters.append(list(cluster))

    return clusters



# def colormap(data, map_scale):
#     """
#     :param data: 原始漏磁数据
#     :param map_scale: 颜色图对应矩阵
#     :return: RGB颜色矩阵
#     """
#     data = np.array(data)
#     # averdata=np.mean(data[:,36:])
#     # stddata = np.std(data[:,36:])
#     averdata = np.mean(data)
#     stddata = np.std(data)
#     maxvalue = averdata+5*stddata
#     minvalue = averdata-2*stddata
#     # for i in range(data.shape[1]):
#     #     top_k_max = data[:,i].argsort()[::-1][0:top_k]
#     #     top_k_min = data[:, i].argsort()[::1][0:top_k]
#     #     maxlist.append(data[top_k_max])
#     #     minlist .append(data[top_k_min])
#     # maxaver = np.mean(np.array(maxlist))
#     # minaver = np.mean(np.array(minlist))
#     # if maxvalue > maxaver:
#     #     diff1 = maxvalue-maxaver
#     #     maxvalue = maxaver+0.1*diff1
#     # if minvalue < minaver:
#     #     diff2 = minaver - minvalue
#     #     minvalue = minvalue + 0.5*diff2
#     # # minvalue = 100
#     # # maxvalue = 400
#     data = (data-minvalue)/(maxvalue-minvalue)
#     data = np.ceil(data*1023)
#     data[data > 1023] = 1023
#     data[data < 0] = 0
#     finaldata = np.transpose(data)
#     # finaldata = finaldata[::-1]  # 为了可视化效果保证未添加前原来的最下边是第一个通道数据
#     finaldata = finaldata.astype(np.int16)
#     out = map_scale[finaldata, :]
#     return out


# def colormap(data, map_scale):
#     """
#     :param data: 原始漏磁数据
#     :param map_scale: 颜色图对应矩阵
#     :return: RGB颜色矩阵
#     """
#     data = np.array(data)
#     averdata = np.mean(data)
#     stddata = np.std(data)
#     maxvalue = averdata + 5 * stddata
#     minvalue = averdata - 2 * stddata
#
#     # 非线性映射（指数映射）
#     data = (data - minvalue) / (maxvalue - minvalue)
#     data = np.exp(2 * data) - 1
#
#     data = (data - np.min(data)) / (np.max(data) - np.min(data))
#     data = np.ceil(data * 1023)
#     data[data > 1023] = 1023
#     data[data < 0] = 0
#     finaldata = np.transpose(data)
#     finaldata = finaldata.astype(np.int16)
#     out = map_scale[finaldata, :]
#     return out

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

def grey_erosion(channel):
    kernel = np.ones((3, 3), np.uint8)  # 定义腐蚀操作的结构元素
    eroded = cv2.erode(channel, kernel, iterations=1)  # 对通道进行灰度腐蚀操作
    return eroded
def grey_dilation(channel):
    kernel = np.ones((3, 3), np.uint8)  # 定义膨胀操作的结构元素
    dilated = cv2.dilate(channel, kernel, iterations=1)  # 对通道进行灰度膨胀操作
    return dilated

def sharpen_image(image):
    # 创建锐化滤波器
    kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])

    # 对图像应用锐化滤波器
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def desharpen(image):
    # 去锐化操作
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def power_mapping(data, gamma):
    return np.power(data, gamma)
def colormap(data, map_scale, gamma):
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
    finaldata = np.transpose(data)
    finaldata = finaldata.astype(np.int16)

    # # 获取红色通道
    # red_channel = finaldata[0]  # 修改这里，仅使用一个索引获取红色通道
    # red_channel = red_channel.flatten()  # 修改这里，使用 flatten() 将红色通道转换为一维数组
    #
    # # 获取蓝色通道
    # blue_channel = finaldata[2]  # 修改这里，仅使用一个索引获取蓝色通道
    # blue_channel = blue_channel.flatten()  # 修改这里，使用 flatten() 将蓝色通道转换为一维数组
    #
    # # 对红色通道进行灰度腐蚀
    # red_channel_eroded = grey_erosion(red_channel)
    # red_channel_eroded = red_channel_eroded.flatten()  # 修改这里，使用 flatten() 将红色通道腐蚀后的结果转换为一维数组
    #
    # # 对蓝色通道进行灰度膨胀
    # blue_channel_dilated = grey_dilation(blue_channel)
    # blue_channel_dilated = blue_channel_dilated.flatten()  # 修改这里，使用 flatten() 将蓝色通道膨胀后的结果转换为一维数组
    #
    # # 将修改后的通道数据合并回原始图像
    # finaldata[:, 0] = red_channel_eroded[:finaldata.shape[0]]  # 修改这里，直接更新红色通道的第一列，并确保形状匹配

    out = map_scale[finaldata, :]
    return out



def generate_detection_boxes(image_path, threshold_r, threshold_g, threshold_b, window_size, stride):
    # 加载图像
    image = cv2.imread(image_path)

    # 分离图像的RGB通道
    r, g, b = cv2.split(image)

    # 获取图像尺寸和窗口大小
    height, width = r.shape[:2]
    window_height, window_width = window_size

    # 创建空白图像，用于绘制检测框
    result = image.copy()

    # 遍历图像并生成检测框
    for y in range(0, height - window_height + 1, stride):
        for x in range(0, width - window_width + 1, stride):
            # 提取当前窗口的图像块
            r_block = r[y:y + window_height, x:x + window_width]
            g_block = g[y:y + window_height, x:x + window_width]
            b_block = b[y:y + window_height, x:x + window_width]

            # 根据阈值生成掩码
            mask_r = np.zeros_like(r_block)
            mask_g = np.zeros_like(g_block)
            mask_b = np.zeros_like(b_block)

            mask_r[r_block > threshold_r] = 255
            mask_g[g_block > threshold_g] = 255
            mask_b[b_block > threshold_b] = 255

            # 查找当前窗口中的检测框边界
            contours_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours_r:
                x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
                if cv2.contourArea(contour) > 0:  # 只绘制面积大于0的轮廓
                    result[y + y_r:y + y_r + h_r, x + x_r:x + x_r + w_r] = (0, 119, 246)

            for contour in contours_g:
                x_g, y_g, w_g, h_g = cv2.boundingRect(contour)
                if cv2.contourArea(contour) > 0:
                    result[y + y_g:y + y_g + h_g, x + x_g:x + x_g + w_g] = (0, 119, 246)

            for contour in contours_b:
                x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
                if cv2.contourArea(contour) > 0:
                    result[y + y_b:y + y_b + h_b, x + x_b:x + x_b + w_b] = (0, 119, 246)

    return result

def datatoRgb(sourcepath, savepath,savepath_sharpen, axialCol, index=0, start=None, space=None, length=None):
    """
    :param sourcepath: 源数据文件夹路径
    :param savepath: 转换后的漏磁图片保存路径
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
    k=0
    flag_ = ""
    for sourcedata in sourcedatas:
        if sourcedata.endswith(('.mat', '.txt')):
            k=k+1
            data = scio.loadmat(os.path.join(sourcepath, sourcedata))
            print(sourcedata + 'image conversion ends\n')
            keylist =list(data.keys())
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
            # start = round(float(data_[0][0] * 0.002), 3)
            print(sourcedata + 'image conversion ends-bin', end='\n')
            flag_ = "bin"
        else:
            print("输入数据类型错误")
            # mfData = 0-np.array(data['data'][:, 0:])
        length = np.int16(2.5*(mfData.shape[1]+36))  # 可以自己设定

        # start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
        for i in range(0, mfData.shape[0], space):
            if i < mfData.shape[0] - length:
                data = mfData[i:i + length, :]
                data = f_interpolation_v1(mfDataDisplay=data)#基值校正
                data_add = data[:, -36:]
                data = np.hstack((data_add, data))
                # sns.distplot(data)
                # matplotlib.pyplot.show()
                # data = f_lineadptive(X=data,p1=data.shape[0],p2=10,p3=0.2)
                data = filtering(Axialdata=data, ksize2=(19, 19))
                out = colormap(data=data,map_scale=map,gamma=1)
                matplotlib.image.imsave(savepath + '\\' +
                                        str(format(start + index * mile, '.3f')) + '.jpg', out)

                # path=savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg'
                # path2=savepath_sharpen + '\\' +str(format(start + index * mile, '.3f')) + '.jpg'


                # inverted_image = 255 - image_gray  # 颜色反转
                inverted_image2 = convert_to_embossed_gray(data)
                matplotlib.image.imsave(savepath_sharpen + '\\' + str(format(start + index * mile, '.3f')) + '.jpg',
                                        inverted_image2,
                                        cmap='gray')
                # cv2.imshow('浮雕', inverted_image2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite(savepath_kai_, inverted_image2)
                # transposed_image = np.transpose(inverted_image)
                ########上述代码为调用灰度浮雕#######

                # result_image =  generate_detection_boxes(image_path=path, threshold_r=40, threshold_g=200, threshold_b=200,
                #                                     window_size=(3,3),stride=2)
                # cv2.imshow('Result', result_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # #霍夫检测
                # image=savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg'
                # a=huofu(image)
                # # 显示结果
                # cv2.imshow('Image', a)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # #锐化
                # image=cv2.imread(savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg')
                # a=sharpen_image(image)
                # savepath_sharpen_=savepath_sharpen + '\\' +str(format(start + index * mile, '.3f')) + '.jpg'
                # cv2.imwrite(savepath_sharpen_, a)
                # #去锐化
                # desharpen=desharpen(savepath_sharpen_)
                # cv2.imshow(desharpen)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # X = np.array(out)
                # clusters = dbscan(X, eps=2, min_samples=2)
                # for i, cluster in enumerate(clusters):
                #     print(f"Cluster {i + 1}: {cluster}")
                # matplotlib.image.imsave(savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg', out)

                index = index + 1
            else:
                data = mfData[i:, :]
                data = f_interpolation_v1(mfDataDisplay=data)#基值校正
                data2 = data[:, -36:]
                data = np.hstack((data2, data))#补36个通道
                data = filtering(Axialdata=data, ksize2=(19, 19))
                out = colormap(data=data,map_scale=map,gamma=1)
                # matplotlib.image.imsave(savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg', out)\
                if (flag_ == "mat") :
                    plt.imsave(savepath+'\\' + str(format(start + index * mile, '.3f')) + '.jpg',out)
                    # start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
                # matplotlib.image.imsave(savepath+'\\' + str(sourcedata[:-4]) + '-' + str(format(start + index * mile, '.3f')) + '.jpg',datacolor)
                if (flag_ == "bin") :
                    plt.imsave(savepath+'\\' + str(format(start + index * mile, '.3f')) + '.jpg',out)
                    start = start + data.shape[1] * 0.002 + index * mile
                # matplotlib.image.imsave(savepath + '\\' +str(sourcedata[:-4])+'-'+
                #                         str(format(start + index * mile, '.3f')) + '.jpg', out)

                index = 0
                break



#下列代码在应用红色通道腐蚀，蓝色通道膨胀时解注释
# def datatoRgb(sourcepath, savepath, axialCol, index=0, start=None, space=None, length=None):
#     """
#     :param sourcepath: 源数据文件夹路径
#     :param savepath: 转换后的漏磁图片保存路径
#     :param index: 起始图片标号，默认为0
#     :param start: 数据对应的起始里程
#     :param space: 每张图片之间的滑动步长
#     :param length: 每张图片的长度可不设置,程序默认为2倍的通道数
#     """
#     sourcedatas = os.listdir(sourcepath)
#     # sourcedatas = sorted(sourcedatas, key=lambda i: int(re.search(r'(\d+)', i).group()))
#     mile = space * 0.002
#     map = scio.loadmat('./jet.mat')
#     map = map['map_scale']
#     k=0
#     flag_ = ""
#     for sourcedata in sourcedatas:
#         if sourcedata.endswith(('.mat', '.txt')):
#             k=k+1
#             data = scio.loadmat(os.path.join(sourcepath, sourcedata))
#             print(sourcedata + 'image conversion ends\n')
#             keylist =list(data.keys())
#             AxialCol = data_extract(axialCol) # 根据不同数量的传感器加载不同的轴向数据索引
#             extract_data = extract(data[keylist[-1]], AxialCol)
#             mfData = np.array(extract_data[:, 1:]) # 根据实际数据的键值对做对应的修改
#             start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
#             flag_ = "mat"
#         elif sourcedata.endswith(('.bin', '.txt')):
#             bin_data = data_input(os.path.join(sourcepath, sourcedata)) # 读取数据，将数据与通道数格式匹配
#             AxialCol = data_extract(12) # 根据不同数量的传感器加载不同的轴向数据索引
#             data_ = extract(bin_data, AxialCol)
#             mfData = data_[:, 1:]
#             # start = round(float(data_[0][0] * 0.002), 3)
#             print(sourcedata + 'image conversion ends-bin', end='\n')
#             flag_ = "bin"
#         else:
#             print("输入数据类型错误")
#         # mfData = 0-np.array(data['data'][:, 0:])
#         length = np.int16(2.5*(mfData.shape[1]+36)) # 可以自己设定
#         # start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
#         for i in range(0, mfData.shape[0], space):
#             if i < mfData.shape[0] - length:
#                 data = mfData[i:i + length, :]
#                 data = f_interpolation_v1(mfDataDisplay=data)#基值校正
#                 data_add = data[:, -36:]
#                 data = np.hstack((data_add, data))
#                 # sns.distplot(data)
#                 # matplotlib.pyplot.show()
#                 # data = f_lineadptive(X=data,p1=data.shape[0],p2=10,p3=0.2)
#                 data = filtering(Axialdata=data, ksize2=(19, 19))
#                 out = colormap(data=data,map_scale=map,gamma=1)
#                 matplotlib.image.imsave(savepath + '\\' +
#                                         str(format(start + index * mile, '.3f')) + '.jpg', out)
#                 # matplotlib.image.imsave(savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg', out)
#
#                 index = index + 1
#             else:
#                 data = mfData[i:, :]
#                 data = f_interpolation_v1(mfDataDisplay=data)#基值校正
#                 data2 = data[:, -36:]
#                 data = np.hstack((data2, data))#补36个通道
#                 data = filtering(Axialdata=data, ksize2=(19, 19))
#                 out = colormap(data=data,map_scale=map,gamma=1)
#                 # matplotlib.image.imsave(savepath + '\\' +str(format(start + index * mile, '.3f')) + '.jpg', out)\
#                 if (flag_ == "mat") :
#                     plt.imsave(savepath+'\\' + str(format(start + index * mile, '.3f')) + '.jpg',out)
#                     # start = round(float(data[keylist[-1]][0][0] * 0.002), 3)
#                 # matplotlib.image.imsave(savepath+'\\' + str(sourcedata[:-4]) + '-' + str(format(start + index * mile, '.3f')) + '.jpg',datacolor)
#                 if (flag_ == "bin") :
#                     plt.imsave(savepath+'\\' + str(format(start + index * mile, '.3f')) + '.jpg',out)
#                     start = start + data.shape[1] * 0.002 + index * mile
#                 # matplotlib.image.imsave(savepath + '\\' +str(sourcedata[:-4])+'-'+
#                 #                         str(format(start + index * mile, '.3f')) + '.jpg', out)
#
#                 index = 0
#                 break




if __name__ == '__main__':
    # 源数据文件夹路径
    sourcepath = r'C:\Users\Administrator\Desktop\b\data-lu'
    # sourcepath = r'C:\Users\Administrator\Desktop\博士\data\2020\matdata'
    # sourcepath = r'C:\Users\Administrator\Desktop\12寸数据\d'
    #灰度图
    savepath = r'C:\Users\Administrator\Desktop\b\result-f'
    #彩色图
    savepath_sharpen=r'C:\Users\Administrator\Desktop\b\result-sharpen'
    datatoRgb(sourcepath=sourcepath, savepath=savepath_sharpen,savepath_sharpen=savepath, index=0, start=0.002, space=100, length=500, axialCol= '12S1')
