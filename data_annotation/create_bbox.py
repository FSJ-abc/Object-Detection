from matplotlib.patches import Rectangle
from scipy.signal import medfilt, correlate2d
import matplotlib.pyplot as plt
from xml.dom import minidom
import scipy.io as scio
from PIL import Image
import numpy as np
import os
import progressbar

def gen_xml_rect(xml_path, img_full_path, obj_list):
    """
    生成矩形框标注文件

    :param xml_path: 标注文件路径
    :param img_full_path: 图像文件路径
    :param obj_list: 目标位置信息列表 [(xmin, ymin, xmax, ymax, label), ...]
    """
    with Image.open(img_full_path) as img:
        width, height = img.size

    # 创建DOM树对象
    dom = minidom.Document()

    # 创建根节点
    root_node = dom.createElement('annotation')
    dom.appendChild(root_node)

    # 其他节点保持不变...
    folder_node = dom.createElement('folder')
    root_node.appendChild(folder_node)
    img_folder_path = os.path.basename(img_full_path)
    folder_node_text = dom.createTextNode(img_folder_path)
    folder_node.appendChild(folder_node_text)

    filename_node = dom.createElement('filename')
    root_node.appendChild(filename_node)
    file_name = img_full_path.split('\\')[-1]
    filename_node_text = dom.createTextNode(file_name)
    filename_node.appendChild(filename_node_text)

    path_node = dom.createElement('path')
    root_node.appendChild(path_node)
    path_node_text = dom.createTextNode(img_full_path)
    path_node.appendChild(path_node_text)

    source_node = dom.createElement('source')
    root_node.appendChild(source_node)

    database_node = dom.createElement('database')
    source_node.appendChild(database_node)
    database_node_text = dom.createTextNode('Unknown')
    database_node.appendChild(database_node_text)

    size_node = dom.createElement('size')
    root_node.appendChild(size_node)

    width_node = dom.createElement('width')
    size_node.appendChild(width_node)
    width_node_text = dom.createTextNode(str(width))
    width_node.appendChild(width_node_text)

    height_node = dom.createElement('height')
    size_node.appendChild(height_node)
    height_node_text = dom.createTextNode(str(height))
    height_node.appendChild(height_node_text)

    depth_node = dom.createElement('depth')
    size_node.appendChild(depth_node)
    depth_node_node_text = dom.createTextNode(str(3))
    depth_node.appendChild(depth_node_node_text)

    segmented_node = dom.createElement('segmented')
    root_node.appendChild(segmented_node)
    segmented_node_text = dom.createTextNode(str(0))
    segmented_node.appendChild(segmented_node_text)

    for obj in obj_list:
        object_node = dom.createElement('object')
        root_node.appendChild(object_node)

        name_node = dom.createElement('name')
        object_node.appendChild(name_node)
        name_node_text = dom.createTextNode(str(obj[4]))
        name_node.appendChild(name_node_text)

        pose_node = dom.createElement('pose')
        object_node.appendChild(pose_node)
        pose_node_text = dom.createTextNode('Unspecified')
        pose_node.appendChild(pose_node_text)

        truncated_node = dom.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_node_text = dom.createTextNode(str(0))
        truncated_node.appendChild(truncated_node_text)

        difficult_node = dom.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_node_text = dom.createTextNode('0')
        difficult_node.appendChild(difficult_node_text)

        bndbox_node = dom.createElement('bndbox')
        object_node.appendChild(bndbox_node)

        xmin_node = dom.createElement('xmin')
        bndbox_node.appendChild(xmin_node)
        xmin_node_text = dom.createTextNode(str(obj[0]))
        xmin_node.appendChild(xmin_node_text)

        ymin_node = dom.createElement('ymin')
        bndbox_node.appendChild(ymin_node)
        ymin_node_text = dom.createTextNode(str(obj[1]))
        ymin_node.appendChild(ymin_node_text)

        xmax_node = dom.createElement('xmax')
        bndbox_node.appendChild(xmax_node)
        xmax_node_text = dom.createTextNode(str(obj[2]))
        xmax_node.appendChild(xmax_node_text)

        ymax_node = dom.createElement('ymax')
        bndbox_node.appendChild(ymax_node)
        ymax_node_text = dom.createTextNode(str(obj[3]))
        ymax_node.appendChild(ymax_node_text)

    try:
        with open(xml_path, 'w', encoding='UTF-8') as fh:
            dom.writexml(fh, addindent=" ", newl="\n", encoding='UTF-8')
            # print('写入xml OK!')
    except Exception as err:
        print('错误信息：{0}'.format(err))


def save_plot_datasource_arrays(large_array, small_matrices_list, dy, save_path_before, save_path_after, objects,
                                xml_path):
    """
     绘制大矩阵，并在其中找到小矩阵列表中的小矩阵，然后用标记小矩阵的角点。

     参数:
     - large_array: 大数据矩阵
     - small_matrices_list: 小矩阵列表
     - dy: 偏移量
     - save_path_before: 处理前的图形保存的文件路径或文件名
     - save_path_after: 处理后的图形保存的文件路径或文件名
     - objects: 用于表示小矩阵标记的对象信息
     - xml_path: 生成的 XML 文件的保存路径或文件名
     """
    # 大矩阵的行数和列数
    rows_large, cols_large = large_array.shape

    # 处理大矩阵
    enlarge_array = large_array + np.arange(cols_large) * dy

    # 绘制大矩阵的图形
    ax = plt.gca()
    for j in range(cols_large):
        ax.plot(enlarge_array[:, j], color='blue', label='Large Array')

    # 保存处理前的图形
    if save_path_before:
        plt.savefig(save_path_before)

    # 标记区域的列表
    marked_regions = []

    # 在大矩阵中查找每个小矩阵
    for small_matrix in small_matrices_list:
        if small_matrix.shape[0] > enlarge_array.shape[0] or small_matrix.shape[1] > enlarge_array.shape[1]:
            # 跳过尺寸太大的小矩阵
            continue

        # 计算两个矩阵的相关性
        correlation = correlate2d(enlarge_array, small_matrix, mode='valid')

        # 设置一个阈值来确定匹配的准确度
        threshold = np.max(correlation) * 0.3
        match_indices = np.where(correlation >= threshold)

        # 对每个匹配位置进行处理
        for (x, y) in zip(*match_indices):
            # 计算小矩阵在大矩阵中的位置
            min_x, max_x = x, x + small_matrix.shape[0] - 1
            min_y, max_y = np.min(correlation), np.max(correlation)

            # 绘制矩形以标记匹配区域
            rect_x = [min_x, max_x, max_x, min_x, min_x]
            rect_y = [min_y, min_y, max_y, max_y, min_y]
            ax.plot(rect_x, rect_y, color='red', linestyle='dashed')
            print("有标记")
            # 添加到标记区域列表
            marked_regions.append([min_x, min_y, max_x, max_y, objects])

    # 生成 XML 文件
    if marked_regions:
        # 这里假设 gen_xml_rect 是一个自定义函数，用于生成 XML 文件
        gen_xml_rect(xml_path, save_path_before, marked_regions)

    # 保存处理后的图形
    if save_path_after:
        fig = plt.gcf()
        ax.figure = fig
        fig.savefig(save_path_after)

    plt.clf()  # 清除当前绘图

def split_matrix_circular(matrix, rows, cols):
    """
    将一个矩阵尽可能多地切割成指定份数的小矩阵，将前面的数据补充到后面形成环型数据

    参数:
    - matrix: 原始矩阵
    - rows: 水平方向上切割的份数
    - cols: 垂直方向上切割的份数

    返回:
    - submatrices: 切割后的小矩阵列表
    """
    matrix_rows, matrix_cols = matrix.shape

    row_size = matrix_rows // rows
    col_size = matrix_cols // cols

    submatrices = []

    for i in range(rows):
        for j in range(cols):
            row_start = i * row_size
            row_end = (i + 1) * row_size if i < rows - 1 else matrix_rows
            col_start = j * col_size
            col_end = (j + 1) * col_size if j < cols - 1 else matrix_cols

            submatrix = matrix[row_start:row_end, col_start:col_end]

            # 补充环型数据
            if row_end < matrix_rows:
                circular_rows = (row_start + row_size) % matrix_rows
                submatrix = np.concatenate((submatrix, matrix[circular_rows:row_end, col_start:col_end]), axis=0)

            if col_end < matrix_cols:
                circular_cols = (col_start + col_size) % matrix_cols
                submatrix = np.concatenate((submatrix, matrix[row_start:row_end, circular_cols:col_end]), axis=1)

            submatrices.append(submatrix)

    return submatrices


# 随机进行矩阵切片
def random_submatrix(matrix, submatrix_size):
    # 获取矩阵的形状
    matrix_rows, matrix_cols = matrix.shape

    # 确保子矩阵大小不超过原矩阵
    if submatrix_size[0] > matrix_rows or submatrix_size[1] > matrix_cols:
        raise ValueError("子矩阵大小超过了原矩阵的大小")

    # 随机选择子矩阵的左上角位置
    row_start = np.random.randint(0, matrix_rows - submatrix_size[0] + 1)
    col_start = np.random.randint(0, matrix_cols - submatrix_size[1] + 1)

    # 切片获取子矩阵
    submatrix = matrix[row_start:row_start + submatrix_size[0], col_start:col_start + submatrix_size[1]]

    return submatrix
# 画出线状图
# datasource_array:原始数据
# dy：数据偏移量
def plot_datasource_array(datasource_array,dy=5):
    # 求出 datasource_array 的行数和列数
    rows, cols = datasource_array.shape
    # 创建 Axes 对象
    ax = plt.gca()
    # 先画出第一列的图
    ax.plot(datasource_array[:, 0], color='blue')

    # 对每一列的数据进行偏移并设置横坐标
    for j in range(1, cols):
        # 循环的方式以每一列每一行为单位叠加偏移量
        for i in range(rows):
            datasource_array[i, j] = datasource_array[i, j] + dy * j

        # 使用 medfilt 对每一列的数据进行中值滤波
        datasource_array[:, j] = medfilt(datasource_array[:, j], kernel_size=7)

        # 绘制每一列偏移后的数据
        ax.plot(datasource_array[:, j], color='blue')

    # 显示图形
    plt.show()
# 根据缺陷所在位置从原始数据中提取缺陷的数据
def extractingDefectData(defectLogo, initData, dx=None):
    """
    通过矩阵中的位置矩阵从原始数据中提取出缺陷矩阵

    参数:
    - initData: 原始矩阵
    - defectLogo: 缺陷在矩阵中的位置集合
    - dx: 提取时的偏移量，如果提取纯缺陷矩阵，值为零

    返回:
    - result_list: 提取矩阵的列表
    """
    firstColumn = initData[:, 0]

    result_list = []

    for i in range(defectLogo.shape[0]):
        # 通过 [0][0] 从元组中获取实际值
        height_low = np.where(firstColumn == defectLogo[i, 0])[0][0]
        height_high = np.where(firstColumn == defectLogo[i, 1])[0][0]
        width_low = int(defectLogo[i, 2])
        width_high = int(defectLogo[i, 3])

        if width_high < width_low:
            temp_1 = initData[height_low:height_high, width_low:194]
            temp_2 = initData[height_low:height_high, 2:width_high]
            result_list.append(np.concatenate((temp_1, temp_2), axis=1))
        else:
            result_list.append(initData[height_low:height_high, width_low:width_high])

    return result_list

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_data(data_init_address, data_logo_address, img_noRectangle_address=None, img_rectangle_address=None, xml_save_address=None, partition=None):
    # 从 data.mat 读取数据
    data_init_dict = scio.loadmat(data_init_address)
    data_logo_dict = scio.loadmat(data_logo_address)

    # 获取实际的数据，使用 "Data" 键
    data_value = data_init_dict.get("Data", None)
    data_logo = data_logo_dict.get("m_miDoneDataFea", None)

    if data_value is not None and data_logo is not None:
        # 提取logo的有效数字
        logo_array = data_logo[:, 8:12]

        # 提取第二个子数组
        data_array = data_value[0][0][1]

        # 将 array2 作为 datasource_array
        datasource_array = data_array[:, 1:194]

        # 输出数据转换完成的提示
        print("数据已成功转换")

        # # 沿着axis=0的方向计算每一列的均值
        # column_means = np.mean(datasource_array, axis=0)
        # # 将每一行同时减去对应列的均值
        # centered_matrix = datasource_array - column_means
        # print("数据处理完成")
        # 画线状图
        # plot_datasource_array(datasource_array)
        # 调用提取缺陷数据的函数，并将提取出的纯缺陷保存在defect_data_list中
        defect_data_list = extractingDefectData(logo_array, data_array)

        # 首先对原始矩阵均分，均分成指定份数
        data_list = split_matrix_circular(datasource_array, partition*100, partition)

        # 循环处理当前的数据列表
        for i, data in enumerate(data_list):
            # for j, defect_data in enumerate(defect_data_list):
            # 构建保存路径
            save_path_before = os.path.join(img_noRectangle_address,
                                            f'before_{i}.png') if img_noRectangle_address else None
            save_path_after = os.path.join(img_rectangle_address,
                                           f'after_{i}.png') if img_rectangle_address else None
            xml_path = os.path.join(xml_save_address, f'annotation_{i}.xml') if xml_save_address else None

            # Ensure directories exist
            for path in (
            os.path.dirname(save_path_before), os.path.dirname(save_path_after), os.path.dirname(xml_path)):
                ensure_directory(path)

            save_plot_datasource_arrays(data, defect_data_list, 5, save_path_before, save_path_after, 'defect', xml_path)
            # 打印提示信息
            print(f'第{i}张图片已保存')
        print("图片处理完成")
        return defect_data_list
    else:
        print("未找到有效的数据键（Data）")
        return None

# 调用主处理函数
data_init_address = f"D:/管道文件/No.3/data.mat"
data_logo_address = f"D:/管道文件/No.3/DoneDataFea.mat"
img_noRectangle_address = f"./imgNoRectangle"
img_rectangle_address = f"./imgRectangle"
xml_save_address = f"./VOCdevkit/VOC2012/Annotations"
defect_data_list = process_data(data_init_address, data_logo_address, img_noRectangle_address, img_rectangle_address, xml_save_address, partition=5)
