
# 图像路径和 XML 文件路径
img_path = "after_plot.png"
xml_path = "example_annotation.xml"

# 矩形框坐标和标签
bounding_box = [50, 50, 300, 300, 'object1']
obj_list = [bounding_box]

# 生成 XML 文件
gen_xml_rect(xml_path, img_path, obj_list)
