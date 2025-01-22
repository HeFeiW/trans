import os
import pickle
import json
import numpy as np

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (dict, list, tuple, set)):
        # 处理容器类型
        if isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [numpy_to_python(item) for item in obj]
        else:  # set
            return [numpy_to_python(item) for item in obj]  # 转换为列表以确保 JSON 兼容
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        # 处理 numpy 标量类型
        return obj.item()  # 使用 item() 更高效地转换为 Python 原生类型
    return obj

def load_and_save_pkl_to_json(task_name, skiping_category):
    # 定义输入和输出路径
    pkl_folder = f'{task_name}-train'
    json_folder = f'{task_name}-train-json'

    # 如果json文件夹不存在，则创建
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    # 遍历pkl下的所有子文件夹，每个子文件夹是一个分类
    for subfolder in os.listdir(pkl_folder):
        #获取子文件夹名称
        category = subfolder
        if category in skiping_category:
            continue
        for filename in os.listdir(os.path.join(pkl_folder, subfolder)):
            if filename.endswith('.pkl'):
                pkl_file_path = os.path.join(pkl_folder, subfolder, filename)
                
                # 打开并加载pkl文件
                with open(pkl_file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    
                # 将NumPy数组转换为Python原生类型
                converted_data = numpy_to_python(loaded_data)
                
                # 构造json文件路径
                json_file_path = os.path.join(json_folder, subfolder, filename.replace('.pkl', '.json'))
                # 如果json文件夹不存在，则创建
                if not os.path.exists(os.path.dirname(json_file_path)):
                    os.makedirs(os.path.dirname(json_file_path))
                # 将数据保存为json文件
                with open(json_file_path, 'w') as json_file:
                    json.dump(converted_data, json_file, ensure_ascii=False, indent=4)

                print(f"已保存 {json_file_path}")

# 使用示例
task_name = "packing-with-error"
skiping_category = ['color', 'depth']
load_and_save_pkl_to_json(task_name, skiping_category)
