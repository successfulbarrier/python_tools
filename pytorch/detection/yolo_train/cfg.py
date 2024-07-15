# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   cfg.py
# @Time    :   2023/11/15 13:00:19
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   配置文件加载

"""
    加载所有用到的参数
    1.模型文件
    2.数据集文件
    3.超参数文件
"""
import yaml

class Cfg:
    def __init__(self, cfg_path="test.yaml",root_path=None):
        # 项目根目录
        self.root_path = root_path
        # 加载超参数 
        self.cfg_hyp = self.root_path + "cfg/" + cfg_path
        self.cfg_hyp = self.load_yaml(self.cfg_hyp)
        # 加载模型配置文件路径和数据集配置文件
        self.cfg_model = self.root_path + "cfg/" + self.cfg_hyp["model"]
        self.cfg_dataset = self.root_path + "cfg/" + self.cfg_hyp["data"]
        self.cfg_dataset = self.load_yaml(self.cfg_dataset)
        
        # 加载权重,当发生错误时执行except
        try:
            self.cfg_weight = self.root_path + "weights/" + self.cfg_hyp["weight"]
        except Exception as e:
            self.cfg_weight = None
            print("权重路径存在问题！！！")
        
        # 加载训练集和验证集路径
        self.dataset_train = self.cfg_dataset["path"]+"/"+self.cfg_dataset["train"]
        self.dataset_val = self.cfg_dataset["path"]+"/"+self.cfg_dataset["val"]        
        self.names = self.cfg_dataset["names"]
        self.nc = len(self.names)

        # 常用超参数定义
        self.weight = self.cfg_hyp["weight"]
        self.imgsz = self.cfg_hyp["imgsz"]
        self.device = self.cfg_hyp["device"]
        self.epochs = self.cfg_hyp["epochs"]
        self.batch = self.cfg_hyp["batch"]
        self.ch = self.cfg_hyp["ch"]
        self.lrf = self.cfg_hyp["lrf"]

    def load_yaml(self, path):
        if isinstance(path, str):    
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.load(f.read(), Loader=yaml.FullLoader)
        return data
    

# main函数
if __name__ == '__main__':
    cfg = Cfg("test.yaml")

