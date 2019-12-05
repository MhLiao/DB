# -*- coding:utf-8 -*-
# @author :adolf
import yaml

with open('../experiments/seg_detector/totaltext_resnet50_deform_thre.yaml') as loadfile:
    p_yaml = yaml.load(loadfile)

print(p_yaml)
