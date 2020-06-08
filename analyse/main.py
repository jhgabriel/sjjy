#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 对爬虫抓取的佳缘数据进行统计和图表输出
 
# %%
# base code
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

#为图表设置中文字体
# make matplot support chinese 
import platform
print(platform.system())
if(platform.system()=='Windows'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif(platform.system()=='Linux'):
    pass
elif(platform.system()=='Darwin'):
    plt.rcParams["font.family"] = 'Arial Unicode MS'
else:
    pass

# 加载数据
def load_data():
    userData = pd.read_csv('../世纪佳缘_去重_UserInfo.csv',
                           names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education',
                                  'matchCondition', 'marriage', 'income', 'shortnote', 'image'])
    return userData


# %%
df = load_data()
print(df.dtypes)

# %%
#按性别统计
sub = df['sex'].value_counts(ascending=True)
print(sub)
sub.plot.bar()

# %%
#按年龄统计
sub = df['age'].value_counts(ascending=True)
print(sub)
sub.plot.bar()

# %%
#按工作地统计
sub = df['work_location'].value_counts(ascending=True)
print(sub)
sub.plot.barh()

# %%
print('按身高统计')
sub = df['height'].value_counts(ascending=True)
print(sub)
sub.plot.barh()

# %%
print('按 受教育程度 统计')
sub = df['education'].value_counts(ascending=True)
print(sub)
sub.plot.barh()

# %%
print('按 婚姻状况 统计')
sub = df['marriage'].value_counts(ascending=True)
print(sub)
sub.plot.barh()

# %%
# print('按 收入 统计')
# sub = df['income'].value_counts(ascending=True)
# print(sub)
# sub.plot.barh()

# %%
# print('按 是否有照片 统计')
# sub = df['image'].value_counts(ascending=True)
# print(sub)
# sub.plot.barh()

# %%
