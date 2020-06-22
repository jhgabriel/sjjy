import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


all_data = pd.read_csv("E:\PythonProjects\sjjy\marked.csv", header=None)

all_data.columns = ['id', 'name', 'gender', 'age', 'city',
                    'height', 'xueli', 'details', 'married', 'unknown', 'like', 'photo', 'life']


# age part
age_count = all_data['age'].value_counts()
plt.figure()
index = np.arange(len(age_count))
plt.bar(index, age_count.values)
plt.xlabel("age")
plt.ylabel('count')
plt.xticks(index, np.array(age_count.index))
plt.title("this is a title")
plt.show()


# city part
# 显示中文
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False
city_count = all_data['city'].value_counts()
city_count = city_count.iloc[0:10] # # city 太多了，选择一部分
index = np.arange(len(city_count))
plt.bar(index, city_count.values)
plt.xlabel("city")
plt.ylabel('count')
plt.xticks(index, np.array(city_count.index))
plt.title("this is a title")
plt.show()

# 身高部分
height_fz = pd.cut(all_data['height'], bins=10) # 将身高分成10组
height_fz_count = height_fz.value_counts() # 计算每一组的出现的频率

plt.figure()
index = np.arange(len(height_fz_count))
plt.bar(index, height_fz_count.values)
plt.xlabel("height")
plt.ylabel('count')
plt.xticks(index, np.array(height_fz_count.index))
plt.title("this is a title")
plt.show()


# 需要安装jieba： pip install jieba
# 需要安装wordcloud： pip install wordcloud




import jieba
import re

# all_list = [list(jieba.cut(re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ",
#                                   line.strip()))) for line in all_data.like]
# all_list_f = list()
# for i in all_list:
#     for j in i:
#         all_list_f.append(j)
#
# from collections import Counter
# result_count = Counter(all_list_f)
#
# result_df = pd.DataFrame(result_count.items(), columns=['key', 'value'])
#
# result_df.sort_values('value', inplace=True, ascending=False)



from wordcloud import WordCloud
font = r'C:\Windows\Fonts\STFANGSO.TTF'#电脑自带的字体
wordcloud2 = WordCloud(font_path=font,
                       width=600,
                       height=600).generate(' '.join(all_data['like']))
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad = 0)