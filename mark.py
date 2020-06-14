import pandas as pd
import csv
import numpy as np

##读取文件
#参数1:文件路径
#参数2:写权限
csv_file = open('E:\PythonProjects\sjjy\世纪佳缘_去重_UserInfo.csv',encoding='utf-8')
csv_reader_lines = csv.reader(csv_file)
#写入文件
#参数1:文件路径
#参数2:写权限
#参数3:编码格式
#参数4:换行符标识（因为读出的文件自带换行符，所以需要设置为空，否则写入时会空一行
writeFile = open('E:\PythonProjects\sjjy\marked.csv','w',encoding='utf-8',newline='')
writer = csv.writer(writeFile)

##按行操作文件
for one_line in csv_reader_lines:
    if int(one_line[3]) < 26:
      one_line.append('优秀')
    elif int(one_line[3]) < 30:
      one_line.append('良好')
    else:
      one_line.append('一般')
    writer.writerow(one_line)

















