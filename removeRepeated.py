import pandas as pd
import glob

##参数1：文件名字
##参数2：不要加行号
frame = pd.read_csv("E:\PythonProjects\sjjy\世纪佳缘_UserInfo.csv",index_col=0);
##去重
data = frame.drop_duplicates()
##把文件写到
data.to_csv("E:\PythonProjects\sjjy\listWithoutDuplicated.csv")



