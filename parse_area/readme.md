### 地区的可视化

1. 清洗脏数据，根据中国行政区域映射表，将原数据中的不规范的地名都转化成省级区域，
清洗后的数据形成 `世纪佳缘_去重_UserInfo_女_area.csv` 和 
`世纪佳缘_去重_UserInfo_男_area.csv` 两个文件

2. 对新文件按照区域（work_location）分组和排序，排序后的数据存入 
`女_area_group.xls` 和 `男_area_group.xls` 

3. 排序后的数据做进一步清理（去除总数量在50以下的，以及其他国家的），然后拿到
数可视实现数据的可视化