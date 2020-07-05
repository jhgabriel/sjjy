# 输入
- 世纪佳缘_去重_UserInfo_男.csv
- 世纪佳缘_去重_UserInfo_女.csv
- 每个文件包含数据5000条左右

# 算法逻辑
## 训练模型
- 根据shortnote的描述手工标注了男性数据500条，女生数据500条，分为“平凡/优秀/卓越"三类。
- 标注后的文件是"世纪佳缘_去重_UserInfo_男_label.csv"和"世纪佳缘_去重_UserInfo_女_label.csv"
- 训练

## 对输入数据打分
- 打分使用了加权求和策略
- score = w_age*age_score + w_eduction* eduction_score + w_height*height_score + w_shortnote*shortnote_score
- 生成打分文件"世纪佳缘_去重_UserInfo_男_score.csv"和"世纪佳缘_去重_UserInfo_女_score.csv"

## 进行恋爱对象推荐
- 输入当前用户的基本信息：性别 年龄 教育程度 身高 自我介绍
- 计算当前用户的分值
- 在数据中遍历查找与当前用户分值接近的用户，找到10个后即返回

## 算法背景
文本分类是NLP的常见任务，本算法采用LSTM模型，训练一个能够识别用户自我介绍 的"平凡/优秀/卓越"的三分类器。
处理过程，包括：
- 不同类别数据整理成输入矩阵
- jieba分词
- Word2Vec词向量模型训练
- LSTM三分类模型
LSTM三分类的参数设置跟二分有区别，选用softmax，并且loss函数也要改成categorical_crossentropy，代码如下：

体会：
总而言之，训练数据的质量是非常重要的，本实验中基于单个标注者思维判断而标注的数据有较为明显的倾向性。



