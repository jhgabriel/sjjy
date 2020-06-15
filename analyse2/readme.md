main.py 实现佳缘用户的画像分析
主要思路为先对一批用户进行手工标注，然后用训练出对模型对新用户进行分类，分类为一般/良好/优秀


# 主要步骤
- jieba库进行分词
- word2vec进行向量化
- 使用keras进行模型设计和训练


# 详细步骤
## 数据清洗

- 筛选要参与nlp对字段
  - [ ] uid，样本：199797256
  - [ ] 'nickname', 梦梦大仙女
  - [ ] 'sex', 女
  - [ ] 'age', 30
  - [ ] 'work_location', 南京
  - [ ] 'height', 155
  - [*] 'education', 本科                          
  - [ ] 'matchCondition', "28-36岁,170-180cm,有照片,江苏,南京"
  - [*] 'marriage', 离异
  - [ ] 'income', --
  - [*] 'shortnote', "看了《亲爱的，热爱的》，好想谈一场甜甜的恋爱，我的现男友在哪里
  - [ ] 'image' --
- 对上面的字段*号字段合并，然后分词
- 分词结果进行word2vec


[参考1](https://www.jianshu.com/p/158c3f02a15b)
[参考2](https://blog.csdn.net/dreamtheworld1/article/details/80634611)