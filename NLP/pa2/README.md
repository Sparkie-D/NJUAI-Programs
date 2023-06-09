南京大学人工智能学院自然语言处理课程第二次编程作业：文本极性分类

### 运行方法

在解压后的文件夹中添加train.txt和test.txt后，直接运行roberta.py文件即可获得输出结果。

注意：首次运行可能需要下载RoBERTa模型，请等待下载完毕后即可开始运行。训练的批大小可以自行设置，取尽可能大但内存分配足够即可
### 复现实验结果
在运行部分注释下述代码的第一行，并取消第二行的注释，即
```python
    # classifier = torch.load('roberta_model40.pth').to(device) # roberta85, 0.854  #roberta40 0.8589
    train(classifier, texts, target_words, train_label) # 再次训练
```
### 直接获得实验结果
从提供的roberta_model40.pth中获得模型，并不经过训练直接预测输出结果，得到的.txt文件即为复现出的实验结果。