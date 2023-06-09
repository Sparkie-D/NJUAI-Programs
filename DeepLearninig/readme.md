<center>
<h1>myDeepCluster 运行说明 </h1>
<p1>201300096 人工智能学院 杜兴豪</p1>
</center>

代码框架组织如下：
```python
-myDeepCluster
    -report.pdf              # 实验报告
    -readme.md
    -src                     
        -ScanpyBased.py      
        -AEbased.py          
        -VAEbased.py         
        -MLP.py              
        -model               
        -myDeepCluster.py    # 最终实现的模型
        -NMI_ARI.py    
        -data_process.py     
        -figures             
    -spca_dat                # 存放实验数据集 （已删除，若要使用请重定位path到新数据集）
```
对于最终的模型：
* 若要对单个样例文件进行聚类，则需要运行myDeepCluster.py文件，修改其中path为对应想要聚类的文件名（默认为`sample_151510.h5`）
  * 默认显示训练进程。若要关闭训练进程展示，则需要取消注释文件中`model.show=False`的注释
* 若要获得最终的NMI和ARI的对照图，请直接运行NMI_ARI.py文件
* 若需要生成并更新figures文件夹下的聚类图，则将NMI_ARI.py文件中的全局变量`ifplot`设置为True即可。

如果想观察其他尝试中的模型的聚类过程，则需要运行对应的文件：
* AE: AEbased.py
* VAE: VAEbased.py
* MLP: MLP.py
* Scanpy: ScanpyBased.py

它们都从data_process.py中读取数据集，若要更改默认的数据集，可以：
1. 改变data_process.py中的全局变量path为对应的数据集的路径
2. 在模型文件中传入路径，不适用默认值
