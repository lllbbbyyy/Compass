**调度节点的设置**

需要具备的信息：

1. layermapper
2. 代价
3. 层
4. 核心的坐标
5. 是否写回dram
6. 该层所在的段的层列表

**对于network.add的改动**

对于输入特征图：
total_ifm为固有的输入特征图大小
real_ifm(padded_ifm)为用于计算ubuf大小的输入特征图
原本的计算逻辑是：**首先，padded_ifm为空**，当第一次被更新时，设置为被更新对象的形状。后续更新时，保持h和w不变，只增加c的大小。
更新的顺序是：先叠加所有的输入来源（更新external_C），接着叠加所有的前驱层的输出。

对于prevWgts不为空时，从0开始累加prevWgts，然后和本身的wgt进行比较。

修改后的逻辑：
padded_ifm首先初始化为（c=0,h,w同步为原来特征图的）,

TODO: 增加对于生成的kvcache的写回功能

**对于model engine的设计**

构造函数：成批量的模型，layer engine列表，段划分编码，长度为layers-1，层到芯片的映射编码