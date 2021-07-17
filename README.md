rcnn, rcnn-2d, tr_net的实验结果：  
|Segment-level|rcnn|rcnn-2d|tr_net|
|:--:|:--:|:--:|:--:|
|best_epoch|1306|1446|954|
|performance|75.29|73.75|74.00|
|type_acc|78.73|77.14|76.35|
|type_f1|35.38|33.85|31.88|
|stenosis_acc|71.85|70.37|71.64|
|stenosis_f1|53.22|46.66|53.08|

|Branch-level|rcnn|rcnn-2d|tr_net|
|:--:|:--:|:--:|:--:|
|stenosis_acc|52.30|63.96|60.44|
|stenosis_f1|23.14|36.74|39.58|

|Patient-level-level|rcnn|rcnn-2d|tr_net|
|:--:|:--:|:--:|:--:|
|stenosis_acc|57.14|65.31|63.95|
|stenosis_f1|19.28|34.05|37.12|

6.15号会议TODO：  
1. 训练时候的pad方式要改，从中间向两边pad
2. 斑块长度从中间最严重帧crop或pad到二三十帧
3. 做一个五折交叉验证，更能看出是不是数据集的问题

发现一个bug：在utils.Balanced_Sampler中__len__方法错误，应返回self.num_samples. ————fixed

改完之后反而在performance上更差了：
|Segment-level|rcnn|center_pad|
|:--:|:--:|:--:|
|best_epoch|52|10|
|performance|72.11|68.39|
|type_acc|74.92|77.62|
|type_f1|32.37|37.81|
|stenosis_acc|69.31|59.15|
|stenosis_f1|47.38|34.95|

用训练好的网络，对测试集的branch进行推断，发现type和stenosis的结果绝大多数都不为零。甚至在训练集上推断也是这样，loss曲线是下降的，所以很奇怪。  
可能是正常的branch采样太少的原因？打算加大sample_prob。

发现一个bug：utils.merge_plaque逻辑错误(和上面的错误没有太大关系). ————fixed

