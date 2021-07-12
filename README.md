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