# HW9 Integral Image & Face Detection
#### An Inplmentation of the paper - [Rapid object detection using a boosted cascade of simple features](https://github.com/SitiBanc/1061_NCTU_IOMDS/blob/master/1122/Course%20Material/viola01rapid.pdf)
### Goal:
1. SC改成取前20個特徵，用這20個特徵組成一個分類器
2. 找一張1920\*1080的照片(最好裡面有很多人臉)來TEST，標出照片裡所有的人臉(座標跟theta都會隨著image的大小變，從19*19開始)
