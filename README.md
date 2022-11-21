# karyotype_detectron2

## image match
### method
- 手工标注
    - 缺点（mask）：边界描绘能力差
    - 缺点（类别）：标注得对照着核型分析图，出错率高
- cut and paste
    - 与真实图像差距较大
#### 引入图像匹配全自动算法 对比各个图像匹配算法
- sift+ransac
    - 染色体类间差距小，类内差距大
    - 比较敏感，效果差
- orb+ransac+尺度约束
    - **实际需要解决的问题，分开黏在一起的染色体！**
    - 尝试图像反转
### metrics
- match rate/acc
- mDSC

## model train
### keys
- keypoints detection / 关键点数量约束
    - 帮助识别小物体
- anchor -- [kmeans聚类](https://zhuanlan.zhihu.com/p/109968578)
    - yolo？
- SERoIAlign
- class_weight
    - implement
- train
    - keep image ratio 特别是在obb中
### metrics
- mDSC
- compare with some SOTA