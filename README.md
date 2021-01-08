# std-project3 

视听信息系统导论第三次大作业

具体可见报告[report](report.pdf)

> 参考文献见[此处](references/swoosh.pdf)

## 对声音数据的处理

根据原文中的描述，对原始声音数据进行了如下处理（如此可以显著减小数据集大小）

- This is first done by subsampling each audio channel from 44.1KHz to 11KHz. 
- Then, a Shorttime Fourier transform (STFT) [12] with a FFT window size of 510 and hop length of 128 is applied on the subsampled
and clipped audio data.
- We further apply a log transformation and clip the representation to between [−5, 5].

## 权重参数

请创建weights文件夹，并从[该链接](https://cloud.tsinghua.edu.cn/d/45313d2093f140acb53f/)下载权重文件放置于此文件夹内运行相关脚本。

## 维护者

[@BobAnkh](https://github.com/BobAnkh)

[@zxdclyz](https://github.com/zxdclyz)

[@duskmoon314](https://github.com/duskmoon314)

## 关联项目

[视听导第一次大作业](https://github.com/zxdclyz/std-project1)

[视听导第二次大作业](https://github.com/duskmoon314/std-project2)

[视听导第三次大作业](https://github.com/BobAnkh/std-project3)
