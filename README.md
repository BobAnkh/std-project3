# std-project3

> 参考文献见[此处](references/swoosh.pdf)

## 对声音数据的处理

根据原文中的描述，对原始声音数据进行了如下处理（我认为会显著减小数据集大小）

- This is first done by subsampling each audio channel from 44.1KHz to 11KHz. 
- Then, a Shorttime Fourier transform (STFT) [12] with a FFT window size of 510 and hop length of 128 is applied on the subsampled
and clipped audio data.
- We further apply a log transformation and clip the representation to between [−5, 5].

## 权重参数

请创建weights文件夹，并从[该链接](https://cloud.tsinghua.edu.cn/d/45313d2093f140acb53f/)下载权重文件放置于此文件夹内运行相关脚本。
