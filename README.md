This repository is used to reproduce memory leak problem in dgl.  
Codes Implementary is based on [GCResNet](https://github.com/Boyan-Lenin-Xu/GCResNet).  
Sub-dataset of GoPro(an image deblurring dataset) is already in the following path:  
`dataset/GoPro/GOPR0372_07_00/`  
To reproduce memory leak problem, Some steps are suggested to be followed:  
`cd src`  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode deblur --save_results`  
My issue can be found in [here](https://github.com/dmlc/dgl/issues/3551).
