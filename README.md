Scalable and Sound Low-Rank Tensor Learning
=================

This repository includes the softwares for 
[Scalable and Sound Low-Rank Tensor Learning](http://www.jmlr.org/proceedings/papers/v51/cheng16.pdf).
```
@InProceedings{Cheng2016AISTATS,
  author    = {Hao Cheng and Yaoliang Yu and Xinhua Zhang and Eric Xing and Dale Schuurmans},
	title     = {Scalable and Sound Low-Rank Tensor Learning},
	booktitle = {Proc. International Conference on Aritificial Intelligence and Statistics (AISTATS)},
	year      = {2016}
}
```
It is an extended version of 
[Approximate Low-Rank Tensor Learning](http://opt-ml.org/papers/opt2014_submission_7.pdf) 
in NIPS Workshop on Optimization for Machine Learning.

## Requirements
- [PROPACK] (http://sun.stanford.edu/~rmunk/PROPACK/)
- [LBFGSB] Matlab interface (https://github.com/pcarbo/lbfgsb-matlab)
- [Matlab] 2012 or higher

## Usage
To run the package, please first make sure the lbfgsb and PROPACK work properly.
Then, please run the ./mex/make_mex.m to make all required mex.

A example run code for low-rank tensor completion is shown in ./Synthetic_exp.m.

## Question Contact
- Hao Cheng 
- Yaoliang Yu
- Xinhua Zhang
