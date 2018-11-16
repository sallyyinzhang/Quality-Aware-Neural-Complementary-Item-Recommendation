# Quality-Aware Neural Complementary Item Recommendation
This is our TensorFlow implementation for the paper；

@inproceedings{zhang2018quality,
title={Quality-aware neural complementary item recommendation},
author={Zhang, Yin and Lu, Haokai and Niu, Wei and Caverlee, James},
booktitle={Proceedings of the 12th ACM Conference on Recommender Systems},
pages={77--85},
year={2018},
organization={ACM}
}


## Model Traing
Dataset is from 
McAuley, Julian, et al. "Image-based recommendations on styles and substitutes." Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 2015.

First get the product image, textual and rating information and store them in .pickle. Format please see the code.

A quick way to use the model is:
``` 
python CompleRec.py
```
