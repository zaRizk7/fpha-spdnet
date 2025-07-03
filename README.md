# SPDNet Models Performance on FPHA

In this repository, I provide my attempt to reproduce [Wang et al. (2023)](https://doi.org/10.1016/j.neunet.2022.11.030)'s result for both SPDNet and U-SPDNet on the FPHA dataset by [Garcia-Hernando et al. (2018)](https://guiggh.github.io/publications/first-person-hands/). It is expected that the result will not be the same as the one in the paper given that there are significant differences in implementation, including:
- Programming language and frameworks, the original uses MATLAB and Manopt while my implementation uses Python, PyTorch and Geoopt.
- Checkpointing objective is not given to provide how they select their results in the paper. The results shown will be based on the validation loss.
- The backprop used for the matrix ops (e.g., `expm`, `logm`, etc) in my code is based on the Daleckii-Krein
  theorem defined by [Brooks et al. (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Paper.pdf)
  and [Engin et al. (2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Melih_Engin_DeepKSPD_Learning_Kernel-matrix-based_ECCV_2018_paper.pdf), while [Wang et al. (2023)](https://doi.org/10.1016/j.neunet.2022.11.030) uses the same one used by [Huang et al. (2017)](https://arxiv.org/abs/1608.04233) from [Ionescu et al. (2015)](https://openaccess.thecvf.com/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf).
- I used Riemannian Adam instead of SGD to ensure quick convergence.

The results are quite far from the original paper's result, where I obtained:
|    Model   | Accuracy |
| :--------- | :------: |
|   SPDNet   |  0.8557  |
|  SPDNetBN  |  **0.8713**  |
|  U-SPDNet  |  0.7791  |
| U-SPDNetBN |  0.8122  |


One of the odd things I found for my reconstruction loss the large values that can go over `10^9`, which potentially be the reason why
U-SPDNet may perform worse in my reproduction, while the training curve for U-SPDNet in the original paper has highest loss less than 10.
I'm not entirely clear why the reconstruction loss's magnitude is that high. I believe if the reconstruction error is similar to the original,
it should be able to improve U-SPDNet's performance by a lot.

# First Person Hand Action (FPHA)

FPHA is a computer vision dataset mainly intended to build a model for hand pose coordinate estimation.
It also provides a classification problem where we want to identify the activity that is being done based
on the hand pose coordinate through time. The dataset contains over 1,175 hand action videos categorized
over 45 different hand actions. FPHA provides a pre-defined splits where the training and test split has 575
and 600 samples respeciively.

For SPDNets, we extract the covariance from the hand pose coordinates which contains 21 hand points in 3D
coordinates, implying we have `21 x 3 = 63` time series that can be converted into `63 x 63`
symmetric positive definite (SPD) matrices used for SPDNet's input. For the task, we will do a multiclass
classification on the 45 different categories mentioned.

# Installation and Preprocessing

TBA