# SPDNet Models Performance on FPHA
In this repository, I provide my attempt to reproduce [Wang et al. (2023)](https://doi.org/10.1016/j.neunet.2022.11.030)'s result for both SPDNet and U-SPDNet on the FPHA dataset by [Garcia-Hernando et al. (2018)](https://guiggh.github.io/publications/first-person-hands/). It is expected that the result will not be the same as the one in the paper given that there are significant differences in implementation, including:
- Programming language and frameworks, the original uses MATLAB and Manopt while my implementation uses Python, PyTorch and Geoopt.
- Checkpointing objective is not given to provide how they select their results in the paper. The results shown will be based on the validation accuracy.
- The backprop used for the matrix ops (e.g., `expm`, `logm`, etc) in my code is based on the Daleckii-Krein
  theorem defined by [Brooks et al. (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Paper.pdf)
  and [Engin et al. (2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Melih_Engin_DeepKSPD_Learning_Kernel-matrix-based_ECCV_2018_paper.pdf), while [Wang et al. (2023)](https://doi.org/10.1016/j.neunet.2022.11.030) uses the same one used by [Huang et al. (2017)](https://arxiv.org/abs/1608.04233) from [Ionescu et al. (2015)](https://openaccess.thecvf.com/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf).
- I used Riemannian Adam instead of SGD to ensure quick convergence.

For U-SPDNet, the results are quite far from the original paper's result, where I obtained:
|    Model   | Accuracy |
| :--------- | :------: |
|   SPDNet   |  0.8557  |
|  SPDNetBN  |  **0.8713**  |
|  U-SPDNet  |  0.7791  |
| U-SPDNetBN |  0.8122  |

One of the odd things I found for my reconstruction loss is the large values that can go over `10^9`, which potentially could be the reason why
U-SPDNet may perform worse in my reproduction, while the training curve for U-SPDNet in the original paper has the highest loss of less than 10.
I'm not entirely clear why the magnitude of the reconstruction loss is so high. If the reconstruction error is similar to the original,
it should be able to improve U-SPDNet's performance by a lot. Although when the covariance matrices are inspected, the magnitude is surprisingly
large.

The code for SPDNets can be found in my other [repo](https://github.com/zaRizk7/spd-net).

# Installation
By default, you can easily use this code as a package using `pip` by calling:
```bash
pip install git+https://github.com/zaRizk7/fpha-spdnet
```
However, I recommended cloning this repository, setting the repo as a working directory, then calling:
```bash
pip install -e .
```
In case there are bugs in the code, and you want to change it quickly. Some shell scripts can help you get started
on preprocessing the FPHA dataset and running the model.

# Data Preparation

To prepare the data, you want to request [access](https://goo.gl/forms/FIsXpYVIUov0j7Wv2) to the FPHA dataset.
After you have gained access and opened the shared drive, you want to download the `Hand_pose_annotation_v1.zip`
and `data_split_action_recognition.txt`. Afterwards, unzip the `Hand_pose_annotation_v1.zip` file and you can run a script
to extract the covariance/correlation, depending on your preference
```bash
python -m fpha_spdnet.data ${DATA_DIR} \
  --split_file ${SPLIT_FILE} \
  --standardize ${true|false} \
  --estimator ${emp|lw}
```
The arguments' explanation are as follows:
- `${DATA_DIR}`: The directory for the hand pose coordinate, you may assume it is the `Hand_pose_annotation_v1` folder by default.
- `--split_file ${SPLIT_FILE}`: The file to define the split used in FPHA, it is the `data_split_action_recognition.txt`.
- `--standardize ${true|false}`: Whether you want to standardize the coordinate time series, if standardized, then it will produce correlation instead of covariance.
- `--estimator ${emp|lw}`: The covariance estimator you want to apply, by default, I set it to empirical (`emp`) covariance, where if we to retain SPDness, we add `10^-3 * trace(M)` to the diagonal. Alternatively, you may use Ledoit-Wolf shrinkage (`lw`), which may have performance differences when training the SPDNets.

# Training

I will not go into details regarding training; the important thing is that for SPDNets, you need to use the Riemannian optimizers implemented in `geoopt`, as the parametrizations in my SPDNet code used it to enforce constraints on the parameters (i.e., Stiefel or SPD). If you did not use `geoopt` optimizers, you will essentially train just a normal Euclidean neural network, and it may make the hidden SPD outputs ill-conditioned. You may refer to the `configs` folder for setting up a configuration file to run this code without modifying the code itself.

To run the preprocessed data for training, you can run:
```bash
python -m fpha_spdnet fit -c "${CONFIG}.yml" --seed_everything=${SEED}
```
where the `-c ${CONFIG}` is just the path to the configuration files and `--seed_everything=${SEED}` is the random seed to ensure reproducibility of the result.
