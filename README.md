# Adaptive Task Sampling for Meta-Learning

Implementation for "Adaptive Task Sampling for Meta-Learning" on ECCV 2020.

In the original code, the adaptive sampling part is integrated with various open-source projects too closely and deeply, which makes it complicated for reuse. Therefore, the common adaptive sampling code could be and has been reorganized in this project, which can be integrated with different kinds of meta-learning methods.

The baseline meta model implementations are borrowed from open-source repositories, as follows:

* Matching Network and PN: [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot/);
* MAML and MAML++: [HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch);
* Reptile: [Reptile-Pytorch](https://github.com/dragen1860/Reptile-Pytorch).

And the integration steps are as follows:

1. Declare and initialize the `global_vars`, which stores some global variables that are easy to check and update (a very naive implementation is used here, but you can always try a more elegant or efficient way).
2. Sample classes adaptively by the `adaptive_sampling()` method.
3. Update sampling weights by the `update_weights()` method after each training batch.

The key parameters are as follows:

* `adaptive_sampling_mode`: It indicates the adaptive sampling method, including uniform sampling (0), class-based sampling (1-100), class-pair-based sampling (101-200), etc.
* `utility_function_mode`: It controls what classes we preferentially want to select in the adaptive process. For example, we can prioritize difficult pairs (101), simple pairs (102), or uncertain pairs (103).
* `tau`: The weight discounting parameter, which controls the speed of "forgetting".
* `alpha`: The scaling factor for the impact of the current prediction, which controls the speed of "updating".

For these settings, we have done a lot of experiments, so we have left some "unused" mode numbers, e.g., we have tried to accumulate multiple batches before updating the weights, but it didn't help a lot. It is encouraged to try some of the readers' own.

## Citation

```plain
@inproceedings{ECCV2020LiuAdaptive,
  title={Adaptive Task Sampling for Meta-Learning},
  author={Liu, Chenghao and Wang, Zhihao and Sahoo, Doyen and Fang, Yuan and Zhang, Kun and Hoi, Steven C.H.},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
