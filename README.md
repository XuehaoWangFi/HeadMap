# (ICLR 2025) HeadMap: Locating and Enhancing Knowledge Circuits in LLMs

[Xuehao Wang](https://openreview.net/profile?id=~Xuehao_Wang3), [Liyuan Wang](https://openreview.net/profile?id=~Liyuan_Wang3), [Binghuai Lin](https://openreview.net/profile?id=~Binghuai_Lin1), [Yu Zhang](https://openreview.net/profile?id=~Yu_Zhang3)

Official Implementation of ICLR 2025 paper "[HeadMap: Locating and Enhancing Knowledge Circuits in LLMs](https://openreview.net/forum?id=jUsrbOuQ5e)".

## Abstract

Large language models (LLMs), through pretraining on extensive corpora, encompass rich semantic knowledge and exhibit the potential for efficient adaptation to diverse downstream tasks. However, the intrinsic mechanisms underlying LLMs remain unexplored, limiting the efficacy of applying these models to downstream tasks. In this paper, we explore the intrinsic mechanisms of LLMs from the perspective of knowledge circuits. Specifically, considering layer dependencies, we propose a layer-conditioned locating algorithm to identify a series of attention heads, which is a knowledge circuit of some tasks. Experiments demonstrate that simply masking a small portion of attention heads in the knowledge circuit can significantly reduce the model's ability to make correct predictions. This suggests that the knowledge flow within the knowledge circuit plays a critical role when the model makes a correct prediction. Inspired by this observation, we propose a novel parameter-efficient fine-tuning method called HeadMap, which maps the activations of these critical heads in the located knowledge circuit to the residual stream by two linear layers, thus enhancing knowledge flow from the knowledge circuit in the residual stream. Extensive experiments conducted on diverse datasets demonstrate the efficiency and efficacy of the proposed method.

## Environment

+ Python 3.10.4
+ torch 2.0.0+cu118
+ numpy 1.22.3
+ transformers 4.35.2

## Citation

If you find MTSAM is useful for your research and applications, please cite using this BibTeX:

```
@inproceedings{
    wang2025headmap,
    title={HeadMap: Locating and Enhancing Knowledge Circuits in {LLM}s},
    author={Xuehao Wang, Liyuan Wang, Binghuai Lin, Yu Zhang},
    booktitle={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=jUsrbOuQ5e}
}
```

