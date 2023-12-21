# ColossalAI

## Training

Implement code list during Creative Integrated Design:

This is part of learning.

- [train_sft.py](train_sft.py)
- [train_sft_deepspeed.py](train_sft_deepspeed.py)
- [train_rm.py](train_rm.py)
- [train_ppo.py](train_ppo.py)
- [data_utils.py](utils/data_utils.py)
- [model_utils.py](model/model_utils.py)
- [ds_config_zero3.json](ds_config_zero3.json)
- [train_rubis.sh](train_rubis.sh)

This is part of inference

- [chatbot.py](inference/chatbot.py)

## Citations

```bibtex
@article{yao2023dschat,
  title={{DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales}},
  author={Zhewei Yao and Reza Yazdani Aminabadi and Olatunji Ruwase and Samyam Rajbhandari and Xiaoxia Wu and Ammar Ahmad Awan and Jeff Rasley and Minjia Zhang and Conglong Li and Connor Holmes and Zhongzhu Zhou and Michael Wyatt and Molly Smith and Lev Kurilenko and Heyang Qin and Masahiro Tanaka and Shuai Che and Shuaiwen Leon Song and Yuxiong He},
  journal={arXiv preprint arXiv:2308.01320},
  year={2023}
}
```
