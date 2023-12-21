# ColossalAI

## Training

Implement code list during Creative Integrated Design:

This is part of learning without quantization.

### Without quantization

- [gptNeoX](coati/models/gptNeoX)
- [polyglot-ko](coati/models/polyglotko)
- [train_sft.py](learning/train_sft.py)
- [sft_dataset.py](coati/dataset/sft_dataset.py)
- [train_reward_model.py](learning/train_reward_model.py)
- [train_prompts.py](learning/train_prompts.py)
- [inference.py](learning/inference.py)

### With quantization

This part is additional change for support learning with quantization model.
This is not official feature in Colossal, implemented personally.
Learning may be instability.

Those are codes different without quantization.

- [train_sft.py](learning/train_sft.py)
- [train_prompts.py](learning/train_prompts.py)
- [polyglotko](coati/models/polyglotko)
- [ddp.py](coati/trainer/strategies/ddp.py)

Thos are codes in library.

- colossalai/zero/low_level/low_level_optim.py
- peft/tuners/lora.py

You can find more details in [8bit_learning](https://github.com/retato-snu/FOCUSPANG_LLM/blob/colossal_load8bit/Colossal/learning/8bit_learning.md)

## Authors

Coati is developed by ColossalAI Team:

- [Fazzie](https://fazzie-key.cool/about/index.html)
- [FrankLeeeee](https://github.com/FrankLeeeee)
- [BlueRum](https://github.com/ht-zhou)
- [ver217](https://github.com/ver217)
- [ofey404](https://github.com/ofey404)
- [Wenhao Chen](https://github.com/CWHer)

The PhD student from [(HPC-AI) Lab](https://ai.comp.nus.edu.sg/) also contributed a lot to this project.

- [Zangwei Zheng](https://github.com/zhengzangw)
- [Xue Fuzhao](https://github.com/XueFuzhao)

Modified coati

- [Oh Gyuhyeok](htpps://github.com/retato-snu)

## Citations

```bibtex
@article{Hu2021LoRALA,
    title   = {LoRA: Low-Rank Adaptation of Large Language Models},
    author  = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.09685}
}

@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={arXiv preprint arXiv:2203.02155},
  year={2022}
}

@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}

@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}

@misc{instructionwild,
  author = {Fuzhao Xue and Zangwei Zheng and Yang You },
  title = {Instruction in the Wild: A User-based Instruction Dataset},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/InstructionWild}},
}
```
