# AI-based-synthesizers  

## GAN (Generative Adversarial Network)
- [CTGAN (Xu, Lei, et al., NIPS 2019)](https://arxiv.org/pdf/1907.00503)
  ![image](https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/24a6e327-153f-4636-bb4d-01d2263d0f5e)
  -  TVAE 성능이 더 좋지만, CTGAN이 TVAE보다 더 쉽게 differential privacy를 달성  

- [CTABGAN+]


## VAE (Variational Auto-Encoder)-based method: 
  - GOGGLE (Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, and Mihaela van der Schaar. **Goggle**: Generative modelling for tabular data by learning relational structure. In The Eleventh International Conference on Learning Representations, 2023b.)

## Diffusion-based methods: 


  - [TabDDPM (Kotelnikov et al., 2023 ICML)](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf): Diffusion-based models [github](https://github.com/yandex-research/tab-ddpm)
      - RTX 2080 Ti GPU and Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz.
   
  - [TabMT (Manbir and Roysdon, NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2023/file/90debc7cedb5cac83145fc8d18378dc5-Paper-Conference.pdf): Adopt the masked **transformer**, similar to what is used in BERT, to sequentially impute masked entries
      - 성능은 높지만 속도 느림, All experiments were conducted using cloud A10 or V100 GPUs. For algorithm design and experiment result generation roughly 410 GPU days of compute were used. 
      - Searching temperatures also adds time if optimal privacy is needed. Additionally, we must quantize continuous fields, while we outperform methods which do not quantize fields, this could pose issues in some applications. Future work might examine learning across tabular datasets, alternative masking procedures and networks to improve speed, or integration with diffusion models to better tackle continuous fields.
    ![image](https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/219f7846-819e-4c0d-bffb-6d126ea328cf)

  - [STaSy (Kim et al., ICLR 2023)](https://openreview.net/pdf?id=1mNssCWt_v): Use  Score-based generative models
      - UBUNTU 18.04 LTS, PYTHON 3.8.2, PYTORCH 1.8.1, CUDA 11.4, and NVIDIA Driver 470.42.01, i9 CPU, and NVIDIA RTX 3090. 
     ![image](https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/a8729679-2da5-4864-b859-193fccc977ae)
    
  - CoDi (Lee et al., ICML 2023): Uses two separate diffusion models for cts and discrete variables 
- LLM model based 
  - GReaT (Borisov et al., 2023): Use Transformer-based Large Language Models (LLMs) to create synthetic data 
  - ![image-20240519194223115](AI-based synthesizers.assets/image-20240519194223115.png)
  - ![image-20240519194257628](AI-based synthesizers.assets/image-20240519194257628.png)
-  TABSYN (Zhang et al, ICLR 2024): Diffusion model within the latent variable space of VAE
- Forest-VP (Score-based diffusion models, trained with gradient-boosted tree) Forest-Flow (Flow-based model, trained with gradient-boosted tree) Jolicoeur-Martineau et al ICML 2024. 


Noseong Park, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park, and Youngmin Kim. Data synthesis based on generative adversarial networks. arXiv preprint arXiv:1806.03384, 2018.

Chaejeong Lee, Jayoung Kim, and Noseong Park. **Codi**: Co-evolving contrastive diffusion models for mixed-type tabular synthesis. In International Conference on Machine Learning, pp. 18940–18956. PMLR, 2023.

Jayoung Kim, Chaejeong Lee, and Noseong Park. **Stasy**: Score-based tabular data synthesis. In The Eleventh International Conference on Learning Representations, 2023.

Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. **Tabddpm**: Modelling tabular data with diffusion models. In International Conference on Machine Learning, pp. 17564–PMLR, 2023.

Gulati, Manbir, and Paul Roysdon. "**TabMT**: Generating tabular data with masked transformers." *Advances in Neural Information Processing Systems* 36 (2024).

Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2022). Language models are realistic tabular data generators. ICLR 2023. *arXiv preprint arXiv:2210.06280*. 

Jolicoeur-Martineau, Alexia, Kilian Fatras, and Tal Kachman. "Generating and Imputing Tabular Data via Diffusion and Flow-based **Gradient-Boosted Trees**." *International Conference on Artificial Intelligence and Statistics*. PMLR, 2024. 

Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023, July). **Tabddpm**: Modelling tabular data with diffusion models. In *International Conference on Machine Learning* (pp. 17564-17579). PMLR.

Zhao, Z., Kunar, A., Birke, R., & Chen, L. Y. (2021, November). Ctab-gan: Effective table data synthesizing. In *Asian Conference on Machine Learning* (pp. 97-112). PMLR.

Zhao, Z., Kunar, A., Birke, R., Van der Scheer, H., & Chen, L. Y. (2024). Ctab-gan+: Enhancing tabular data synthesis. *Frontiers in big Data*, *6*, 1296508.

This is partially because generative modelling of tabular data entails a particular set of challenges, including heterogeneous relationships, limited number of samples, and difficulties in incorporating prior knowledge. 



