# AI-based-synthesizers  

## GAN (Generative Adversarial Network)
- [Table-GAN](https://arxiv.org/pdf/1806.03384)
  ```
  Noseong Park, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park, and Youngmin Kim. Data synthesis based on generative adversarial networks. arXiv preprint arXiv:1806.03384, 2018.
  ```


- [CTGAN (Xu, Lei, et al., NIPS 2019)](https://arxiv.org/pdf/1907.00503)
    -  TVAE 성능이 더 좋지만, CTGAN이 TVAE보다 더 쉽게 differential privacy를 달성  
       <img src="https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/24a6e327-153f-4636-bb4d-01d2263d0f5e" alt="image" width="500"/>



- [CTABGAN+]
  ```
  Zhao, Z., Kunar, A., Birke, R., & Chen, L. Y. (2021, November). Ctab-gan: Effective table data synthesizing. In *Asian Conference on Machine Learning* (pp. 97-112). PMLR.
  ```
  ```
  Zhao, Z., Kunar, A., Birke, R., Van der Scheer, H., & Chen, L. Y. (2024). Ctab-gan+: Enhancing tabular data synthesis. *Frontiers in big Data*, *6*, 1296508.
  ```



## VAE (Variational Auto-Encoder)-based method: 
  - [GOGGLE (Liu et al, ICLR 2023)](https://openreview.net/pdf?id=fPVRcJqspu)
    ```
    Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, and Mihaela van der Schaar. **Goggle**: Generative modelling for tabular data by learning relational structure. In The Eleventh International Conference on Learning Representations, 2023b.
    ```

  - 
-  [TABSYN (Zhang et al, ICLR 2024)](https://arxiv.org/pdf/2310.09656):[[github]](https://github.com/amazon-science/tabsyn) Diffusion model within the latent variable space of VAE  
   (1) 일반성: 다양한 데이터 타입을 단일 통합 공간으로 변환하고 열 간 관계를 명시적으로 포착할 수 있는 능력  
   (2) 품질: 잠재 임베딩의 분포를 최적화하여 디퓨전 모델의 후속 훈련을 향상시키고, 이를 통해 고품질의 합성 데이터를 생성할 수 있음  
   (3) 속도: 기존 디퓨전 기반 방법보다 훨씬 적은 역방향 단계와 빠른 합성 속도.
   
  ```
  Mixed-type tabular data synthesis with score-based diffusion in latent space
  ```
   


## Diffusion-based methods: 


  - [TabDDPM (Kotelnikov et al., 2023 ICML)](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf): [[github]](https://github.com/yandex-research/tab-ddpm) Diffusion-based models
      - RTX 2080 Ti GPU and Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz.
        ```
        Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. **Tabddpm**: Modelling tabular data with diffusion models. In International Conference on Machine Learning, pp. 17564–17579, PMLR, 2023.
        ```

   
  - [TabMT (Manbir and Roysdon, NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2023/file/90debc7cedb5cac83145fc8d18378dc5-Paper-Conference.pdf): Adopt the masked **transformer**, similar to what is used in BERT, to sequentially impute masked entries
      - 성능은 높지만 속도 느림, All experiments were conducted using cloud A10 or V100 GPUs. For algorithm design and experiment result generation roughly 410 GPU days of compute were used. 
      - Searching temperatures also adds time if optimal privacy is needed. Additionally, we must quantize continuous fields, while we outperform methods which do not quantize fields, this could pose issues in some applications. Future work might examine learning across tabular datasets, alternative masking procedures and networks to improve speed, or integration with diffusion models to better tackle continuous fields.
       <img src="https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/219f7846-819e-4c0d-bffb-6d126ea328cf" alt="image" width="500"/>

      - This quality is verified at scales that are orders of magnitude larger than prior work and with missing data present. Our model achieves superior privacy and is able to easily trade off between privacy and quality. Our model is a substantial advancement compared to previous work, due to its scalability, missing data robustness, privacypreserving generation, and superior data quality.
        ```
        Gulati, Manbir, and Paul Roysdon. "**TabMT**: Generating tabular data with masked transformers." *Advances in Neural Information Processing Systems* 36 (2024).
        ```


  - [STaSy (Kim et al., ICLR 2023)](https://openreview.net/pdf?id=1mNssCWt_v): Use  Score-based generative models
      - UBUNTU 18.04 LTS, PYTHON 3.8.2, PYTORCH 1.8.1, CUDA 11.4, and NVIDIA Driver 470.42.01, i9 CPU, and NVIDIA RTX 3090.
       <img src="https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/a8729679-2da5-4864-b859-193fccc977ae" alt="image" width="400"/>

        ```
        Jayoung Kim, Chaejeong Lee, and Noseong Park. **Stasy**: Score-based tabular data synthesis. In The Eleventh International Conference on Learning Representations, 2023.
        ```
 

  - [Forest-VP & Forest-Flow (Martineau et al AISTATS 2024)](https://proceedings.mlr.press/v238/jolicoeur-martineau24a/jolicoeur-martineau24a.pdf): [[github](https://github.com/SamsungSAILMontreal/ForestDiffusion)]
    - Forest-VP (Score-based diffusion models, trained with gradient-boosted tree) Forest-Flow (Flow-based model, trained with gradient-boosted tree) 
    - gpu 필요없음
    - We trained the tree-based models on a cluster of 10-20 CPUs with 64-256Gb of RAM.
    - We trained the other models on a cluster with 8 CPUs, 1 GPU, and 48-128Gb of RAM

      ```
      Jolicoeur-Martineau, Alexia, Kilian Fatras, and Tal Kachman. "Generating and Imputing Tabular Data via Diffusion and Flow-based **Gradient-Boosted Trees**." *International Conference on Artificial Intelligence and Statistics*. PMLR, 2024.
      ```
      ![image](https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/d320195f-d751-4c13-931d-f8782f9c43b7)

    
  - [CoDi (Lee et al., ICML 2023)](https://proceedings.mlr.press/v202/lee23i/lee23i.pdf): [[github]](https://github.com/ChaejeongLee/CoDi) Uses two separate diffusion models for continuous and discrete variables
     - UBUNTU 18.04.6 LTS, PYTHON 3.10.8, PYTORCH 1.11.0, CUDA 11.7, and NVIDIA Driver 470.161.03, i9, CPU, and NVIDIA RTX 3090.     
     - STaSy보다 속도 빠름
       ```
       Chaejeong Lee, Jayoung Kim, and Noseong Park. **Codi**: Co-evolving contrastive diffusion models for mixed-type tabular synthesis. In International Conference on Machine Learning, pp. 18940–18956. PMLR, 2023.
       
       ```


    
## LLM model based 
- [GReaT (Borisov et al., ICLR 2023)](https://arxiv.org/pdf/2210.06280): [[github]](https://github.com/kathrinse/be_great) Use Transformer-based Large Language Models (LLMs) to create synthetic data
    ```
    Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2022). Language models are realistic tabular data generators. ICLR 2023. *arXiv preprint arXiv:2210.06280*.
    ```
    <img src="https://github.com/soyul5458/AI-based-synthesizers/assets/54921677/fc59b51f-7e4d-49c1-8c92-0ec0f98c1db1" alt="image" width="500"/>







- [CuTS: Customizable Tabular Synthetic Data Generation, ICML 2024, POSTER](https://icml.cc/virtual/2024/poster/33789)




