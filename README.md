# AI-based-synthesizers  

Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Modeling tabular data using conditional gan. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 7335–7345, 2019



- 
- Classical GAN and VAE 기반 방법: CTGAN and TVAE (Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Modeling tabular data using conditional gan. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 7335–7345, 2019)

- VAE-based method: 
  - GOGGLE (Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, and Mihaela van der Schaar. **Goggle**: Generative modelling for tabular data by learning relational structure. In The Eleventh International Conference on Learning Representations, 2023b.)
- Diffusion-based methods: 
  - STaSy (Kim et al., ICLR 2023): Use  Score-based generative models 
  - TabDDPM (Kotelnikov et al., 2023 ICML): Diffusion-based models
  - CoDi (Lee et al., ICML 2023): Uses two separate diffusion models for cts and discrete variables 
- LLM model based 
  - GReaT (Borisov et al., 2023): Use Transformer-based Large Language Models (LLMs) to create synthetic data 
  - ![image-20240519194223115](AI-based synthesizers.assets/image-20240519194223115.png)
  - ![image-20240519194257628](AI-based synthesizers.assets/image-20240519194257628.png)
-  TABSYN (Zhang et al, ICLR 2024): Diffusion model within the latent variable space of VAE
- Forest-VP (Score-based diffusion models, trained with gradient-boosted tree) Forest-Flow (Flow-based model, trained with gradient-boosted tree) Jolicoeur-Martineau et al ICML 2024. 
- **TabMT**: Adopt the masked **transformer**, similar to what is used in BERT, to sequentially impute masked entries (Manbir and Roysdon, NeurIPS 2024)

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



