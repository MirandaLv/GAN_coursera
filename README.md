# Coursera's specialization of Generative Adversarial Networks (GANs) offered by Deeplearning.AI

Below are some useful python notebooks for different GAN model implementation.

## Course 1. Building Basic GAN
#### Week 1
#### Reading recommendations

- Hyperspherical Variational Auto-Encoders (Davidson, Falorsi, De Cao, Kipf, and Tomczak, 2018): https://www.researchgate.net/figure/Latent-space-visualization-of-the-10-MNIST-digits-in-2-dimensions-of-both-N-VAE-left_fig2_324182043
- Analyzing and Improving the Image Quality of StyleGAN (Karras et al., 2020): https://arxiv.org/abs/1912.04958
- Semantic Image Synthesis with Spatially-Adaptive Normalization (Park, Liu, Wang, and Zhu, 2019): https://arxiv.org/abs/1903.07291
- Few-shot Adversarial Learning of Realistic Neural Talking Head Models (Zakharov, Shysheya, Burkov, and Lempitsky, 2019): https://arxiv.org/abs/1905.08233
- Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (Wu, Zhang, Xue, Freeman, and Tenenbaum, 2017): https://arxiv.org/abs/1610.07584
- These Cats Do Not Exist (Glover and Mott, 2019): http://thesecatsdonotexist.com/
- Large Scale GAN Training for High Fidelity Natural Image Synthesis (Brock, Donahue, and Simonyan, 2019): https://arxiv.org/abs/1809.11096
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html#pytorch-documentation
- MNIST Database: http://yann.lecun.com/exdb/mnist/

#### Week 2
#### Reading recommendations
- **Transposed convolutions :** Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003
- **DCGAN paper :** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016): https://arxiv.org/abs/1511.06434
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016): https://arxiv.org/abs/1511.06434

#### Python notebooks
- In this notebook, you're going to learn about TGAN, from the paper Temporal Generative Adversarial Nets with Singular Value Clipping (Saito, Matsumoto, & Saito, 2017), and its origins in image generation. 
Notebook link: https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W2_Video_Generation_(Optional).ipynb

#### Week 3
#### Reading recommendations
- **Spectral normalization (SNGAN)**, a weight normalization technique to stabilize the training of the discriminator, as proposed in Spectral Normalization for Generative Adversarial Networks (Miyato et al. 2018).
- **Protein GAN**: https://www.biorxiv.org/content/10.1101/789719v2
- **WGAN** Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): https://arxiv.org/abs/1701.07875
- **WGAN-GP** Improved Training of Wasserstein GANs (Gulrajani et al., 2017): https://arxiv.org/abs/1704.00028 Gradient penalty (GP) as well as weight clipping to WGAN in order to enforce 1-Lipschitz continuity and improve stability
- **From GAN to WGAN (Weng, 2017):** https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html This article provides a great walkthrough of how WGAN addresses the difficulties of training a traditional GAN with a focus on the loss functions.
#### Python notebooks
- **ProteinGAN:** https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/ProteinGAN.ipynb

#### Week 4
#### Reading recommendations
- **Conditional GAN:** Conditional Generative Adversarial Nets (Mirza and Osindero, 2014): https://arxiv.org/abs/1411.1784
- An example of Controllable GAN: Interpreting the Latent Space of GANs for Semantic Face Editing (Shen, Gu, Tang, and Zhou, 2020): https://arxiv.org/abs/1907.10786


## Course 2. Building Better GAN
#### Week 1
#### Reading recommendations
- A Note on the Inception Score (Barratt and Sharma, 2018): https://arxiv.org/abs/1801.01973 Know more about why Fréchet Inception Distance has overtaken Inception Score, this paper illustrates the problems with using Inception Score.
- HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models (Zhou et al., 2019): https://arxiv.org/abs/1904.01121 Human evaluation and HYPE (Human eYe Perceptual Evaluation) of GANs.
- Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi, Karras, Laine, Lehtinen, and Aila, 2019): https://arxiv.org/abs/1904.06991
- Fréchet Inception Distance (Jean, 2018): https://nealjean.com/ml/frechet-inception-distance/
- GAN — How to measure GAN performance? (Hui, 2018): https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732

#### Week 2
#### Reading recommendations
- Machine Bias (Angwin, Larson, Mattu, and Kirchner, 2016): https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- Fairness Definitions Explained (Verma and Rubin, 2018): https://fairware.cs.umass.edu/papers/Verma.pdf
- Machine Learning Glossary: Fairness (2020): https://developers.google.com/machine-learning/glossary/fairness
- A Survey on Bias and Fairness in Machine Learning (Mehrabi, Morstatter, Saxena, Lerman, and Galstyan, 2019): https://arxiv.org/abs/1908.09635
- Does Object Recognition Work for Everyone? (DeVries, Misra, Wang, and van der Maaten, 2019): https://arxiv.org/abs/1906.02659
- What a machine learning tool that turns Obama white can (and can't) tell us about AI bias (Vincent, 2020): https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias
- Fair Attribute Classification through Latent Space De-biasing. Vikram V. Ramaswamy, Sunnie S. Y. Kim, Olga Russakovsky. CVPR 2021.
- Hyperspherical Variational Auto-Encoders (Davidson, Falorsi, De Cao, Kipf, and Tomczak, 2018): https://arxiv.org/abs/1804.00891
- Generating Diverse High-Fidelity Images with VQ-VAE-2 (Razavi, van den Oord, and Vinyals, 2019): https://arxiv.org/abs/1906.00446
- Conditional Image Generation with PixelCNN Decoders (van den Oord et al., 2016): https://arxiv.org/abs/1606.05328
- Glow: Better Reversible Generative Models (Dhariwal and Kingma, 2018): https://openai.com/blog/glow/
- PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models (Menon, Damian, Hu, Ravi, and Rudin, 2020): https://arxiv.org/abs/2003.03808

#### Python notebooks
- **VAE**
- **Score-based GAN model:** https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_(Optional_Notebook)_Score_Based_Generative_Modeling.ipynb This is a hitchhiker's guide to score-based generative models, a family of approaches based on estimating gradients of the data distribution. They have obtained high-quality samples comparable to GANs (like below, figure from this paper) without requiring adversarial training, and are considered by some to be the new contender to GANs.
- **GAN debiasing:** https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_GAN_Debiasing_(Optional).ipynb In this notebook, you will learn about Fair Attribute Classification through Latent Space De-biasing (Ramaswamy et al. 2020) that introduces a method for training accurate target classifiers while mitigating biases that stem from these correlations. Specifically, this work uses GANs to generate realistic-looking images and perturb these images in the underlying latent space to generate training data that is balanced for each protected attribute. They augment the original dataset with this perturbed generated data, and empirically demonstrate that target classifiers trained on the augmented dataset exhibit a number of both quantitative and qualitative benefits.
- **NeRF: Neural Radiance Fields:** https://colab.research.google.com/drive/18DladhUz7_U8iBkkQxMBk2f7C2NAvPCC?usp=sharing In this notebook, you'll learn how to use Neural Radiance Fields to generate new views of a complex 3D scene using only a couple input views, first proposed by NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al. 2020). Though 2D GANs have seen success in high-resolution image synthesis, NeRF has quickly become a popular technique to enable high-resolution 3D-aware GANs.

[comment]: <> ([![Anurag's GitHub stats]&#40;https://github-readme-stats.vercel.app/api?username=mirandalv&#41;]&#40;https://github.com/anuraghazra/github-readme-stats&#41;)

#### Week 3
#### Reading recommendations
- **StyleGAN**: A Style-Based Generator Architecture for Generative Adversarial Networks (Karras, Laine, and Aila, 2019): https://arxiv.org/abs/1812.04948
- GAN — StyleGAN & StyleGAN2 (Hui, 2020): https://medium.com/@jonathan_hui/gan-stylegan-stylegan2-479bdf256299
- Generative Adversarial Networks (Goodfellow et al., 2014): https://arxiv.org/abs/1406.2661
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016): https://arxiv.org/abs/1511.06434
- Coupled Generative Adversarial Networks (Liu and Tuzel, 2016): https://arxiv.org/abs/1606.07536
- Progressive Growing of GANs for Improved Quality, Stability, and Variation (Karras, Aila, Laine, and Lehtinen, 2018): https://arxiv.org/abs/1710.10196
- The Unusual Effectiveness of Averaging in GAN Training (Yazici et al., 2019): https://arxiv.org/abs/1806.04498v2
- StyleGAN - Official TensorFlow Implementation (Karras et al., 2019): https://github.com/NVlabs/stylegan
- StyleGAN Faces Training (Branwen, 2019): https://www.gwern.net/images/gan/2019-03-16-stylegan-facestraining.mp4
- Facebook AI Proposes Group Normalization Alternative to Batch Normalization (Peng, 2018): https://medium.com/syncedreview/facebook-ai-proposes-group-normalization-alternative-to-batch-normalization-fb0699bffae7

#### Python notebooks
- In this notebook, you're going to learn about StyleGAN2, from the paper Analyzing and Improving the Image Quality of StyleGAN (Karras et al., 2019), and how it builds on StyleGAN. This is the V2 of StyleGAN, so be prepared for even more extraordinary outputs. [a relative link](C2_BetterGAN/W3/StyleGAN2.ipynb)
- In this notebook, you'll learn about and implement the components of BigGAN, the first large-scale GAN architecture proposed in Large Scale GAN Training for High Fidelity Natural Image Synthesis (Brock et al. 2019). BigGAN performs a conditional generation task, so unlike StyleGAN, it conditions on a certain class to generate results. BigGAN is based mainly on empirical results and shows extremely good results when trained on ImageNet and its 1000 classes. [a relative link](C2_BetterGAN/W3/BigGAN.ipynb) 

## Course 3. Apply Generative Adversarial Networks (GANs)





