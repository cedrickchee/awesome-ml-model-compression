# Awesome ML Model Compression [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

An awesome style list that curates the best machine learning model compression and acceleration research papers, articles, tutorials, libraries, tools and more. PRs are welcome!

# Contents

- [Papers](#papers)
  - [General](#general)
  - [Architecture](#architecture)
  - [Quantization](#quantization)
  - [Binarization](#binarization)
  - [Pruning](#pruning)
  - [Distillation](#distillation)
  - [Low Rank Approximation](#low-rank-approximation)
  - [Offloading](#offloading)
  - [Parallelism](#parallelism)
- [Articles](#articles)
  - [Howtos](#howtos)
  - [Assorted](#assorted)
  - [Reference](#reference)
  - [Blogs](#blogs)
- [Tools](#tools)
  - [Libraries](#libraries)
  - [Frameworks](#frameworks)
- [Videos](#videos)
  - [Talks](#talks)
  - [Training & tutorials](#training--tutorials)

---

## Papers

### General

- [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282)
- [Model compression as constrained optimization, with application to neural nets. Part I: general framework](https://arxiv.org/abs/1707.01209)
- [Model compression as constrained optimization, with application to neural nets. Part II: quantization](https://arxiv.org/abs/1707.04319)
- [Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better](https://arxiv.org/abs/2106.08962)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) by NVIDIA, Arm, and Intel, 2022 - FP8 delivered the performance of INT8 with accuracy of FP16. E4M3, a variant of FP8 has the benefits of INT8 with none of the loss in accuracy and throughput.

### Architecture

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
- [Fast YOLO: A Fast You Only Look Once System for Real-time Embedded Object Detection in Video](https://arxiv.org/abs/1709.05943)
- [AddressNet: Shift-based Primitives for Efficient Convolutional Neural Networks](https://arxiv.org/abs/1809.08458)
- [ResNeXt: Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- [ResBinNet: Residual Binary Neural Network](https://arxiv.org/abs/1711.01243)
- [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)
- [Squeezedet: Uniﬁed, small, low power fully convolutional neural networks](https://arxiv.org/abs/1612.01051)
- [SEP-Nets: Small and Effective Pattern Networks](https://arxiv.org/abs/1706.03912)
- [Dynamic Capacity Networks](https://arxiv.org/abs/1511.07838)
- [Learning Infinite Layer Networks Without the Kernel Trick](https://arxiv.org/abs/1606.05316v2)
- [Efficient Sparse-Winograd Convolutional Neural Networks](https://openreview.net/pdf?id=r1rqJyHKg)
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://openreview.net/pdf?id=HyoST_9xl)
- [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746v3)
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

### Quantization

- [Quantized Convolutional Neural Networks for Mobile Devices](https://arxiv.org/abs/1512.06473)
- [Towards the Limit of Network Quantization](https://arxiv.org/abs/1612.01543)
- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061)
- [Compressing Deep Convolutional Networks using Vector Quantization](https://arxiv.org/abs/1412.6115)
- [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
- [The ZipML Framework for Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning](https://arxiv.org/abs/1611.05402)
- [ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks](https://arxiv.org/abs/1706.02393)
- [Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/abs/1702.00953)
- [Loss-aware Binarization of Deep Networks](https://arxiv.org/abs/1611.01600)
- [Quantize weights and activations in Recurrent Neural Networks](https://arxiv.org/abs/1611.10176)
- [Fixed-Point Performance Analysis of Recurrent Neural Networks](https://arxiv.org/abs/1512.01322)
- [And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) [[blog post](https://huggingface.co/blog/hf-bitsandbytes-integration)]
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) by MIT and NVIDIA (2022) [[code](https://github.com/mit-han-lab/smoothquant)]
- [ZeroQuant: Efficient and Affordable Post-training Quantization for Large Transformer-based Models](https://arxiv.org/abs/2206.01861) by Microsoft (2022) [[code](https://github.com/microsoft/DeepSpeed)]
- [nuQmm: Quantized MatMul for Efficient Inference of Large-Scale Generative Language Models](https://arxiv.org/abs/2206.09557) by NAVER CLOVA and Pohang University of Science and Technology, Korea (2022)
- [MKQ-BERT: Quantized BERT with 4-bits Weights and Activations](https://arxiv.org/abs/2203.13483) by Tencent AIPD (2022)
- [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948) by Qualcomm AI Research (2021) [[code](https://github.com/qualcomm-ai-research/transformer-quantization)]
- [Mesa: A Memory-saving **Training** Framework for Transformers](https://arxiv.org/abs/2111.11124v1) by Monash University (2021)
- [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720) by Tim Dettmers et al. (2022) - Overall, their findings show that **4-bit precision is almost universally optimal for total model bits and zero-shot accuracy**.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) by Elias Frantar et al., 2022.
- Other lists:
  - [htqin/awesome-model-quantization](https://github.com/htqin/awesome-model-quantization)

### Binarization

- [Binarized Convolutional Neural Networks with Separable Filters for Efficient Hardware Acceleration](https://arxiv.org/abs/1707.04693)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [Local Binary Convolutional Neural Networks](https://arxiv.org/abs/1608.06049)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)

### Pruning

- [Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128)
- [Learning to Prune: Exploring the Frontier of Fast and Accurate Parsing](http://www.cs.jhu.edu/~jason/papers/vieira+eisner.tacl17.pdf)
- [Fine-Pruning: Joint Fine-Tuning and Compression of a Convolutional Network with Bayesian Optimization](https://arxiv.org/abs/1707.09102)
- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
- [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342)
- [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213)
- [Soft Weight-Sharing for Neural Network Compression](https://arxiv.org/abs/1702.04008)
- [Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493)
- [Channel pruning for accelerating very deep neural networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)
- [AMC: AutoML for model compression and acceleration on mobile devices](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf)
- [ESE: Efficient Speech Recognition Engine with Sparse LSTM on FPGA](https://arxiv.org/abs/1612.00694)
- [Massive Language Models Can Be Accurately Pruned in One-Shot (2023)](https://arxiv.org/abs/2301.00774) - Pruning methods: post-training, layer-wise. Quantization methods: joint sparsification & post-training quantization.
    > They propose `SparseGPT`, the first accurate one-shot pruning method which works efficiently at the scale of models with 10-100 billion parameters. `SparseGPT` works by reducing the pruning problem to an extremely large-scale instance of sparse regression. It is based on a new approximate sparse regression solver, used to solve a layer-wise compression problem, which is efficient enough to execute in a few hours on the largest openly-available GPT models (175B parameters), using a single GPU. At the same time, SparseGPT is accurate enough to drop negligible accuracy post-pruning, without any fine-tuning.
- [UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://arxiv.org/abs/2301.13741) by Tsinghua University et al. (ICML 2023) [[Code](https://github.com/sdc17/UPop)]
- [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) by CMU, Meta AI Research et al. (May 2024) - The popular approach known as magnitude pruning removes the smallest weights in a network based on the assumption that weights closest to 0 can be set to 0 with the least impact on performance. In LLMs, the magnitudes of a subset of outputs from an intermediate layer may be up to 20x larger than those of other outputs of the same layer. Removing the weights that are multiplied by these large outputs — even weights close to zero — could significantly degrade performance. Thus, a pruning technique that considers both weights and intermediate-layer outputs can accelerate a network with less impact on performance. Why it matters: The ability to compress models without affecting their performance is becoming more important as mobiles and personal computers become powerful enough to run them. [Code: [Wanda](https://github.com/locuslab/wanda)]

### Distillation

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Deep Model Compression: Distilling Knowledge from Noisy Teachers](https://arxiv.org/abs/1610.09650)
- [Learning Efficient Object Detection Models with Knowledge Distillation](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf)
- [Data-Free Knowledge Distillation For Deep Neural Networks](https://arxiv.org/abs/1710.07535)
- [Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks](https://arxiv.org/abs/1710.09505)
- [Moonshine: Distilling with Cheap Convolutions](https://arxiv.org/abs/1711.02613)
- [Model Distillation with Knowledge Transfer from Face Classification to Alignment and Verification](https://arxiv.org/abs/1709.02929)
- [Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/abs/1707.01219)
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- [Learning Loss for Knowledge Distillation with Conditional Adversarial Networks](https://arxiv.org/abs/1709.00513)
- [Dark knowledge](http://www.ttic.edu/dl/dark14.pdf)
- [DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer](https://arxiv.org/abs/1707.01220)
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)
- [MobileID: Face Model Compression by Distilling Knowledge from Neurons](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11977)
- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928)

### Low Rank Approximation

- [Speeding up convolutional neural networks with low rank expansions](http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14b/jaderberg14b.pdf)
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)
- [Convolutional neural networks with low-rank regularization](https://arxiv.org/abs/1511.06067)
- [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736)
- [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)
- [Efficient and Accurate Approximations of Nonlinear Convolutional Networks](https://arxiv.org/abs/1411.4229)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685 ) - Low-rank adapters were proposed for GPT-like models by Hu et al.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) by Tim Dettmers et al. (2023)

### Offloading

Recent years have witnessed the emergence of systems that are specialized for LLM inference, such as FasterTransformer (NVIDIA, 2022), PaLM inference (Pope et al., 2022), Deepspeed-Inference (Aminabadi et al., 2022), Accelerate (HuggingFace, 2022), LightSeq (Wang et al., 2021), TurboTransformers (Fang et al., 2021).

To enable LLM inference on easily accessible hardware, offloading is an essential technique — to our knowledge, among current systems, only Deepspeed-Inference and Huggingface Accelerate include such functionality.

- [FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU](https://raw.githubusercontent.com/FMInference/FlexGen/main/docs/paper.pdf) by HazyResearch@Stanford et al., 2023. [[Tweet]](https://archive.is/2bqSy)

### Parallelism

Compression methods for model acceleration (i.e., model parallelism) papers:

- [Does compressing activations help model parallel training? (2023)](https://arxiv.org/abs/2301.02654) - They presents the first empirical study on the effectiveness of compression algorithms (pruning-based, learning-based, and quantization-based - using a Transformer architecture) to improve the communication speed of model parallelism. **Summary:** 1) activation compression not equal to gradient compression; 2) training setups matter a lot; 3) don't compress early layers' activation.

## Articles

Content published on the Web.

### Howtos

- [How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
- [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware (2023)](https://huggingface.co/blog/peft) - The Hugging Face PEFT library enables using the most popular and performant models from Transformers coupled with the simplicity and scalability of Accelerate. Currently supported PEFT methods: LoRA, prefix tuning, prompt tuning, and P-Tuning (which employs trainable continuous prompt embeddings). They'll be exploring more PEFT methods, such as (IA)3 and bottleneck adapters. Results: The number of parameters needed to fine-tune Flan-T5-XXL is now 9.4M, about 7X fewer than AlexNet (source: [Tweet](https://twitter.com/dmvaldman/status/1624143468003221504)).

### Assorted

- [Why the Future of Machine Learning is Tiny](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/)
- [Deep Learning Model Compression for Image Analysis: Methods and Architectures](https://medium.com/comet-app/deep-learning-model-compression-for-image-analysis-methods-and-architectures-398f82b0c06f)
- [A foolproof way to shrink deep learning models](https://news.mit.edu/2020/foolproof-way-shrink-deep-learning-models-0430) by MIT (Alex Renda et al.) - A pruning algorithm: train to completion, globally prune the 20% of weights with the lowest magnitudes (the weakest connections), retrain with **learning rate rewinding** for the original (early training) rate, iteratively repeat until the desired sparsity is reached (model is as tiny as you want).

### Reference

### Blogs

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Demystifying the Compression of Large Language Models by Maarten Grootendorst (Jul 2024) - An approachable and great primer into quantization and widely supported quantization methods in tools and libraries including GPTQ, GGUF, and BitNet (1-bit).
- [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers) (Sept 2023)
- [TensorFlow Model Optimization Toolkit — Pruning API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?linkId=67380711)
- [Compressing neural networks for image classification and detection](https://ai.facebook.com/blog/compressing-neural-networks-for-image-classification-and-detection/) - Facebook AI researchers have developed a new method for reducing the memory footprint of neural networks by quantizing their weights, while maintaining a short inference time. They manage to get a 76.1% top-1 ResNet-50 that fits in 5 MB and also compress a Mask R-CNN within 6 MB.
- [All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html) - An overview of different compression methods for large NLP models (BERT) based on different characteristics and compares their results.
- [Deep Learning Model Compression](https://rachitsingh.com/deep-learning-model-compression/) methods.
- [Do We Really Need Model Compression](http://mitchgordon.me/machine/learning/2020/01/13/do-we-really-need-model-compression.html) in the future?
- Quantization: [Breakdown of Nvidia H100s for Transformer Inferencing](https://carolchen.me/blog/h100-inferencing/) by Carol Chen, ML ops at Cohere.
  > Transformer Engine utilizes FP8 and FP16 together to reduce memory usage and increase performance while still maintaining accuracy for large language models.
- [Comparison between quantization techniques and formats for LLMs](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)  (Oct 2023)- A detailed comparison between GGUF (llama.cpp), GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, model size, and loading time.
- [Which Quantization Method Is Best for You?: GGUF, GPTQ, or AWQ](https://archive.is/Yy9tE) (Jan 2024) - A gentle introduction to three prominent quantization methods — GPTQ, AWQ, and GGUF.
- [Comparing Quantized Performance in Llama Models](https://www.lesswrong.com/posts/qmPXQbyYA66DuJbht/comparing-quantized-performance-in-llama-models) (Jul 2024) - 8 bit quantized seems fine, for 4 bit it depends. It covers different quantization schemes including GGUF, [HQQ](https://mobiusml.github.io/hqq_blog/) (Half-Quadratic Quantization), AWQ, GPTQ, and BnB.

## Tools
  
### Libraries

- [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization). Accompanied blog post, [TensorFlow Model Optimization Toolkit — Pruning API](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?linkId=67380711)
- [XNNPACK](https://github.com/google/xnnpack) is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 (SSE2 level) platforms. It's a based on QNNPACK library. However, unlike QNNPACK, XNNPACK focuses entirely on floating-point operators.
- [Bitsandbytes](https://github.com/facebookresearch/bitsandbytes) is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers and quantization functions.
- [NNCP](https://bellard.org/nncp/) - An experiment to build a practical lossless data compressor with neural networks. The latest version uses a Transformer model (slower but best ratio). LSTM (faster) is also available.

### Frameworks

### Paper Implementations

- [facebookresearch/kill-the-bits](https://github.com/facebookresearch/kill-the-bits) - code and compressed models for the paper, "And the bit goes down: Revisiting the quantization of neural networks" by Facebook AI Research.

## Videos

### Talks

### Training & tutorials

- [ICML'21 Tutorial : Sparsity in Deep Learning: Pruning and growth for efficient inference and training](https://icml.cc/virtual/2021/tutorial/10845) - Organized by Torsten Hoefler, Dan Alistarh

### Courses
- [MIT 6.5940 TinyML and Efficient Deep Learning Computing](https://efficientml.ai) - Course by Song Han, MIT

## License

I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer.

* Code: [MIT](LICENSE) license. Copyright 2022- Cedric Chee
* Text content: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](http://creativecommons.org/licenses/by-sa/4.0/)
