<h1 align="center"> Mask Condition Video Diffusion</h1>

<h3 align="center"> <a href="https://voletiv.github.io" target="_blank">Vikram Voleti</a>*, <a href="https://ajolicoeur.wordpress.com/about/" target="_blank">Alexia Jolicoeur-Martineau</a>*, <a href="https://sites.google.com/view/christopher-pal" target="_blank">Chris Pal</a></h3>

### <div align="center"> [Paper](xxxxxxxx), [Code](https://github.com/voletiv/ncsnv2-gff), Blog </div>

### <div align="center"> Summary </div>

* General purpose model for video generation, forward/backward prediction, and interpolation
* uses a [diffusion loss function](https://yang-song.github.io/blog/2021/score/)
* injects Gaussian noise into the current frames and denoise them conditional on past and future frames
* randomly mask past or/and future frames during training which allows the model to handle the four cases:
  * both past and present are known: interpolation
  * the past is unknown: past prediction
  * the future is unknown: future prediction
  * the past and future are unknown: unconditional generation
* high-quality and diverse video samples
* uses a simple 2D convolutional network; 3D convolutions or recurrent layers are not needed.
* runs on 1-4 GPUs
* scales very well with the number of channels and could be scaled much further than in the paper

### <div align="center"> Abstract </div>

Current state-of-the-art (SOTA) methods for video prediction/generation generally require complex 3D Transformers or recurrent auto-encoders. In spite of these complex architectures, results often remain low quality due to significant underfitting. On the other hand, the very few methods that don't underfit often require complex data augmentations to prevent severe overfitting which limits generalization beyond the training data. Meanwhile, we achieve SOTA results with a simple architecture with no recurrent layer, expensive 3D convolution, space-time attention, or complex data augmentation (except for the Gaussian noise added by the diffusion loss). They key to achieving such high quality videos is the use of a diffusion loss function and conditioning on past frames through concatenation or space-time adaptive normalization.
{: style="text-align: justify"}


### Video Prediction : cond=5, train=5, autoregressive pred=20

![SMMNIST_big_c5t5_SPADE](./SMMNIST_big_c5t5_SPADE_videos_300000.gif "SMMNIST c5t5")
