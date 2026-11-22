# DTIS

**Density and Text Informed Medical Image Segmentation**

DTIS is a lung lesion segmentation method built on nnU-Net v2. At the encoder
bottleneck, it first fuses features from a precomputed density map with image
features, then modulates the fused representation with a global text embedding.

This repository contains the DTIS model and trainer, data preparation utilities,
and segmentation evaluation scripts.
