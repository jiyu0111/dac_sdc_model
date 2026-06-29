1. Model architecture:
   HLSLikeQATDetector, int4 QAT model, 7 classes, input size 640x320.
   model structure and initial weight are from https://github.com/PKU-SEC-Lab/dac-sdc-2023-designs/tree/main/FPGA_Track_First_Place

2. Dataset format:
   Images: data/JPEGImages/*.jpg
   Labels: data/label/*.json
   JSON fields: type, x, y, width, height
   type 1~7 maps to class 0~6.
   Training data could be found at https://drive.google.com/file/d/1ceQ5y_rCReSZ26HzzCf2muDNbovjyl5k/view?usp=share_link which is provided by DAD_SDC Contest 2023(https://dac-sdc.github.io/2023/info/)    

3. Training setting:
   batch_size = 16
   val_ratio = 0.1
   random seed = 42
   image size = 640x320
   grid size = 40x20
   anchors = [(6,7), (13,10), (11,20), (24,16), (41,30), (90,64)]

4. Initialization:
   The model is initialized from weights.hpp using champ_weight_loader.py.

5. Training schedule:
   Head-only warmup for 200 epochs with Adam lr=1e-3.
   Then unfreeze all layers with Adam lr=5e-5.
   Total epochs: 251.