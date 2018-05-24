# salinet
Experimental implementation of saliency detection network

This is an incomplete implementation of the network described in Najibi et. al's [Towards the Success Rate of One: Real-time Unconstrained Salient Object Detection](https://arxiv.org/abs/1708.00079).

## TODO
- [x] Single bounding box generation (`utils.py:single_box_gen`)
- [ ] Multiple bounding box generation (`utils.py:multi_box_gen`)
- [x] Ground truth saliency map generation (`utils.py:ground_truth_saliency_map`)
- [ ] Multi-task loss (`losses.py`)
- [x] VGG network implementation (`rsd_vgg.py`)