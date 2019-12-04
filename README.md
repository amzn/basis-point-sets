# Efficient Learning on Point Clouds with Basis Point Sets

Sergey Prokudin, Christoph Lassner, Javier Romero
ICCV 2019

![Teaser Image](bps.gif)

Basis Point Set (BPS) is a simple and efficient method for encoding 3D point  clouds into fixed-length representation.
This representation can be used  as input to arbitrary machine learning methods, in particular it can be used
 as input to off-the-shelf neural networks. 
 
 Check our [ICCV 2019 paper](https://arxiv.org/abs/1908.09186) for more 
 details.
 
 
## Usage

### Requirements

- Python 3.7

### Installation


```
pip3 install git+ssh://git.amazon.com/pkg/Bps
```

### Code snippet


```
import numpy as np
from bps import bps

# batch of 100 point clouds to convert
x = np.random.normal(size=[100, 2048, 3])

# optional point cloud normalization to fit the unit sphere
x_norm = bps.normalize(x)

# option 1: encode with 1024 random basis and distances as features
x_bps_random = bps.encode(x_norm, bps_arrangement='random', n_bps_points=1024, bps_cell_type='dists')

# option 2: encode with 32^3 grid basis and full vectors to nearest points as features
x_bps_grid = bps.encode(x_norm, bps_arrangement='grid', n_bps_points=32**3, bps_cell_type='deltas')
x_bps_grid = x_bps_grid.reshape([-1, 32, 32, 32, 3])

```

## Citation

If you find our work useful in your research, please consider citing:
```
@article{prokudin2019efficient,
  title={Efficient Learning on Point Clouds with Basis Point Sets},
  author={Prokudin, Sergey and Lassner, Christoph and Romero, Javier},
  journal={arXiv preprint arXiv:1908.09186},
  year={2019}
}
```
## License

This library is licensed under the MIT-0 License. See the LICENSE file.

