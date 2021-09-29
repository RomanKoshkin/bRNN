## Self-organization in a recurrent network of binary neurons &mdash; Single-core C++/Python implementation

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https:/travis-ci.org/joemccann/dillinger.svg?branch=master)

![Teaser image](https://svgshare.com/i/acV.svg)

## Release notes

This repository is a single-core C++/Python implementation of a recurrent network of binary neurons that given the right parameters, self-organizes to a fixed number of cell-assemblies, focusing on ease-of-use, performance and reproducibility and speed.

**Ease of use**
* Supports in-simulation parameter manipulation.
* Easy-to-use Python API

**Performance**
* Performance-critical code written in C++

**Reproducibility and Speed**
* Saves model states, model states can be loaded at any time, which speeds up experimentation.


## Requirements

* Linux (other OSs are not tested)
* Python 3.9
* Python libraries: `pip install scikit-learn==0.23.2 scikit-network==0.20.0 scipy==1.6.1 numpy==1.19.2 networkx==2.5 tqdm matplotlib==3.3.4`.


## Getting started

First compile the C++ module:

```.bash
g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_9_haga_grid.cpp -o
```

## Running a grid of simulations on a cluster

```.bash
sh grid_search.sh
```
This runs a grid of 10000 simulations. You can control the cells that will run in the `grid_serach.sh` script.

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.



## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

## License

Copyright &copy; 2021, Okinawa Institute of Science and Technology.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

Big thanks to Naoki Hiratani for his support.