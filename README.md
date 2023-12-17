## What is torchcast? <img align="right" src="docs/source/_static/torchcast_logo.png">

torchcast is a library for time series forecasting, classification, and regression in PyTorch. It focuses primarily on making it easy to fetch, preprocess, and iterate over time series datasets. **It is still under heavy construction.**

## Installation

torchcast includes C++ code that must be compiled to be used. To install torchcast, first make sure that you have g++ installed, along with Python, including its development headers. Then, run:

```
git clone https://github.com/TheoremEngine/torchcast.git
cd torchcast
python3 setup.py install
```