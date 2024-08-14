# F1NN - Formula 1 Neural Network

The idea is to get an understanding of neural networks and predict a race outcome.


### Prerequisites

Following packages are required:

- [pytorch](https://pytorch.org)
- [tensorflow](https://www.tensorflow.org)
- [fastf1](https://docs.fastf1.dev)
- [pandas](https://pandas.pydata.org)


### Usage

Notice: For now, only pytorch is implemented. In the future, you should be able to choose which nn framework you want to use.

```bash
# get data from fastf1
python3 f1nn.py data

# train model with data
python3 f1nn.py train

# predict race outcome
python3 f1nn.py predict
```

To predict race outcome you will need to create a `grid.txt` file in the project root directory for input. An example can be seen here:

```
VER
LEC
RUS
SAI
PER
ALO
NOR
PIA
HAM
HUL
TSU
STR
ALB
RIC
MAG
BOT
ZHO
SAR
OCO
GAS
```


### Roadmap

This project is by far not done or in any way perfect. You can track the progress of further plans via issues. Progress so far:

- [x] Prototype
- [x] CLI
- [x] Pytorch
- [ ] Tensorflow
- [x] Input: Recency
- [ ] Input: Track
- [ ] Input: Weather


### Notice

This project is unofficial and is not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One Licensing B.V.
