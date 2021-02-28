# marioKartAI
AI plays Mario Kart

![Super Mario Kart](img/Mario+Kart+64.jpg)

<!-- ![controller](controller.png) -->

## Record

Define `OFFSET_Y = 60` in `utils.py`

![record_samples](img/record_samples.png)

## Viewer

[TensorKart](https://github.com/kevinhughes27/TensorKart) viewer

Run `python utils.py viewer samples` to view the samples (stored at the root of the samples folder)

![tensorkart_viewer](img/tensorkart_viewer.png)

## Train AI

Run `python utils.py prepare samples/*` with an array of sample directories to build an `X` and `y` matrix for training. (zsh will expand samples/* to all the directories. Passing a glob directly also works)

`X` is a 3-Dimensional array of images

`y` is the expected joystick ouput as an array:

```
  [0] joystick x axis
  [1] joystick y axis
  [2] button a
  [3] button b
  [4] button rb
```

The Deep Learning model used is the one from NVIDIA in this famous [paper](https://arxiv.org/pdf/1604.07316.pdf) from 2016:
![nvidia_network](img/nvidia_network.png)

## AI Playing

Almost there... WIP

### h5py error

Apparently h5py >= 3 results in the [following problem](https://github.com/tensorflow/tensorflow/issues/44467)([stackoverflow thread](https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi)):
```
AttributeError: 'str' object has no attribute 'decode'
```

To solve it:
`pip install h5py==2.10.0 --user`

![error_weight](img/error_weight.png)

### clipboard error

`OSError: failed to open clipboard`

![error_clipboard](img/error_clipboard.png)

## Sources

* [TensorKart](https://github.com/kevinhughes27/TensorKart)
    * [Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow)
* [NeuralKart](https://github.com/rameshvarun/NeuralKart)
* [NEAT Mario Kart 64](https://github.com/nicknlsn/MarioKart64NEAT)
* [BizHawk](https://github.com/TASVideos/BizHawk)
    * [Super Mario Kart ROM](https://www.emulatorgames.net/roms/super-nintendo/super-mario-kart/)
    * [Mario Kart 64 ROM](https://wowroms.com/en/roms/nintendo-64/mario-kart-64-usa/24662.html)
* MariFlow
    * [download](https://sethbling.s3-us-west-2.amazonaws.com/Downloads/MariFlow.zip)
    * [doc](https://docs.google.com/document/d/1p4ZOtziLmhf0jPbZTTaFxSKdYqE91dYcTNqTVdd6es4/edit#)
* [MarIQ](https://sethbling.s3-us-west-2.amazonaws.com/Downloads/MarIQ.zip)
    * [download](https://sethbling.s3-us-west-2.amazonaws.com/Downloads/MarIQ.zip)
    * [doc](https://docs.google.com/document/d/1uxzeSMqj56YGWh8LkzfNriuGvA3aWU3olg-MSCgWuSI/edit)
