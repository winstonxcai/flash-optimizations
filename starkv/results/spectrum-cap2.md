# STAR-CSA spectrum

Capture: `starkv/captures/cap2`
Retention target for the rank profile: **0.990**

`selffit` fits and scores the same rows (the oracle ceiling). `heldout`
fits on one half and scores the other (a basis frozen before these
latents existed). `drift` fits on the earliest quarter of the capture
window and scores the latest (what a longer session costs). `eps_*` is
the relative logit error on the entries the model actually selected.

## Layer 2

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8317 | 0.7500 | 0.7180 | -- | -- |
| 192 | 324 | 0.9052 | 0.8310 | 0.8010 | -- | -- |
| 256 | 388 | 0.9490 | 0.8904 | 0.8649 | -- | -- |
| 320 | 456 | 0.9757 | 0.9360 | 0.9166 | -- | -- |
| 384 | 520 | 0.9917 | 0.9722 | 0.9612 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 4

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9059 | 0.8470 | 0.8022 | -- | -- |
| 192 | 324 | 0.9507 | 0.9052 | 0.8698 | -- | -- |
| 256 | 388 | 0.9751 | 0.9437 | 0.9191 | -- | -- |
| 320 | 456 | 0.9892 | 0.9702 | 0.9551 | -- | -- |
| 384 | 520 | 0.9971 | 0.9905 | 0.9833 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 384 (520 B, retention 0.9905); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 6

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8768 | 0.8048 | 0.7497 | -- | -- |
| 192 | 324 | 0.9310 | 0.8704 | 0.8266 | -- | -- |
| 256 | 388 | 0.9629 | 0.9177 | 0.8851 | -- | -- |
| 320 | 456 | 0.9825 | 0.9537 | 0.9323 | -- | -- |
| 384 | 520 | 0.9942 | 0.9813 | 0.9705 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 8

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8758 | 0.8029 | 0.7495 | -- | -- |
| 192 | 324 | 0.9288 | 0.8666 | 0.8217 | -- | -- |
| 256 | 388 | 0.9602 | 0.9119 | 0.8783 | -- | -- |
| 320 | 456 | 0.9800 | 0.9475 | 0.9245 | -- | -- |
| 384 | 520 | 0.9927 | 0.9766 | 0.9636 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 10

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8420 | 0.7500 | 0.6859 | -- | -- |
| 192 | 324 | 0.9101 | 0.8303 | 0.7740 | -- | -- |
| 256 | 388 | 0.9510 | 0.8901 | 0.8441 | -- | -- |
| 320 | 456 | 0.9766 | 0.9369 | 0.9057 | -- | -- |
| 384 | 520 | 0.9923 | 0.9744 | 0.9581 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 12

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8554 | 0.7539 | 0.6813 | -- | -- |
| 192 | 324 | 0.9161 | 0.8292 | 0.7675 | -- | -- |
| 256 | 388 | 0.9531 | 0.8863 | 0.8381 | -- | -- |
| 320 | 456 | 0.9766 | 0.9320 | 0.8998 | -- | -- |
| 384 | 520 | 0.9916 | 0.9702 | 0.9533 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 14

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8543 | 0.7558 | 0.6880 | -- | -- |
| 192 | 324 | 0.9141 | 0.8300 | 0.7725 | -- | -- |
| 256 | 388 | 0.9508 | 0.8854 | 0.8404 | -- | -- |
| 320 | 456 | 0.9748 | 0.9308 | 0.8995 | -- | -- |
| 384 | 520 | 0.9906 | 0.9687 | 0.9516 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 16

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8461 | 0.7421 | 0.6716 | -- | -- |
| 192 | 324 | 0.9087 | 0.8186 | 0.7581 | -- | -- |
| 256 | 388 | 0.9474 | 0.8772 | 0.8286 | -- | -- |
| 320 | 456 | 0.9728 | 0.9245 | 0.8907 | -- | -- |
| 384 | 520 | 0.9898 | 0.9654 | 0.9475 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 18

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8625 | 0.7633 | 0.6827 | -- | -- |
| 192 | 324 | 0.9196 | 0.8372 | 0.7709 | -- | -- |
| 256 | 388 | 0.9546 | 0.8917 | 0.8401 | -- | -- |
| 320 | 456 | 0.9772 | 0.9359 | 0.9015 | -- | -- |
| 384 | 520 | 0.9918 | 0.9724 | 0.9545 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 20

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8470 | 0.7470 | 0.6676 | -- | -- |
| 192 | 324 | 0.9096 | 0.8247 | 0.7605 | -- | -- |
| 256 | 388 | 0.9483 | 0.8836 | 0.8336 | -- | -- |
| 320 | 456 | 0.9734 | 0.9299 | 0.8956 | -- | -- |
| 384 | 520 | 0.9900 | 0.9682 | 0.9497 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 22

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8703 | 0.7931 | 0.7329 | -- | -- |
| 192 | 324 | 0.9264 | 0.8628 | 0.8125 | -- | -- |
| 256 | 388 | 0.9594 | 0.9113 | 0.8726 | -- | -- |
| 320 | 456 | 0.9800 | 0.9484 | 0.9231 | -- | -- |
| 384 | 520 | 0.9927 | 0.9773 | 0.9646 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 24

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8315 | 0.7416 | 0.6799 | -- | -- |
| 192 | 324 | 0.9035 | 0.8270 | 0.7755 | -- | -- |
| 256 | 388 | 0.9468 | 0.8885 | 0.8484 | -- | -- |
| 320 | 456 | 0.9738 | 0.9343 | 0.9083 | -- | -- |
| 384 | 520 | 0.9905 | 0.9704 | 0.9567 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 26

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8812 | 0.8018 | 0.7362 | -- | -- |
| 192 | 324 | 0.9316 | 0.8669 | 0.8138 | -- | -- |
| 256 | 388 | 0.9615 | 0.9130 | 0.8721 | -- | -- |
| 320 | 456 | 0.9805 | 0.9491 | 0.9212 | -- | -- |
| 384 | 520 | 0.9928 | 0.9775 | 0.9636 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 28

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9044 | 0.8473 | 0.7841 | -- | -- |
| 192 | 324 | 0.9464 | 0.8989 | 0.8509 | -- | -- |
| 256 | 388 | 0.9713 | 0.9364 | 0.9013 | -- | -- |
| 320 | 456 | 0.9868 | 0.9654 | 0.9430 | -- | -- |
| 384 | 520 | 0.9961 | 0.9878 | 0.9769 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 30

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8870 | 0.8145 | 0.7470 | -- | -- |
| 192 | 324 | 0.9373 | 0.8764 | 0.8240 | -- | -- |
| 256 | 388 | 0.9665 | 0.9210 | 0.8815 | -- | -- |
| 320 | 456 | 0.9842 | 0.9550 | 0.9286 | -- | -- |
| 384 | 520 | 0.9948 | 0.9819 | 0.9681 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 32

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8418 | 0.7513 | 0.6835 | -- | -- |
| 192 | 324 | 0.9071 | 0.8287 | 0.7703 | -- | -- |
| 256 | 388 | 0.9470 | 0.8861 | 0.8400 | -- | -- |
| 320 | 456 | 0.9730 | 0.9311 | 0.8998 | -- | -- |
| 384 | 520 | 0.9899 | 0.9686 | 0.9531 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 34

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8546 | 0.7624 | 0.6852 | -- | -- |
| 192 | 324 | 0.9147 | 0.8373 | 0.7767 | -- | -- |
| 256 | 388 | 0.9514 | 0.8914 | 0.8423 | -- | -- |
| 320 | 456 | 0.9751 | 0.9347 | 0.9009 | -- | -- |
| 384 | 520 | 0.9907 | 0.9706 | 0.9542 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 36

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9259 | 0.8874 | 0.8437 | -- | -- |
| 192 | 324 | 0.9579 | 0.9257 | 0.8916 | -- | -- |
| 256 | 388 | 0.9773 | 0.9536 | 0.9297 | -- | -- |
| 320 | 456 | 0.9894 | 0.9748 | 0.9590 | -- | -- |
| 384 | 520 | 0.9966 | 0.9907 | 0.9830 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 384 (520 B, retention 0.9907); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 38

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8517 | 0.7748 | 0.7106 | -- | -- |
| 192 | 324 | 0.9179 | 0.8559 | 0.8026 | -- | -- |
| 256 | 388 | 0.9568 | 0.9119 | 0.8731 | -- | -- |
| 320 | 456 | 0.9808 | 0.9540 | 0.9288 | -- | -- |
| 384 | 520 | 0.9949 | 0.9860 | 0.9745 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 40

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8058 | 0.7118 | 0.6521 | -- | -- |
| 192 | 324 | 0.8850 | 0.8013 | 0.7495 | -- | -- |
| 256 | 388 | 0.9341 | 0.8662 | 0.8251 | -- | -- |
| 320 | 456 | 0.9660 | 0.9179 | 0.8895 | -- | -- |
| 384 | 520 | 0.9871 | 0.9620 | 0.9470 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

smallest rank at drift >= 0.990: 448 (584 B, retention 1.0000); measured over r=128..448

## Layer 42

rows sampled: 2708

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9594 | 0.9404 | 0.9179 | -- | -- |
| 192 | 324 | 0.9860 | 0.9768 | 0.9650 | -- | -- |
| 256 | 388 | 0.9953 | 0.9911 | 0.9854 | -- | -- |
| 320 | 456 | 0.9983 | 0.9963 | 0.9936 | -- | -- |
| 384 | 520 | 0.9996 | 0.9989 | 0.9978 | -- | -- |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | -- | -- |

smallest rank at heldout >= 0.990: 256 (388 B, retention 0.9911); measured over r=128..448

smallest rank at drift >= 0.990: 320 (456 B, retention 0.9936); measured over r=128..448
