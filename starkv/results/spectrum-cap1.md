# STAR-CSA spectrum

Capture: `starkv/captures/cap1`
Retention target for the rank profile: **0.990**

`selffit` fits and scores the same rows (the oracle ceiling). `heldout`
fits on one half and scores the other (a basis frozen before these
latents existed). `drift` fits on the earliest quarter of the capture
window and scores the latest (what a longer session costs). `eps_*` is
the relative logit error on the entries the model actually selected.

## Layer 2

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9317 | 0.7665 | -- | -- | -- |
| 192 | 324 | 0.9769 | 0.8329 | -- | -- | -- |
| 256 | 388 | 0.9936 | -- | -- | -- | -- |
| 320 | 456 | 0.9988 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 4

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9623 | 0.8590 | -- | -- | -- |
| 192 | 324 | 0.9869 | 0.9008 | -- | -- | -- |
| 256 | 388 | 0.9960 | -- | -- | -- | -- |
| 320 | 456 | 0.9991 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 6

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9545 | 0.8296 | -- | -- | -- |
| 192 | 324 | 0.9840 | 0.8773 | -- | -- | -- |
| 256 | 388 | 0.9952 | -- | -- | -- | -- |
| 320 | 456 | 0.9989 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 8

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9557 | 0.8293 | -- | -- | -- |
| 192 | 324 | 0.9842 | 0.8766 | -- | -- | -- |
| 256 | 388 | 0.9951 | -- | -- | -- | -- |
| 320 | 456 | 0.9989 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 10

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9401 | 0.7877 | -- | -- | -- |
| 192 | 324 | 0.9789 | 0.8454 | -- | -- | -- |
| 256 | 388 | 0.9937 | -- | -- | -- | -- |
| 320 | 456 | 0.9986 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 12

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9513 | 0.8059 | -- | -- | -- |
| 192 | 324 | 0.9817 | 0.8571 | -- | -- | -- |
| 256 | 388 | 0.9940 | -- | -- | -- | -- |
| 320 | 456 | 0.9985 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 14

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9488 | 0.8118 | -- | -- | -- |
| 192 | 324 | 0.9801 | 0.8597 | -- | -- | -- |
| 256 | 388 | 0.9932 | -- | -- | -- | -- |
| 320 | 456 | 0.9983 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 16

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9465 | 0.8014 | -- | -- | -- |
| 192 | 324 | 0.9794 | 0.8520 | -- | -- | -- |
| 256 | 388 | 0.9931 | -- | -- | -- | -- |
| 320 | 456 | 0.9983 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 18

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9533 | 0.8263 | -- | -- | -- |
| 192 | 324 | 0.9818 | 0.8722 | -- | -- | -- |
| 256 | 388 | 0.9937 | -- | -- | -- | -- |
| 320 | 456 | 0.9984 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 20

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9458 | 0.8103 | -- | -- | -- |
| 192 | 324 | 0.9784 | 0.8587 | -- | -- | -- |
| 256 | 388 | 0.9926 | -- | -- | -- | -- |
| 320 | 456 | 0.9981 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 22

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9521 | 0.8193 | -- | -- | -- |
| 192 | 324 | 0.9832 | 0.8695 | -- | -- | -- |
| 256 | 388 | 0.9950 | -- | -- | -- | -- |
| 320 | 456 | 0.9989 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 24

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9386 | 0.7866 | -- | -- | -- |
| 192 | 324 | 0.9795 | 0.8445 | -- | -- | -- |
| 256 | 388 | 0.9942 | -- | -- | -- | -- |
| 320 | 456 | 0.9988 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 26

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9565 | 0.8372 | -- | -- | -- |
| 192 | 324 | 0.9832 | 0.8808 | -- | -- | -- |
| 256 | 388 | 0.9944 | -- | -- | -- | -- |
| 320 | 456 | 0.9986 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 28

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9658 | 0.8741 | -- | -- | -- |
| 192 | 324 | 0.9872 | 0.9088 | -- | -- | -- |
| 256 | 388 | 0.9958 | -- | -- | -- | -- |
| 320 | 456 | 0.9990 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 30

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9618 | 0.8388 | -- | -- | -- |
| 192 | 324 | 0.9871 | 0.8809 | -- | -- | -- |
| 256 | 388 | 0.9962 | -- | -- | -- | -- |
| 320 | 456 | 0.9991 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 32

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9388 | 0.7806 | -- | -- | -- |
| 192 | 324 | 0.9774 | 0.8384 | -- | -- | -- |
| 256 | 388 | 0.9929 | -- | -- | -- | -- |
| 320 | 456 | 0.9984 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 34

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9392 | 0.7872 | -- | -- | -- |
| 192 | 324 | 0.9756 | 0.8409 | -- | -- | -- |
| 256 | 388 | 0.9914 | -- | -- | -- | -- |
| 320 | 456 | 0.9978 | -- | -- | -- | -- |
| 384 | 520 | 0.9997 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 36

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9721 | 0.9082 | -- | -- | -- |
| 192 | 324 | 0.9891 | 0.9342 | -- | -- | -- |
| 256 | 388 | 0.9962 | -- | -- | -- | -- |
| 320 | 456 | 0.9990 | -- | -- | -- | -- |
| 384 | 520 | 0.9999 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 38

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9389 | 0.7925 | -- | -- | -- |
| 192 | 324 | 0.9777 | 0.8523 | -- | -- | -- |
| 256 | 388 | 0.9931 | -- | -- | -- | -- |
| 320 | 456 | 0.9984 | -- | -- | -- | -- |
| 384 | 520 | 0.9998 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 40

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9192 | 0.7413 | -- | -- | -- |
| 192 | 324 | 0.9698 | 0.8080 | -- | -- | -- |
| 256 | 388 | 0.9903 | -- | -- | -- | -- |
| 320 | 456 | 0.9978 | -- | -- | -- | -- |
| 384 | 520 | 0.9997 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count

## Layer 42

rows sampled: 491

| r | B | selffit | heldout | drift | eps_basis | eps_quant |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.9824 | 0.9367 | -- | -- | -- |
| 192 | 324 | 0.9954 | 0.9659 | -- | -- | -- |
| 256 | 388 | 0.9989 | -- | -- | -- | -- |
| 320 | 456 | 0.9998 | -- | -- | -- | -- |
| 384 | 520 | 1.0000 | -- | -- | -- | -- |
| 448 | 584 | 1.0000 | -- | -- | -- | -- |

smallest rank at heldout >= 0.990: none -- every measured rank (r=128..192) fell short

smallest rank at drift >= 0.990: NOT MEASURABLE -- no sampled rank could be fit: this split holds back only a quarter of the sample, so it needs a rank well below the row count
