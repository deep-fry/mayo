# Multi-precision

This is an example to showcase how `mayo` can handle multi-precision or even multi-arithmetic overriding for different layers in a CNN.

```bash
./my \
datasets/mnist.yaml \
models/examples/lenet5_multiprecision.yaml \
trainers/lenet5.yaml \
overriders-update info train interact
```

* Command breakdown

    In the above command, we first issue an update of the overriders, this might be absent for fixed-point quantizers, but stays useful for quantizers like `Dynamic fixed-point`.

    `info` shows the details about this fixed-point quantizations, which may have information that you are intrested in (total number of bits, precision per layer, etc).

    `train` will simply train the model with respect to the training parameters declared in `trianers/lenent5.yaml`.

    `interact` can be quickly accessed by interrupting the trianing session. You will then enter an interactive session which means you can access to the current `tf.Session`. Try print out things like `list(self.overriders.values())[0]['weights'].after.eval()` to see whether values are correctly overriden.

* YAML breakdown

    The main overriding information is in `models/examples/lenet5_multiprecision.yaml`.
    `weights_low` and `weights_high` define a 4-bit and 8-bit fixed-point quantization respectively with the integer bits set to 2.

    We set `conv0` to use the high-precision fixed-point while all other parts remain using low-precision fixed points.
