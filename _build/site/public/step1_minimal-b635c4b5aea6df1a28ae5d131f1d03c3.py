"""Step 1 - a minimal CNN chain: the five primitives you need first."""
import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # input image, placed to the left of the first layer
    to_input('../examples/fcn8s/cats.jpg'),

    # conv -> pool -> conv -> pool -> softmax
    to_Conv("conv1", s_filer=224, n_filer=64, offset="(0,0,0)", to="(0,0,0)",
            width=2, height=40, depth=40, caption="Conv1"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
            width=1, height=32, depth=32, opacity=0.5),

    to_Conv("conv2", s_filer=112, n_filer=128, offset="(1.5,0,0)", to="(pool1-east)",
            width=3.5, height=32, depth=32, caption="Conv2"),
    to_connection("pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)",
            width=1, height=25, depth=25, opacity=0.5),

    to_SoftMax("soft1", s_filer=10, offset="(2,0,0)", to="(pool2-east)",
               caption="SoftMax"),
    to_connection("pool2", "soft1"),

    to_end(),
]


def main():
    to_generate(arch, 'step1_minimal.tex')


if __name__ == '__main__':
    main()
