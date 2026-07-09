"""Step 2 - the encoder half: ConvConvRelu + repeatable 2ConvPool blocks."""
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_input('../examples/fcn8s/cats.jpg'),

    # block 1, written by hand: double conv + relu, then a pooling slab
    to_ConvConvRelu(name='ccr_b1', s_filer=500, n_filer=(64, 64),
                    offset="(0,0,0)", to="(0,0,0)",
                    width=(2, 2), height=40, depth=40),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)",
            width=1, height=32, depth=32, opacity=0.5),

    # blocks 2-4: the same motif, one factory call each
    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2',
                     s_filer=256, n_filer=128, offset="(1,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),
    *block_2ConvPool(name='b3', botton='pool_b2', top='pool_b3',
                     s_filer=128, n_filer=256, offset="(1,0,0)",
                     size=(25, 25, 4.5), opacity=0.5),
    *block_2ConvPool(name='b4', botton='pool_b3', top='pool_b4',
                     s_filer=64, n_filer=512, offset="(1,0,0)",
                     size=(16, 16, 5.5), opacity=0.5),

    to_end(),
]


def main():
    to_generate(arch, 'step2_encoder.tex')


if __name__ == '__main__':
    main()
