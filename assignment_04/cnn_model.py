########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK
########################################################################

import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, init_weights=False):
        # TODO: consider the AlexNet implementation
        super(ConvNet, self).__init__()
        #INITIALIZE LAYERS HERE
        first_layer = nn.Sequential(
            nn.Conv2d(3, 256, 7),

        )

        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
		#PASS IMAGE X THORUGH EACH LAYER DEFINED ABOVE
        out = 
        return out

    def _initialize_weights(self):
        #INITIALIZE WEIGHTS

