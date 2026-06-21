import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        z=np.matmul(x,w)+b

        if activation=='sigmoid':
            ans=self.act_sig(z)
            return np.round(ans,5)
        
        if activation=='relu':
            ans=self.act_relu(z)
            return np.round(ans,5)
        

    def act_sig(self,z):
        ans=1/(1+np.exp(np.negative(z)))

        return ans

    def act_relu(self,z):
        ans=np.maximum(0,z)

        return ans