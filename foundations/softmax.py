import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val=np.max(z)
        each_e=np.exp(z-max_val)
        ans=each_e/np.sum(each_e)
        return np.round(ans,4)