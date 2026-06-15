import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum = 0
        ans = []
        max_z = np.max(z)
        for i in range(len(z)):
            sum+=np.exp(z[i] - max_z)
        for i in range(len(z)):
            ans.append(np.round(np.exp(z[i] - max_z)/sum,4))
        return ans
