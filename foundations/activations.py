import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        result = []
        for i in range(len(z)):
            ans = 1 / (1 + math.exp(-z[i]))
            result.append(np.round(ans, 5))
        return result
        # return np.round(your_answer, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        result = []
        for i in range(len(z)):
            result.append(np.round(max(0.0, z[i]),2))
        return result
