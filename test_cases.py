import numpy as np
from MatrixOps import MatrixOps, MatrixIsSingular


def test_isSingular():
    """Test cases for the isSingular function."""
    matrices = [
        np.array([[2, 0, 0, 0],
                  [0, 3, 0, 0],
                  [0, 0, 4, 4],
                  [0, 0, 5, 5]]),  # Singular

        np.array([[0, 7, -5, 3],
                  [2, 8, 0, 4],
                  [3, 12, 0, 5],
                  [1, 3, 1, 3]])  # Non-singular
    ]

    for i, matrix in enumerate(matrices):
        try:
            result = isSingular(matrix)
            print(
                f"Matrix {i + 1} is {'singular' if result else 'non-singular'}.")
        except MatrixIsSingular:
            print(f"Matrix {i + 1} is singular.")


if __name__ == "__main__":
    test_isSingular()
