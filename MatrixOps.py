import numpy as np


class MatrixIsSingular(Exception):
    pass


def isSingular(A):
    """Check if the matrix A is singular."""
    B = np.array(A, dtype=np.float_)
    n = B.shape[0]
    for i in range(n):
        if not fixRow(B, i):
            return True
    return False


def fixRow(A, row):
    """Perform row operations to ensure the pivot element is non-zero and normalize the row."""
    n = A.shape[0]

    # Partial pivoting: Swap rows if the pivot is zero
    if A[row, row] == 0:
        for i in range(row + 1, n):
            if A[i, row] != 0:
                A[[row, i]] = A[[i, row]]  # Swap rows
                break
        else:
            # No non-zero pivot found
            raise MatrixIsSingular()

    # Normalize the pivot row
    A[row] = A[row] / A[row, row]

    # Eliminate the current column from subsequent rows
    for i in range(row + 1, n):
        A[i] = A[i] - A[i, row] * A[row]

    return True
