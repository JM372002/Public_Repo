import ctypes
import os

class BitMatrix:
    def __init__(self, rows, cols):
        # Load the compiled DLL
        dll_path = os.path.abspath(r"BinaryNetworks\custom_lib\bitmatrix.dll")
        self.lib = ctypes.CDLL(dll_path)

        # Declare argument and return types
        self.lib.create_matrix.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.create_matrix.restype = ctypes.c_void_p

        self.lib.free_matrix.argtypes = [ctypes.c_void_p]

        self.lib.set_bit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.clear_bit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.get_bit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.get_bit.restype = ctypes.c_int

        self.lib.print_matrix.argtypes = [ctypes.c_void_p]
        self.lib.print_matrix.restype = None

        self.lib.set_random_n_ones.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.set_random_n_ones.restype = None

        self.lib.get_memory_usage.argtypes = [ctypes.c_void_p]
        self.lib.get_memory_usage.restype = ctypes.c_int

        # Create the matrix
        self.rows = rows
        self.cols = cols
        self._m = self.lib.create_matrix(rows, cols)

    def __del__(self):
        # Clean up memory
        if hasattr(self, 'lib') and self._m:
            self.lib.free_matrix(self._m)

    def set(self, i, j):
        self.lib.set_bit(self._m, i, j)

    def clear(self, i, j):
        self.lib.clear_bit(self._m, i, j)

    def get(self, i, j):
        return self.lib.get_bit(self._m, i, j)

    def print_matrix(self):
        self.lib.print_matrix(self._m)

    def set_random_n_ones(self, n):
        self.lib.set_random_n_ones(self._m, n)

    def memory_usage(self):
        return self.lib.get_memory_usage(self._m)


if __name__ == "__main__":
    bm = BitMatrix(5, 13)

    dummy = (5 * 13) // 2
    bm.set_random_n_ones(dummy)
    bm.print_matrix()
    print("Memory usage in bytes:", bm.memory_usage())