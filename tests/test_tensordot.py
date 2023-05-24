import unittest
import pytest

import numpy as np

import tocha
import torch


class TestTensorDot(unittest.TestCase):
    def test_simple_matmul(self):
        t1 = tocha.tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)  # (3,2)
        t2 = tocha.tensor([[10], [20]], requires_grad=True)  # (2,1)
        t3 = t1 @ t2

        T1 = tocha.tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)  # (3,2)
        T2 = tocha.tensor([[10], [20]], requires_grad=True)  # (2,1)
        T3 = tocha.tensordot(T1, T2, ((1,), (0,)))

        assert t3 == T3
        assert t3.data.tolist() == [[50], [110], [170]]
        assert T3.data.tolist() == [[50], [110], [170]]

        grad = tocha.tensor([[-1], [-2], [-3]])
        t3.backward(grad)
        T3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)  # type: ignore
        np.testing.assert_array_equal(t2.grad.data, t1.data.T @ grad.data)  # type: ignore
        np.testing.assert_array_equal(t1.grad.data, T1.grad.data)  # type: ignore
        np.testing.assert_array_equal(t2.grad.data, T2.grad.data)  # type: ignore

    def test_three_dim(self):
        t1 = tocha.tensor(np.ones((3, 4, 2)), requires_grad=True)
        t2 = tocha.tensor(np.ones((2, 4, 5)), requires_grad=True)
        T1 = tocha.tensor(np.ones((3, 4, 2)), requires_grad=True)
        T2 = tocha.tensor(np.ones((2, 4, 5)), requires_grad=True)
        idx = ((2,), (0,))

        t3 = tocha.tensordot(t1, t2, idx)  # Xabc * Ycde -> Zabde
        T3 = np.tensordot(t1.data, t2.data, idx)
        np.testing.assert_array_almost_equal(t3.data, T3)

        grad = tocha.tensor(np.random.rand(*t3.shape))  # (3, 4, 4, 5)
        t3.backward(grad)

        # Compute the gradients using the backward pass
        T1_grad = np.tensordot(
            grad.data, t2.data, ((2, 3), (1, 2))
        )  # Zabde * Ycde -> Xabc
        T2_grad = np.tensordot(grad.data, t1.data, ((0, 1), (0, 1))).transpose(
            [2, 0, 1]
        )  # Zabde * Xabc -> Ydec -> Ycde
        np.testing.assert_array_almost_equal(t1.grad.data, T1_grad)  # type: ignore
        np.testing.assert_array_almost_equal(t2.grad.data, T2_grad)  # type: ignore

    def test_four_dim(self):
        t1 = tocha.tensor(np.ones((3, 4, 5, 6)), requires_grad=True)
        T1 = tocha.tensor(np.ones((3, 4, 5, 6)), requires_grad=True)
        t2 = tocha.tensor(np.ones((7, 6, 5, 4)), requires_grad=True)
        T2 = tocha.tensor(np.ones((7, 6, 5, 4)), requires_grad=True)
        idx = ((1, 3), (3, 1))

        t3 = tocha.tensordot(t1, t2, idx)  # Xabdc * Ydcfe -> Zbafe
        T3 = np.tensordot(T1.data, T2.data, idx)
        np.testing.assert_array_almost_equal(t3.data, T3)

        grad = tocha.tensor(np.random.rand(*t3.shape))  # (3, 5, 7, 5)
        # print(grad.shape, t3.shape)

        t3.backward(grad)
        assert t1.shape == t1.grad.shape  # type: ignore
        assert t2.shape == t2.grad.shape  # type: ignore

        # # Compute the gradients using the backward pass
        T1_grad = np.tensordot(grad.data, t2.data, ((2, 3), (0, 2))).transpose(
            [0, 3, 1, 2]
        )  # Zbafe * Ydcfe -> Xabdc
        T2_grad = np.tensordot(grad.data, t1.data, ((0, 1), (0, 2))).transpose(
            [0, 3, 1, 2]
        )  # Zbafe * Xabdc -> Yfecd -> Ydcfe
        np.testing.assert_array_almost_equal(t1.grad.data, T1_grad)  # type: ignore
        np.testing.assert_array_almost_equal(t2.grad.data, T2_grad)  # type: ignore

    def test_five_dim(self):
        t1 = tocha.tensor(np.ones((3, 4, 5, 6, 7)), requires_grad=True)
        T1 = tocha.tensor(np.ones((3, 4, 5, 6, 7)), requires_grad=True)
        t2 = tocha.tensor(np.ones((8, 7, 6, 5, 4)), requires_grad=True)
        T2 = tocha.tensor(np.ones((8, 7, 6, 5, 4)), requires_grad=True)
        idx = ((1, 3, 4), (4, 2, 1))

        t3 = tocha.tensordot(t1, t2, idx)  # Xabdec * Ydecfe -> Zbafe
        T3 = np.tensordot(T1.data, T2.data, idx)
        np.testing.assert_array_almost_equal(t3.data, T3)

        grad = tocha.tensor(np.random.rand(*t3.shape))  # (3, 5, 8, 5)
        t3.backward(grad)

        assert t1.shape == t1.grad.shape  # type: ignore
        assert t2.shape == t2.grad.shape  # type: ignore

    def test_six_dim(self):
        t1 = tocha.tensor(np.ones((2, 3, 4, 5, 6, 7)), requires_grad=True)
        t2 = tocha.tensor(np.ones((7, 6, 5, 4, 3, 2)), requires_grad=True)
        idx = ((3, 5), (2, 0))

        t3 = tocha.tensordot(t1, t2, idx)  # Xabcfde * Yefcdgb -> Xabgdb
        T3 = np.tensordot(t1.data, t2.data, idx)
        np.testing.assert_array_almost_equal(t3.data, T3)

        grad = tocha.tensor(np.random.rand(*t3.shape))
        t3.backward(grad)

        assert t1.shape == t1.grad.shape  # type: ignore
        assert t2.shape == t2.grad.shape  # type: ignore

    def test_seven_dim(self):
        t1 = tocha.tensor(np.ones((2, 3, 4, 5, 6, 7, 8)), requires_grad=True)
        t2 = tocha.tensor(np.ones((8, 7, 6, 5, 4, 3, 2)), requires_grad=True)
        idx = ((2, 4, 6), (4, 2, 0))

        t3 = tocha.tensordot(t1, t2, idx)  # Xabcgfde * Ygfedcba -> Xbaedc
        T3 = np.tensordot(t1.data, t2.data, idx)
        np.testing.assert_array_almost_equal(t3.data, T3)

        grad = tocha.tensor(np.random.rand(*t3.shape))
        t3.backward(grad)

        assert t1.shape == t1.grad.shape  # type: ignore
        assert t2.shape == t2.grad.shape  # type: ignore
        print(grad.shape, t2.shape)

        # T1_grad = np.tensordot(grad.data, t2.data, ((2, 3, 4), (0, 2, 4))).transpose(
        #     [0, 3, 1, 2, 4, 5, 6]
        # )  # Xbaedc * Ygfedcba -> Xabcgfde
        # assert T1_grad.shape == t1.grad.shape  # type: ignore

    def setUp(self):
        # seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
    def test_tensordot(self):
        for i in range(1, 5):  # Vary the number of dimensions
            for j in range(1, i+1):  # Vary the places of matching dimensions
                with self.subTest(i=i, j=j):
                    shapeA = np.random.randint(1, 5, size=i).tolist()
                    shapeB = np.random.randint(1, 5, size=i).tolist()
                    
                    axes = None
                    chosenA = []
                    chosenB = []
                    for k in range(j):
                        k1 = np.random.choice(range(len(shapeA)))
                        while k1 in chosenA:
                            k1 = np.random.choice(range(len(shapeA)))
                        k2 = np.random.choice(range(len(shapeB)))
                        while k2 in chosenB:
                            k2 = np.random.choice(range(len(shapeB)))
                        chosenA.append(k1)
                        chosenB.append(k2)
                        shapeA[k1] = shapeB[k2]
                        if axes is None:
                            axes = ((k1,), (k2,))
                        else:
                            axes = (axes[0] + (k1,), axes[1] + (k2,))
                    
                    Bnp = np.random.randn(*shapeB)
                    Anp = np.random.randn(*shapeA)                    
                    
                    A = tocha.tensor(Anp, requires_grad=True)
                    B = tocha.tensor(Bnp, requires_grad=True)
                    A_torch = torch.tensor(Anp, requires_grad=True, dtype=torch.float32)
                    B_torch = torch.tensor(Bnp, requires_grad=True, dtype=torch.float32)
                
                            
                    C = tocha.tensordot(A, B, axes=axes) # type: ignore
                    C_torch = torch.tensordot(A_torch, B_torch, dims=axes) # type: ignore
                    
                    grad = tocha.tensor(np.ones_like(C).astype(np.float32))
                    grad_torch = torch.ones_like(C_torch)
                    
                    C.backward(grad)
                    C_torch.backward(grad_torch)
                    
                    self.assertTrue(np.allclose(C.data, C_torch.data.numpy()))  # type: ignore
                    self.assertTrue(np.allclose(A.grad.data, A_torch.grad.numpy()))  # type: ignore
