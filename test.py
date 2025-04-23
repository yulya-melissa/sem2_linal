import unittest
from pca_ver3 import *


class TestMatrix(unittest.TestCase):
    def test_init_valid_matrix(self):
        """Проверка корректной инициализации матрицы."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertEqual(m.n, 2)
        self.assertEqual(m.m, 2)
        self.assertEqual(m.values, [[1, 2], [3, 4]])

    def test_init_valid_matrix(self):
        """Проверка корректной инициализации матрицы."""
        m = Matrix(2, 3, [[1, 2, 3], [3, 4, 3]])
        self.assertEqual(m.n, 2)
        self.assertEqual(m.m, 3)
        self.assertEqual(m.values, [[1, 2, 3], [3, 4, 3]])

    def test_init_invalid_row_count(self):
        """Проверка ошибки при несоответствии количества строк."""
        with self.assertRaises(ValueError):
            Matrix(2, 2, [[1, 2]])

    def test_init_invalid_row_length(self):
        """Проверка ошибки при несоответствии длины строк."""
        with self.assertRaises(ValueError):
            Matrix(2, 2, [[1, 2], [3, 4, 5]])

    def test_add_matrices(self):
        """Проверка сложения матриц."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = m1 + m2
        self.assertEqual(result.values, [[6, 8], [10, 12]])

    def test_add_incompatible_matrices(self):
        """Проверка ошибки при сложении матриц разного размера."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(AttributeError):
            m1 + m2

    def test_mul_scalar(self):
        """Проверка умножения матрицы на скаляр."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m * 2
        self.assertEqual(result.values, [[2, 4], [6, 8]])

    def test_mul_scalar(self):
        """Проверка умножения матрицы на скаляр слева"""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = 2 * m
        self.assertEqual(result.values, [[2, 4], [6, 8]])
    
    def test_mul_matrix(self):
        """Проверка умножения матриц."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = m1 * m2
        self.assertEqual(result.values, [[19, 22], [43, 50]])

    def test_mul_matrix(self):
        """Проверка умножения неквадратных матриц.1"""
        m1 = Matrix(3, 2, [[1, 2], [3, 4], [1, 1]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = m1 * m2
        self.assertEqual(result.values, [[19, 22], [43, 50], [12, 14]])

    def test_mul_matrix(self):
        """Проверка умножения неквадратных матриц.2"""
        m1 = Matrix(3, 2, [[1, 2], [3, 4], [1, 1]])
        m2 = Matrix(2, 3, [[1, 0, 1], [1, 2, 0]])
        result = m1*m2
        self.assertEqual(result.values, [
            [3, 4, 1],
            [7, 8, 3],
            [2, 2, 1]
            ])

    def test_mul_matrix(self):
        """Проверка умножения неквадратных матриц.3"""
        m1 = Matrix(3, 2, [[1, 2], [3, 4], [1, 1]])
        m2 = Matrix(2, 3, [[1, 0, 1], [1, 2, 0]])
        result = m2*m1
        self.assertEqual(result.values, [
            [2, 3],
            [7, 10]
            ])

    def test_str_representation(self):
        """Проверка строкового представления матрицы."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertEqual(str(m), "1 2 \n3 4 \n")

    def test_transponse(self):
        """Проверка транспонирования матрицы."""
        m = Matrix(2, 2, [[1, 1], [0, 1]])
        m1 = m.T()
        m2 = Matrix(2, 2, [[1, 0], [1, 1]])
        self.assertEqual(m1.values, m2.values)


class TestCovarianceMatrix(unittest.TestCase):
    def setUp(self):
        # Пример центрированной матрицы 3x2
        self.X_centered = Matrix(
            n=3,
            m=2,
            values=[
                [-2.0, -2.0],
                [0.0, 0.0],
                [2.0, 2.0]
            ]
        )

        self.expected_cov = Matrix(
            n=2,
            m=2,
            values=[
                [4.0, 4.0],
                [4.0, 4.0]
            ]
        )

    def test_covariance_matrix(self):
        cov = covariance_matrix(self.X_centered)
        self.assertEqual(cov.n, self.expected_cov.n)
        self.assertEqual(cov.m, self.expected_cov.m)
        # Проверяем значения с небольшим допуском из-за возможных численных ошибок
        for i in range(cov.n):
            for j in range(cov.m):
                self.assertAlmostEqual(cov.values[i][j], self.expected_cov.values[i][j], places=7)

class TestQRDecomposition(unittest.TestCase):
    def setUp(self):
        # Пример простой матрицы 3x3
        self.A_values = [
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0]
        ]
        self.A = Matrix(3, 3, self.A_values)

    def test_qr_decomposition(self):
        Q, R = qr_decomposition(self.A)

        # Проверяем размеры
        self.assertEqual(Q.n, 3)
        self.assertEqual(Q.m, 3)
        self.assertEqual(R.n, 3)
        self.assertEqual(R.m, 3)

        # Проверяем, что Q ортонормирована: Q^T Q = I
        QT = Q.T()
        I_approx = QT * Q
        for i in range(I_approx.n):
            for j in range(I_approx.m):
                if i == j:
                    self.assertAlmostEqual(I_approx.values[i][j], 1.0, places=6)
                else:
                    self.assertAlmostEqual(I_approx.values[i][j], 0.0, places=6)

        # Проверяем, что R верхнетреугольная
        for i in range(1, R.n):
            for j in range(i):
                self.assertAlmostEqual(R.values[i][j], 0.0, places=6)

        # Проверяем, что A ≈ Q * R
        QR = Q * R
        for i in range(QR.n):
            for j in range(QR.m):
                self.assertAlmostEqual(QR.values[i][j], self.A.values[i][j], places=5)

class TestQRAlgorithm(unittest.TestCase):
    def setUp(self):
        # Пример симметричной матрицы 3x3 с известными собственными значениями
        self.A_values = [
            [6, 0, 0],
            [0, 3, 0],
            [0, 0, 1]
        ]
        self.A = Matrix(3, 3, self.A_values)

        self.B_values = [
            [1, 1],
            [0, 2]
        ]
        self.B = Matrix(2, 2, self.B_values)

    def test_eigenvalues_and_vectors(self):
        eigenvalues, eigenvectors = qr_algorithm(self.A, max_iter=5000, tol=1e-6)

        expected_eigenvalues = [6.0, 3.0, 1.0] 

        # Проверяем, что найденные собственные значения близки к ожидаемым (не обязательно в порядке)
        for val in expected_eigenvalues:
            self.assertTrue(any(abs(val - ev) < 1e-1 for ev in eigenvalues))

        # Проверяем, что eigenvectors ортонормированы: eigenvectors^T * eigenvectors = I
        eigenvectors_T = eigenvectors.T()
        I_approx = eigenvectors * eigenvectors_T
        for i in range(I_approx.n):
            for j in range(I_approx.m):
                if i == j:
                    self.assertAlmostEqual(I_approx.values[i][j], 1.0, places=5)
                else:
                    self.assertAlmostEqual(I_approx.values[i][j], 0.0, places=5)

    def test_eigenvalues_and_vectors_2(self):
        eigenvalues, eigenvectors = qr_algorithm(self.B, max_iter=5000, tol=1e-6)

        expected_eigenvalues = [1.0, 2.0] 

        # Проверяем, что найденные собственные значения близки к ожидаемым (не обязательно в порядке)
        for val in expected_eigenvalues:
            self.assertTrue(any(abs(val - ev) < 1e-1 for ev in eigenvalues))

        # Проверяем, что eigenvectors ортонормированы: eigenvectors^T * eigenvectors = I
        eigenvectors_T = eigenvectors.T()
        I_approx = eigenvectors * eigenvectors_T
        for i in range(I_approx.n):
            for j in range(I_approx.m):
                if i == j:
                    self.assertAlmostEqual(I_approx.values[i][j], 1.0, places=5)
                else:
                    self.assertAlmostEqual(I_approx.values[i][j], 0.0, places=5)

class TestGetProj(unittest.TestCase):
    def setUp(self):
        self.eigen_num = [10.0, 5.0, 7.0]
   
        self.eigen_vec = Matrix(3, 3, [
            [1, 0, 0], 
            [0, 1, 0],   
            [0, 0, 1]
        ])

    def test_select_one_vector(self):
        k = 1
        proj = get_proj(copy.deepcopy(self.eigen_num), self.eigen_vec, k)
        expected_values = [
            [1],  
            [0],
            [0]  
        ]
        expected = Matrix(3, 1, expected_values)
        self.assertEqual(proj.values, expected.values)

    def test_select_two_vector(self):
        k = 2
        proj = get_proj(copy.deepcopy(self.eigen_num), self.eigen_vec, k)
        expected_values = [
            [1, 0],  
            [0, 0],
            [0, 1]  
        ]
        expected = Matrix(3, 2, expected_values)
        self.assertEqual(proj.values, expected.values)

class TestPCA(unittest.TestCase):
    def setUp(self):
        # Простая матрица данных 4x3
        self.X = Matrix(4, 3, [
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 0.8],
            [2.2, 2.9, 1.1],
            [1.9, 2.2, 0.9]
        ])

    def test_pca_output_shapes(self):
        k = 2
        X_proj, V_k, ratio = pca(self.X, k)

        # Проверяем размерности
        self.assertEqual(X_proj.n, self.X.n)
        self.assertEqual(X_proj.m, k)

        self.assertEqual(V_k.n, self.X.m)
        self.assertEqual(V_k.m, k)

        # Доля объяснённой дисперсии — число от 0 до 1
        self.assertTrue(0 <= ratio <= 1)

    def test_pca_invalid_k(self):
        with self.assertRaises(ValueError):
            pca(self.X, 0)
        with self.assertRaises(ValueError):
            pca(self.X, -1)

class TestHandleMissingValues(unittest.TestCase):
    def test_replace_nan_with_mean(self):
        data = [
            [1.0, 2.0, float('nan')],
            [float('nan'), 3.0, 3.0],
            [2.0, float('nan'), 4.0]
        ]
        X = Matrix(3, 3, data)
        filled = handle_missing_values(X)

        # Средние по столбцам:
        # столбец 0: (1 + 2) / 2 = 1.5
        # столбец 1: (2 + 3) / 2 = 2.5
        # столбец 2: (3 + 4) / 2 = 3.5

        expected_values = [
            [1.0, 2.0, 3.5],
            [1.5, 3.0, 3.0],
            [2.0, 2.5, 4.0]
        ]
        expected = Matrix(3, 3, expected_values)
        self.assertEqual(filled, expected)

    def test_no_nan(self):
        data = [
            [1, 2],
            [3, 4]
        ]
        X = Matrix(2, 2, data)
        filled = handle_missing_values(X)
        self.assertEqual(filled, X)

    def test_all_nan_in_column(self):
        data = [
            [float('nan'), 2],
            [float('nan'), 3]
        ]
        X = Matrix(2, 2, data)
        filled = handle_missing_values(X)
        # В первом столбце среднее не определено, значения NaN останутся
        expected_values = [
            [0, 2],
            [0, 3]
        ]
        expected = Matrix(2, 2, expected_values)
        self.assertEqual(filled, expected)

if __name__ == "__main__":
    unittest.main()