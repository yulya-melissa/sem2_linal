
from typing import List
import copy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import typing as tp
import random
import pandas as pd


class Matrix:
    def __init__(
        self,
        n: int,
        m: int,
        values: tp.List[tp.List[float]],
    ) -> None:
        """
        Основной констуктор матриц разреженного-строчного типа хранения
        (Для тестов)

        :param n: Кол-во строк
        :param m: Кол-во столбцов
        :param values: Матрица значений(двумерный массив)
        """
        self.n = n
        self.m = m
        if len(values) != n:
            raise ValueError ("Кoличество строк матрицы должно соответствовать указаным размерам")

        for i in range(n):
            if len(values[i]) != m:
                raise ValueError ("Все строки должнгы быть одного размера")

            
        self.values = copy.deepcopy(values)


    def __add__(self, other) -> tp.Self:  # перегрузка оператора сложения (для реализации метода сложения матриц)

        if not isinstance(other, Matrix):
            raise AttributeError("Can't sum a matrix and a non-matrix type") # проверка на типизацию

        if self.n != other.n or self.m != other.m:
            raise AttributeError("Can't sum matrices of different sizes") # проверка размерностей

        mtrx_sum = Matrix(n=self.n, m=self.m, values=self.values) # сюда будем записывать значения суммы матриц

        for i in range(self.n): # проходим по строкам
            for j in range(self.m):
                mtrx_sum.values[i][j] += other.values[i][j]

        return mtrx_sum
    
    def __mul__(self, other) -> tp.Self:
        if not isinstance(other, (Matrix, float, int)):
            raise AttributeError("Invalid multiplication")

        # Умножение на скаляр
        if isinstance(other, (float, int)):
            new_values = [
                [self.values[i][j] * other for j in range(self.m)]
                for i in range(self.n)
            ]
            return Matrix(n=self.n, m=self.m, values=new_values)

        # Умножение матриц
        if self.m != other.n:
            raise AttributeError("Can't multiply matrices of non-compatible sizes")

        new_values = [
            [
                sum(self.values[i][k] * other.values[k][j] for k in range(self.m))
                for j in range(other.m)
            ]
            for i in range(self.n)  # Исправлено: range(self.n) вместо range(self.m)
        ]
        
        return Matrix(n=self.n, m=other.m, values=new_values)

        
    def __rmul__(self, other) -> tp.Self:  # метод обратного умножения для коммутативности умножения на скаляр
        if not isinstance(other, (float, int)):
            raise AttributeError("Invalid multiplication")

        return self.__mul__(other)
    
    def __str__(self) -> str:  # метод для вывода экземпляра класса в обычном матричном виде (для относительно малых матриц)
        matrix = ''
        for i in range(self.n):
            for j in range(self.m):
                matrix += str(self.values[i][j]) + " "
            matrix += "\n"
        return matrix
    
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.n == other.n and self.m == other.m and self.values == other.values

    def T(self) -> tp.Self:  # транспонирование матрицы
        mtrx_t = [
            [self.values[i][j] for i in range(self.n)]
            for j in range(self.m)
        ]

        return Matrix(n=self.m, m=self.n, values=mtrx_t)
    
    def determinant(self) -> float:
        """
        Вычисляет определитель матрицы методом Гаусса.
        z - размер матрицы для которой нужно посчитать определитель
        :return: Определитель матрицы
        """
    
        if self.n != self.m:
            raise ValueError("The matrix is not square")
        
        matrix = copy.deepcopy(self.values)
        det = 1.0

        for i in range(self.n):
            # Поиск максимального элемента в текущем столбце для выбора ведущего элемента
            max_row = i
            for k in range(i, self.n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k

            # Если ведущий элемент ноль, то определитель равен нулю
            if matrix[max_row][i] == 0:
                return 0

            # Меняем строки местами, если нужно
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                det *= -1  # Меняем знак определителя из-за перестановки строк

            # Прямой ход метода Гаусса
            for j in range(i + 1, self.n):
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, self.n):
                    matrix[j][k] -= factor * matrix[i][k]

            # Умножаем определитель на диагональный элемент
            det *= matrix[i][i]
        return det

    
def gauss_solver(A: Matrix, b: Matrix, zero_err: float = 10e-6) -> tuple[Matrix, tp.List[Matrix]]:
    """
    Вход:
    A: матрица коэффициентов (n×n). Используется класс Matrix из предыдущей
    ,→ лабораторной работы
    b: вектор правых частей (n×1)
    zero_err: допустимая ошибка для нуля
    Выход:
    Matrix - частное решение системы
    list[Matrix]: список базисных векторов решения системы
    Raises:
    ValueError: если система несовместна
    """
    if (A.n != A.m): raise ValueError("Matrix is not square((")

    if (A.n != b.n): raise ValueError("Coefs and free terms dimensions do not correlate!")

    if (b.m != 1): raise ValueError("Free term vector is dimensity 1 on axis 1")

    ext = [A.values[i] + b.values[i] for i in range(A.n)]
    n = A.n
    
    for i in range(n):
        max_row = i
        # Находим максимальный по модулю элемент в столбце
        for j in range(i, n):
            if abs(ext[j][i]) > abs(ext[max_row][i]): max_row = j
        
        if ext[max_row][i] == 0: continue

        # Меняем местами строки чтобы максимальный элемент был ведущим
        ext[max_row], ext[i] = ext[i], ext[max_row]

        p = ext[i][i]

        # Обнуляем значения под диагональным элементом
        for j in range(i + 1, n):
            factor = ext[j][i] / p
            for k in range(i, n + 1):
                ext[j][k] -= ext[i][k] * factor
    
    
    X = [0] * n
    basis_vec = []
    free_vars = []

    # Обратный ход Гаусса для нахождения частного решения системы
    for i in range(n - 1, -1, -1):
        r = ext[i][n] - sum(ext[i][j] * X[j] for j in range(i + 1, n))
        if abs(ext[i][i]) < zero_err:
            if abs(r) > zero_err:
                raise ValueError("The system is inconsistent")
            free_vars.append(i)
            continue
        X[i] = r / ext[i][i]
    X = Matrix(n, 1, [[x] for x in X])

    # Нахождение базисных векторов если они есть
    for var in free_vars:
        vec = [0] * n
        vec[var] = 1
        for i in range(n - 1, -1, -1):
            if i in free_vars: continue
            vec[i] -= sum(ext[i][j] * vec[j] for j in range(i + 1, n)) / ext[i][i]
        basis_vec.append(Matrix(n, 1, [[x] for x in vec]))
    
    return X, basis_vec


def create_e_matrix(n:int)-> Matrix:
    """ Формирует квадратную единичую матрицу размером nxn
    Вход: n - размер матрицы
    Выход: экземплер класса матриц
    """
    mtrx_e = [
            [0 for i in range(n)]
            for j in range(n)
        ]
    for i in range(n):
        mtrx_e[i][i] = 1
    return Matrix(n, n, mtrx_e)

def create_zero_matrix(n:int)-> Matrix:
    """ Формирует квадратную единичую матрицу размером nxn
    Вход: n - размер матрицы
    Выход: экземплер класса матриц
    """
    mtrx_e = [
            [0 for i in range(n)]
            for j in range(n)
        ]
    return Matrix(n, n, mtrx_e)

def center_data(X: Matrix) -> Matrix:
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    # Счиатем среденее по стлобцам(сначала сумму)
    x_mean = [0 for i in range(X.m)]
    for i in range(X.n):
        for j in range(X.m):
            x_mean[j] += X.values[i][j]

    for i in range(X.m):
        x_mean[i] = x_mean[i] / X.n
    
    # Создаем новый экземпляр мтрицы
    mtrx_mean = Matrix(n=X.n, m=X.m, values=X.values)

    # Пересчитываем названия
    for i in range(X.n):
        for j in range(X.m):
            mtrx_mean.values[i][j] -= x_mean[j]

    return mtrx_mean

def covariance_matrix(X_centered: Matrix) -> Matrix:
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    return (1 / (X_centered.n - 1)) * (X_centered.T() * X_centered) 

def sq_norm(v: List[float]) -> float:
    """Евклидова норма вектора v."""
    return dot(v, v)**0.5

def scal_mul(v: List[float], scalar: float) -> List[float]:
    """Умножение вектора v на скаляр."""
    return [scalar * vi for vi in v]

def dot(v: List[float], x: List[float]) -> List[float]:
    """Умножение вектора v на x."""
    if len(v) != len(x):
        raise ValueError("Vecors in diferent dimentional")
    return sum([x[i] * v[i] for i in range(len(v))])

def u_minus_v (u: List[float], v: List[float]) -> List[float]:
    """Вычитание векторов u - v."""
    if len(v) != len(u):
        raise ValueError("Vecors in diferent dimentional")
    return [ui - vi for ui, vi in zip(u, v)]

def sq_norm_mat(A: Matrix) -> float:
    """Евклидова норма матрицы."""
    norm = 0
    for i in range(A.n):
        for j in range(A.m):
            norm += A.values[i][j]*A.values[i][j]

    return norm**0.5

def qr_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """
    QR-разложение матрицы A с помощью алгоритма Грама — Шмидта.

    :param A: матрица размером n x n (список списков)
    :return: кортеж (Q, R), где
             Q — матрица с ортонормированными столбцами (n x n),
             R — верхнетреугольная матрица (n x n)
    """

    # Транспонируем A, чтобы работать со столбцами как с векторами
    A_T = A.T().values
    n = A.m

    Q = []
    R = [[0.0] * n for _ in range(n)]

    for j in range(A.m):
        v = A_T[j][:]  # копируем j-й столбец

        for i in range(j):
            # Q[i] - ортонорм. вектор из нового базиса
            # A_T[j] - векторы из изначальной матрицы для которого проводим ортогонализацию
            R[i][j] = dot(Q[i], A_T[j])
            # Умножение вектора из ортонорм. базиса на скаляр
            proj = scal_mul(Q[i], R[i][j])
            # Вычисление ортогональной состовл. 
            v = u_minus_v(v, proj)

        R[j][j] = sq_norm(v)
        if R[j][j] == 0:
            raise ValueError("Матрица не имеет полного ранга, QR-разложение невозможно")

        q_j = scal_mul(v, 1 / R[j][j])
        Q.append(q_j)

    
    Q_l = Matrix(A.n, A.m, Q)

    return Q_l.T(), Matrix(A.n, A.n, R)

def qr_algorithm(A: Matrix, max_iter: int = 1000, tol: float = 1e-6) -> tuple[List[float], Matrix]:
    """
    QR-алгоритм для поиска собственных значений матрицы A.

    :param A: квадратная матрица (numpy.ndarray)
    :param max_iter: максимальное число итераций
    :param tol: точность сходимости
    :return: eigenvalues - список(матрица) собственных значений
    Q_all - строчки матрицы - собственные вектора
    """
    Ak = Matrix(A.n, A.n, A.values)
    Q_all = create_e_matrix(A.n)

    for _ in range(max_iter):
        Q, R = qr_decomposition(Ak)       # QR-разложение
        Ak_next = R * Q               # Перестановка множителей
        Q_all = Q_all * Q
        # Проверка сходимости по норме разности
        if sq_norm_mat(Ak_next + (-1)*Ak) < tol:
            break
        Ak = Ak_next

    eigenvalues = [Ak.values[i][i] for i in range(A.n)]

    return eigenvalues, Q_all

def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    tol: допустимая погрешность
    Выход: список вещественных собственных значений
    """
    if C.m != C.n:
        raise ValueError("Матрица должна быть квадратной")
    eigen_values, eigen_vectors = qr_algorithm(A=C, tol=tol)
    return sorted(eigen_values, reverse=True)
        
def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    if C.m != C.n:
        raise ValueError("Матрица должна быть квадратной")
    eigen_values, Q_all = qr_algorithm(C)
    eigen_vectors = [ Matrix(1, C.n, [Q_all.values[i]]).T() for i in range(C.n)]
    return eigen_vectors

def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент
    Выход: доля объяснённой дисперсии
    """
    eigenvalues.sort(reverse=True)
    return sum(eigenvalues[:k])/sum(eigenvalues)        

def get_proj(eigen_num: List[float], eigen_vec: Matrix, k: int) -> Matrix:
        lams = eigen_num.copy()
        vec_t = eigen_vec.values
        V = []
        for i in range(k):
            max_lam = max(lams)
            lam_num = lams.index(max_lam)
            lams[lam_num] = 0
            V.append(vec_t[lam_num])

        matr = Matrix(k, eigen_vec.n, V)
        return matr.T()

def pca(X: Matrix, k: int) -> tp.Tuple[Matrix, Matrix, float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """
    if k <= 0:
        raise ValueError("Make number of components positive please")
    centred_X = center_data(X)
    cov_X = covariance_matrix(centred_X)
    eigen_num, eigen_vec = qr_algorithm(cov_X)

    V_k = get_proj(eigen_num, eigen_vec, k)
    X_pro = X * V_k
    return X_pro, V_k, explained_variance_ratio(eigen_num, k)

def plot_pca_projection(X_proj: Matrix) -> Figure:
    """
    Вход: проекция данных X_proj (n×2)
    Выход: объект Figure из Matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Разворачиваем данные в два списка координат x и y
    x_vals = X_proj.T().values[0]
    y_vals = X_proj.T().values[1]

    ax.scatter(x_vals, y_vals, c='blue', alpha=0.6, edgecolors='w', s=50)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA Projection (2D)')
    ax.grid(True)
    
    return fig

def calc_recon_matrix(X: Matrix, X_pro: Matrix, V_k: Matrix):

    x_mean = [0]*X.m
    for i in range(X.n):
        for j in range(X.m):
            x_mean[j] += X.values[i][j]

    for i in range(X.m):
        x_mean[i] = x_mean[i] / X.n
    
    mtrx = X_pro * V_k.T() 

    # Пересчитываем названия
    for i in range(X.n):
        for j in range(X.m):
            mtrx.values[i][j] += x_mean[j]
    return mtrx
    
    
def reconstruction_error(X_orig: Matrix, X_recon: Matrix) -> float:
    """
    Вход:
    X_orig: исходные данные (n×m)
    X_recon: восстановленные данные (n×m)
    Выход: среднеквадратическая ошибка MSE
    """
    return sq_norm_mat(X_orig + (-1)*X_recon)**2 /(X_recon.m*X_orig.n)

def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
    eigenvalues: список собственных значений
    threshold: порог объяснённой дисперсии
    Выход: оптимальное число главных компонент k
    """
    for i in range(len(eigenvalues)):
        if explained_variance_ratio(eigenvalues, i) > threshold:
            return i


def handle_missing_values(X: Matrix) -> Matrix:
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    mean_values = [0]*X.m
    count_values = [0]*X.m
    new_X = Matrix(X.n, X.m, X.values)
    for i in range(X.n):
        for j in range(X.m):
            if X.values[i][j] == X.values[i][j]:
                mean_values[j] += X.values[i][j]
                count_values[j] += 1

    for i in range(X.m):
        if count_values[i] != 0:
            mean_values[i] = mean_values[i]/count_values[i]

    for i in range(X.n):
        for j in range(X.m):
            if new_X.values[i][j] != new_X.values[i][j]:
                new_X.values[i][j] = mean_values[j]
    return new_X

def std_dev(values: List[float]) -> float:
    mu = sum(values) / len(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return variance**0.5


def add_noise_and_compare(X: Matrix, noise_level: float = 0.1):
    """
    Вход:
    X: матрица данных (n×m)
    noise_level: уровень шума (доля от стандартного отклонения)
    Выход: результаты PCA до и после добавления шума.
    В этом задании можете проявить творческие способности, поэтому выходные данные не
    ,→ типизированы.
    """
    X_T = X.T()
    n_samples = X.n
    n_features = X.m

    # Вычисляем std для каждого признака
    stds = [std_dev(feature) for feature in X_T.values]

    # Генерируем шум и добавляем к данным
    X_noisy = []
    for i in range(n_samples):
        noisy_row = []
        for j in range(n_features):
            noise = random.gauss(0, stds[j] * noise_level)
            noisy_value = X.values[i][j] + noise
            noisy_row.append(noisy_value)
        X_noisy.append(noisy_row)
    X_noisy = Matrix(X.n, X.m, X_noisy)
    

    # Вычисляем оптимальное число компонент
    centred_X = center_data(X)
    cov_X = covariance_matrix(centred_X)
    eigen_num, eigen_vec = qr_algorithm(cov_X)
    

    centred_nX = center_data(X_noisy)
    cov_nX = covariance_matrix(centred_nX)
    eigen_num_n, eigen_vec_n = qr_algorithm(cov_nX)
    k = min(auto_select_k(eigen_num), auto_select_k(eigen_num_n))

    clear_pca, V, ev = pca(X, k)
    noisy_pca, V_n, ev_n = pca(X_noisy, k)
    # Вывод значений насколько хорошо созранились матрицы при одинаковых k
    var_rat = [ev, ev_n]
    # Насколько совпадают собственные ветора
    vect = [dot(eigen_vec.values[i], eigen_vec_n.values[i]) for i in range(eigen_vec.n)]
    # Норма разницы итоговых матриц
    norm = [reconstruction_error(X, calc_recon_matrix(X, clear_pca, V)), reconstruction_error(X_noisy, calc_recon_matrix(X_noisy, noisy_pca, V_n))]
    return var_rat, vect, norm

def apply_pca_to_dataset(dataset_name: str, k: int) -> tuple[Matrix, float, float]:
    """
    Вход:
    dataset_name: название датасета
    k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    df = pd.read_csv(dataset_name)
    df = df.select_dtypes(include=['number'])
    values = df.values.tolist()  # преобразуем в список списков
    n, m = df.shape
    values_float = [[float(x) for x in row] for row in values]
    X = Matrix(n, m, values_float)
    X = handle_missing_values(X)
    X_pro, V, ev = pca(X, k)
    er = reconstruction_error(X, calc_recon_matrix(X, X_pro, V))
    return X_pro, ev, er

