import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

S0 = 20
b = 4
a = 0
epsilon = 0.01
sigma = lambda x: (1.5 * x + 4) / 100
mu = lambda x: 0.05 * x
func = lambda t: mu(t) - (sigma(t) ** 2) / 2
t_list = np.linspace(a, b, num=9)
integral_trap = []
error_list = [0]
total_error = []
integral_monte = []
S = []
B_t = []
N_iteration = []
n_list = [0]
N_total = 50000  # Общее число итераций


# Rn(f) = max|f"(x)|*(b-a)h^2/12; Rn(f)<e; h=(b-a)/n; n=(b-a)/h
# Вторая производная: 0.05x-(0.5x+4)^2/2 -> -0.4

def n_optimal(b, a):
    # Вычисляем такой n, чтобы ошибка удовлетворяла условию Rn<epsilon
    for i in range(1, 1000):
        h = (b - a) / i
        Rn = np.max(abs(-0.4)) * ((b - a) * h ** 2) / 12
        if Rn < epsilon:
            n = i
            h = (b - a) / n
            break
    # n_optimal: n%8=0
    for i in range(1, 1000):
        if n % 8 != 0:
            n += i
        else:
            break
    print("Оптимальный n кратный 8: ", n)
    return n


# МЕТОД ТРАПЕЦИЙ
def trapz(t_list, N, f):
    for i, j in enumerate(t_list):
        if i == 0:  # Интеграл = 0, если a=b=0
            integral_trap.append(0) if f == func else B_t.append(20)
        else:
            n = int(N / 8 + 8 * (i - 1))
            h = (j - a) / n
            t_values = np.linspace(a, j, n + 1)
            Rn = 0.4 * ((j - 0) * h ** 2) / 12
            integral = (f(0) + f(j)) / 2 + (np.sum(f(t_values[1:n])))
            if f == func:
                integral_trap.append(h * integral + Rn)
                error_list.append(Rn)
                n_list.append(n)
            else:  # Для котировок облигаций
                B_t.append(S0 * np.exp(h * integral + Rn))
    return integral_trap if f != mu else B_t


# МЕТОД МОНТЕ-КАРЛО
def monte_carlo_integral(t_list, N_total):
    for i, j in enumerate(t_list):
        if i == 0:  # Интеграл = 0, если a=b=0
            N_iteration.append(0)
            error_list.append(0)
            integral_monte.append(0)
        else:
            N = int((i) / 8 * N_total)
            N_iteration.append(N)
            X_k = t_list[i] * np.random.normal(0, 1, N)  # Равномерное распределение
            W_j = np.sqrt(t_list[i]) * np.random.normal(0, 1, N)  # Стохастическая часть
            integral = np.sum(sigma(X_k) * W_j) / N_total
            integral_monte.append(integral)
            error_mc = np.abs(integral) / np.sqrt(N)
            error_list.append(error_mc)
    return integral_monte


def stochastic_integration():
    n = n_optimal(b, a)
    t = trapz(t_list, n, func)
    B = trapz(t_list, n, mu)
    total_error = error_list.copy()
    error_list.clear()
    m = monte_carlo_integral(t_list, N_total)
    total_error = [i + j for i, j in zip(total_error, error_list)]
    for i in range(len(t_list)):
        S_combined = S0 * np.exp(t[i] + m[i])
        S.append(S_combined)
    return S, B, total_error


S, B, total_error = stochastic_integration()

plt.figure(figsize=(10, 5))
plt.plot(t_list, S, marker='o', linestyle='-', label="Цена актива (Метод Трапеций + Монте-Карло)")
plt.xlabel("Время t")
plt.ylabel("S(t)")
plt.title("Процесс ценообразования рискового актива")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_list, S, marker='o', linestyle='-', label="S(t) - Цена актива")
plt.plot(t_list, B, marker='s', linestyle='--', label="B(t) - Котировки облигаций")
plt.xlabel("Время t")
plt.ylabel("Стоимость")
plt.title("Сравнение цены актива и котировок облигаций")
plt.legend()
plt.grid()
plt.show()

print("Результаты вычислений:")
print(f"{'t (годы)':<10} {'Число разбиений n':<20} {'Число итераций N':<20} {'Ошибка':<15} {'S(t)':<15} {'B(t)'}")
print("-" * 100)
for i, t in enumerate(t_list):
    print(f"{t:<10.1f} {n_list[i]:<20} {N_iteration[i]:<20} {total_error[i]:<15.6f} {S[i]:<15.6f} {B_t[i]:.6f}")


def mu(S, t):
    return 0.05 * t * S


def sigma(S, t):
    return (1.5*t + 4) / 100 * S


T = 4.0         # Конечное время
N = 80000       # Количество шагов
dt = T / N      # Размер шага

# Стохастический метод Рунге-Кутта 4-го порядка (SRK4)
def srk4(f, g, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n - 1):
        h = t[i+1] - t[i]
        current_t, current_y = t[i], y[i]

        Z = np.random.normal(0, 1)

        # Коэффициенты RK4
        k1 = f(current_y, current_t) * h + g(current_y, current_t) * np.sqrt(h) * Z
        k2 = f(current_y + k1/2, current_t + h/2) * h + g(current_y + k1/2, current_t + h/2) * np.sqrt(h) * Z
        k3 = f(current_y + k2/2, current_t + h/2) * h + g(current_y + k2/2, current_t + h/2) * np.sqrt(h) * Z
        k4 = f(current_y + k3, current_t + h) * h + g(current_y + k3, current_t + h) * np.sqrt(h) * Z

        y[i+1] = current_y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return y

# Генерация временной сетки
t_values = np.linspace(0, T, N+1)

# Решение SRK4
S_srk4 = srk4(mu, sigma, S0, t_values)

# Выбираем каждую 10000-ю точку
indices = np.arange(0, N+1, 10000)
t_sampled = t_values[indices]
S_srk4_sampled = S_srk4[indices]

# Данные для метода трапеций + Монте-Карло
t_prev = np.linspace(0, 4, 9)

# Интерполяция SRK4 на точки сравнения
S_srk4_interp = np.interp(t_prev, t_values, S_srk4)

# Вычисление ошибок
errors = np.abs(S_srk4_interp - S)
relative_errors = (errors / S * 100)

# Визуализация
plt.figure(figsize=(12, 6))

# Ломаная SRK4
plt.plot(t_sampled, S_srk4_sampled, marker='o', linestyle='-', label="SRK4 (разреженные точки)", linewidth=2)

# Точки метода трапеций + Монте-Карло
plt.plot(t_prev, S, marker='s', linestyle='-', color='red', label="Трапеции + Монте-Карло")

plt.xlabel("Время (t)")
plt.ylabel("S(t)")
plt.title("Сравнение SRK4 и метода трапеций + Монте-Карло (разреженные точки)")
plt.legend()
plt.grid(True)
plt.show()

# Вывод результатов сравнения
print("Сравнение решений:")
print(f"{'t':<5} {'SRK4':<20} {'Трапеции+МК':<20} {'Абс. ошибка':<15} {'Отн. ошибка (%)'}")
print("-" * 80)
for i in range(len(t_prev)):
    print(f"{t_prev[i]:<5} {S_srk4_interp[i]:<20.6f} {S[i]:<20.6f} {errors[i]:<15.6f} {relative_errors[i]:.2f}%")

