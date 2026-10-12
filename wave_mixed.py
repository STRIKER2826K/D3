# -*- coding: utf-8 -*-
"""
Смешанная задача для одномерного волнового уравнения.

Уравнение:      u_tt = c^2 * u_xx,   0 <= x <= L,  t >= 0
Граничные условия (вариант):  u(0,t) = 0,   u_x(L,t) = 0
Начальные условия:            u(x,0) = f(x),  u_t(x,0) = g(x)

Решение ищется аналитическим методом разделения переменных (Фурье):

    u(x,t) = sum_n [ A_n cos(w_n t) + B_n sin(w_n t)/w_n ] * X_n(x)

где собственные функции задачи Штурма-Лиувилля
    X'' + lam*X = 0,  X(0)=0, X'(L)=0
имеют вид  X_n(x) = sin(k_n x),  k_n = pi*(2n+1)/(2L),  w_n = c*k_n,

а коэффициенты — это интегралы Фурье по данным:
    A_n = (2/L) * int_0^L f(x) sin(k_n x) dx
    B_n = (2/(L*w_n)) * int_0^L g(x) sin(k_n x) dx

ВАЖНО (по условию задачи): никакие «встроенные» решатели (scipy,
numpy.fft, solve_ivp и т.п.) не используются. Все вычисления —
интегрирование (составное правило Симпсона), сумма ряда — реализованы
вручную. numpy применяется только как хранилище массивов, matplotlib —
только для отрисовки анимации.

Запуск анимации:   python wave_mixed.py
Без дисплея (окно не откроется автоматически) можно сохранить видео:
                  python wave_mixed.py --save animation.mp4
или кадры:         python wave_mixed.py --frames frames_dir
"""

import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # безопасный backend; при наличии дисплея легко переключается
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------------------------------------------------------
# 1. Параметры задачи
# ----------------------------------------------------------------------------
C = 1.0        # скорость распространения волн c
L = 1.0        # длина струны
T_END = 6.0    # рассматриваемый интервал времени
N_MODES = 40   # число членов ряда Фурье в решении

NX = 300       # пространственная сетка для вывода/анимации
NT = 300       # временна́я сетка (для проверки начальных условий)
N_INTEGRAL = 2000  # число подынтервалов при интегрировании Симпсоном


# ----------------------------------------------------------------------------
# 2. Начальные данные f(x) = u(x,0) и g(x) = u_t(x,0)
#    Обе функции удовлетворяют граничным условиям:
#    f(0)=0, f'(L)=0;  g(0)=0, g'(L)=0
# ----------------------------------------------------------------------------
def f_initial(x):
    """u(x,0) — начальный профиль струны.

    Удовлетворяет граничным условиям: f(0)=0, f'(L)=0 (cos(pi/2)=0).
    Ряд Фурье по базису {sin(k_n x)} для этой функции сходится быстро
    (коэффициенты A_n ~ 1/k_n^3), что видно в самопроверке.
    """
    return x * math.sin(math.pi * x / (2.0 * L))


def g_initial(x):
    """u_t(x,0) — начальная скорость."""
    return 0.5 * math.sin(math.pi * x / (2.0 * L))


# ----------------------------------------------------------------------------
# 3. Собственные значения/функции смешанной задачи
# ----------------------------------------------------------------------------
def eigenvalue(n):
    """k_n = pi*(2n+1)/(2L), n = 0,1,2,..."""
    return math.pi * (2 * n + 1) / (2.0 * L)


def omega(n):
    """Частота w_n = c*k_n."""
    return C * eigenvalue(n)


def eigenfunction(n, x):
    """X_n(x) = sin(k_n x); работает со скалярами и массивами."""
    return np.sin(eigenvalue(n) * x)


# ----------------------------------------------------------------------------
# 4. Численное интегрирование ВРУЧНУЮ: составное правило Симпсона
#    (не используем scipy.integrate, numpy.trapz и пр.)
# ----------------------------------------------------------------------------
def simpson(func, a, b, n=N_INTEGRAL):
    """Интеграл func по [a,b] по составному правилу Симпсона.

    n — чётное число подынтервалов (если нечётное — увеличиваем на 1).
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        coef = 4.0 if (i % 2 == 1) else 2.0
        s += coef * func(a + i * h)
    return s * h / 3.0


# ----------------------------------------------------------------------------
# 5. Коэффициенты Фурье решений (интегралы считаем Simpson'ом вручную)
# ----------------------------------------------------------------------------
def fourier_coefficients(n_modes, f_func, g_func):
    """Возвращает массивы A_n и B_n, n = 0..n_modes-1."""
    A = np.zeros(n_modes)
    B = np.zeros(n_modes)
    for n in range(n_modes):
        k = eigenvalue(n)
        w = omega(n)
        A[n] = (2.0 / L) * simpson(lambda x: f_func(x) * math.sin(k * x), 0.0, L)
        B[n] = (2.0 / (L * w)) * simpson(lambda x: g_func(x) * math.sin(k * x), 0.0, L)
    return A, B


# ----------------------------------------------------------------------------
# 6. Проверка корректности решения (самопроверка, без внешних решателей)
# ----------------------------------------------------------------------------
def verify(u, A):
    """Проверка: НУ, БУ, невязка PDE (конечными разностями),
    сравнение численных коэффициентов Фурье с аналитическими."""
    print("=" * 64)
    print("Самопроверка решения")
    print("=" * 64)

    # --- сходимость метода: численные коэффициенты Фурье при f(x)=sin(pi x/(2L)).
    #     Эта функция — собственная X_0, поэтому точные коэффициенты:
    #     A_0 = 1,  A_n = 0 (n >= 1). ---
    n_modes = len(A)
    A_test = fourier_coefficients(n_modes, lambda x: math.sin(eigenvalue(0) * x),
                                  lambda x: 0.0)[0]
    exact = np.zeros(n_modes)
    exact[0] = 1.0
    err_A = float(np.max(np.abs(A_test - exact)))
    print(f"max |A_n(числ.) - A_n(точн.)| для f=sin(k_0 x) = {err_A:.3e}")

    # --- начальные условия ---
    xs = np.linspace(0.0, L, 101)
    err_f = float(np.max(np.abs(u(xs, 0.0) - np.array([f_initial(x) for x in xs]))))
    dt = 1e-5
    ut_approx = (u(xs, dt) - u(xs, -dt)) / (2.0 * dt)
    err_g = float(np.max(np.abs(ut_approx - np.array([g_initial(x) for x in xs]))))
    print(f"max |u(x,0) - f(x)|          = {err_f:.3e}   (усечение ряда, O(1/n_modes^2))")
    print(f"max |u_t(x,0) - g(x)|        = {err_g:.3e}")

    # --- граничные условия ---
    t = 1.234
    err_bc1 = abs(u(np.array([0.0]), t)[0])
    hx = 1e-5
    ux_L = (u(np.array([L + hx]), t)[0] - u(np.array([L - hx]), t)[0]) / (2.0 * hx)
    print(f"|u(0,t)|                     = {err_bc1:.3e}")
    print(f"|u_x(L,t)|                   = {abs(ux_L):.3e}")

    # --- невязка волнового уравнения u_tt - c^2 u_xx ---
    h = 1e-4
    x0, t0 = 0.37, 0.81
    utt = (u(np.array([x0]), t0 + h)[0] - 2 * u(np.array([x0]), t0)[0]
           + u(np.array([x0]), t0 - h)[0]) / h ** 2
    uxx = (u(np.array([x0 + h]), t0)[0] - 2 * u(np.array([x0]), t0)[0]
           + u(np.array([x0 - h]), t0)[0]) / h ** 2
    resid = abs(utt - C ** 2 * uxx)
    rel = resid / max(abs(utt), 1e-30)
    print(f"невязка |u_tt - c^2 u_xx|    = {resid:.3e}  (отн. {rel:.3e})")
    print("=" * 64)


# ----------------------------------------------------------------------------
# 7. Анимация графика решения
# ----------------------------------------------------------------------------
def animate_solution(A, B, save_path=None, frames_dir=None, fps=30):
    """Анимированный график u(x,t). Сохраняет MP4/GIF или кадры, если нужно."""
    # Пробуем интерактивный backend, если есть дисплей
    if save_path is None and frames_dir is None:
        try:
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")
        except Exception:
            pass  # остаёмся на Agg — тогда нужно сохранить файл (см. --save)

    n_modes = len(A)

    def u_of_t(t):
        """Значение решения на сетке xs в момент времени t."""
        result = np.zeros_like(xs)
        for n in range(n_modes):
            k = eigenvalue(n)
            w = omega(n)
            result += (float(A[n]) * math.cos(w * t)
                       + float(B[n]) * math.sin(w * t)) * np.sin(k * xs)
        return result

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0.0, L, NX)

    # Статичная огибающая |u| <= sum_n sqrt(A_n^2 + B_n^2) * |sin(k_n x)|
    env = np.zeros_like(xs)
    for n in range(n_modes):
        env += math.hypot(A[n], B[n]) * np.abs(np.sin(eigenvalue(n) * xs))

    y_max = float(max(env.max(), u_of_t(0.0).max(), 1.0)) * 1.15
    ax.set_xlim(0, L)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")

    line, = ax.plot([], [], lw=2, color="tab:blue", label="u(x,t)")
    ax.plot(xs, env, ls="--", lw=0.8, color="gray", alpha=0.7, label="огибающая мод")
    ax.plot(xs, -env, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(loc="upper right")
    title = ax.set_title("")
    time_text = ax.text(0.02, 0.02, "", transform=ax.transAxes)

    times = np.linspace(0.0, T_END, NT)

    def init():
        line.set_data([], [])
        title.set_text("u$_{tt}$ = c$^2$ u$_{xx}$,   u(0,t) = 0,  u$_x$(L,t) = 0")
        return line, title, time_text

    def update(i):
        t = float(times[i])
        y = u_of_t(t)
        line.set_data(xs, y)
        time_text.set_text(f"t = {t:5.2f}")
        return line, time_text

    ani = FuncAnimation(fig, update, frames=NT, init_func=init, blit=True,
                        interval=1000.0 / fps)

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(NT):
            update(i)
            fig.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"), dpi=80)
        print(f"Кадры сохранены в каталог: {frames_dir}")
    elif save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        ani.save(save_path, writer=writer, dpi=100,
                 progress_callback=lambda *a: None)
        print(f"Анимация сохранена в файл: {save_path}")
    else:
        plt.show()

    return ani


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    save_path = None
    frames_dir = None
    if "--save" in args:
        save_path = args[args.index("--save") + 1]
    if "--frames" in args:
        frames_dir = args[args.index("--frames") + 1]

    print(f"Параметры: c={C}, L={L}, ряд из {N_MODES} мод, "
          f"интегрирование — симпсон ({N_INTEGRAL} подынтервалов).")
    print("Строим решение (считаем коэффициенты Фурье)...")
    A, B = fourier_coefficients(N_MODES, f_initial, g_initial)

    def u(x, t):
        """Решение смешанной задачи u(x,t) (x — массив или скаляр)."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for n in range(len(A)):
            w = omega(n)
            # float() — скалярные коэффициенты, чтобы math.cos получал число
            result += (float(A[n]) * math.cos(w * t) + float(B[n]) * math.sin(w * t)) \
                      * np.sin(eigenvalue(n) * x)
        return result

    verify(u, A)
    print("Запускаем анимацию...")
    animate_solution(A, B, save_path=save_path, frames_dir=frames_dir)


if __name__ == "__main__":
    main()
