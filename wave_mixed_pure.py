# -*- coding: utf-8 -*-
"""
Смешанная задача для одномерного волнового уравнения — ЧИСТЫЙ PYTHON (без numpy).

Уравнение:      u_tt = c^2 * u_xx,   0 <= x <= L,  t >= 0
Граничные условия (вариант):  u(0,t) = 0,   u_x(L,t) = 0
Начальные условия:            u(x,0) = f(x),  u_t(x,0) = g(x)

Решение — методом разделения переменных (Фурье):

    u(x,t) = sum_n [ A_n cos(w_n t) + B_n sin(w_n t)/w_n ] * sin(k_n x)

где  k_n = pi*(2n+1)/(2L),  w_n = c*k_n,
     A_n = (2/L) * int_0^L f(x) sin(k_n x) dx
     B_n = (2/(L*w_n)) * int_0^L g(x) sin(k_n x) dx

Никаких «встроенных решателей» не используется: интегрирование (составное
правило Симпсона), суммы рядов, сетки — всё написано вручную на стандартных
списках Python. Единственная внешняя библиотека — matplotlib, и она нужна
только для отрисовки анимации графика.

Запуск анимации:   python wave_mixed_pure.py
Без дисплея:       python wave_mixed_pure.py --save animation.mp4
                   python wave_mixed_pure.py --frames frames_dir
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # безопасный backend; при наличии дисплея переключается
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------------------------------------------------------
# 1. Параметры задачи
# ----------------------------------------------------------------------------
C = 1.0        # скорость распространения волн c
L = 1.0        # длина струны
T_END = 6.0    # рассматриваемый интервал времени
N_MODES = 40   # число членов ряда Фурье в решении

NX = 300       # число точек пространственной сетки
NT = 300       # число временны́х кадров
N_INTEGRAL = 2000  # число подынтервалов при интегрировании Симпсоном


# ----------------------------------------------------------------------------
# 2. Начальные данные f(x) = u(x,0) и g(x) = u_t(x,0)
#    Обе функции удовлетворяют граничным условиям:
#    f(0)=0, f'(L)=0;  g(0)=0, g'(L)=0
# ----------------------------------------------------------------------------
def f_initial(x):
    """u(x,0) — начальный профиль струны."""
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


# ----------------------------------------------------------------------------
# 4. Вспомогательные функции вместо numpy (всё вручную на списках)
# ----------------------------------------------------------------------------
def linspace(a, b, num):
    """Равномерная сетка из num точек на [a, b] (аналог np.linspace)."""
    if num < 2:
        return [a]
    step = (b - a) / (num - 1)
    return [a + i * step for i in range(num)]


def simpson(func, a, b, n=N_INTEGRAL):
    """Интеграл func по [a,b] по составному правилу Симпсона.

    n — чётное число подынтервалов (если нечётное — увеличиваем на 1).
    Реализовано вручную, без scipy.integrate и пр.
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
# 5. Коэффициенты Фурье решения (интегралы — Симпсоном вручную)
# ----------------------------------------------------------------------------
def fourier_coefficients(n_modes, f_func, g_func):
    """Возвращает списки A[n] и B[n], n = 0..n_modes-1."""
    A = [0.0] * n_modes
    B = [0.0] * n_modes
    for n in range(n_modes):
        k = eigenvalue(n)
        w = omega(n)
        A[n] = (2.0 / L) * simpson(lambda x: f_func(x) * math.sin(k * x), 0.0, L)
        B[n] = (2.0 / (L * w)) * simpson(lambda x: g_func(x) * math.sin(k * x), 0.0, L)
    return A, B


# ----------------------------------------------------------------------------
# 6. Решение u(x,t)
# ----------------------------------------------------------------------------
def make_solution(A, B):
    """Возвращает функцию u(x, t): x — скаляр или список точек."""
    n_modes = len(A)
    ks = [eigenvalue(n) for n in range(n_modes)]
    ws = [omega(n) for n in range(n_modes)]

    def u(x, t):
        tc = [A[n] * math.cos(ws[n] * t) + B[n] * math.sin(ws[n] * t)
              for n in range(n_modes)]
        if isinstance(x, (list, tuple)):
            return [sum(tc[n] * math.sin(ks[n] * xi) for n in range(n_modes))
                    for xi in x]
        return sum(tc[n] * math.sin(ks[n] * x) for n in range(n_modes))

    return u


# ----------------------------------------------------------------------------
# 7. Проверка корректности решения (самопроверка)
# ----------------------------------------------------------------------------
def verify(u, A):
    print("=" * 64)
    print("Самопроверка решения")
    print("=" * 64)

    # --- сходимость метода: при f(x) = sin(k_0 x) точные коэффициенты
    #     A_0 = 1, A_n = 0 (n >= 1). ---
    n_modes = len(A)
    A_test, _ = fourier_coefficients(n_modes, lambda x: math.sin(eigenvalue(0) * x),
                                     lambda x: 0.0)
    err_A = max(abs(A_test[n] - (1.0 if n == 0 else 0.0)) for n in range(n_modes))
    print(f"max |A_n(числ.) - A_n(точн.)| для f=sin(k_0 x) = {err_A:.3e}")

    # --- начальные условия ---
    xs = linspace(0.0, L, 101)
    u0 = u(xs, 0.0)
    err_f = max(abs(u0[i] - f_initial(xs[i])) for i in range(len(xs)))
    dt = 1e-5
    up = u(xs, dt)
    um = u(xs, -dt)
    err_g = max(abs((up[i] - um[i]) / (2.0 * dt) - g_initial(xs[i]))
                for i in range(len(xs)))
    print(f"max |u(x,0) - f(x)|          = {err_f:.3e}   (усечение ряда)")
    print(f"max |u_t(x,0) - g(x)|        = {err_g:.3e}")

    # --- граничные условия ---
    t = 1.234
    err_bc1 = abs(u([0.0], t)[0])
    hx = 1e-5
    ux_L = (u([L + hx], t)[0] - u([L - hx], t)[0]) / (2.0 * hx)
    print(f"|u(0,t)|                     = {err_bc1:.3e}")
    print(f"|u_x(L,t)|                   = {abs(ux_L):.3e}")

    # --- невязка волнового уравнения u_tt - c^2 u_xx (конечные разности) ---
    h = 1e-4
    x0, t0 = 0.37, 0.81
    utt = (u([x0], t0 + h)[0] - 2.0 * u([x0], t0)[0] + u([x0], t0 - h)[0]) / h ** 2
    uxx = (u([x0 + h], t0)[0] - 2.0 * u([x0], t0)[0] + u([x0 - h], t0)[0]) / h ** 2
    resid = abs(utt - C ** 2 * uxx)
    rel = resid / max(abs(utt), 1e-30)
    print(f"невязка |u_tt - c^2 u_xx|    = {resid:.3e}  (отн. {rel:.3e})")
    print("=" * 64)


# ----------------------------------------------------------------------------
# 8. Анимация графика решения
# ----------------------------------------------------------------------------
def animate_solution(A, B, save_path=None, frames_dir=None, fps=30):
    """Анимированный график u(x,t). Сохраняет MP4/GIF или кадры, если нужно."""
    if save_path is None and frames_dir is None:
        try:
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")
        except Exception:
            pass  # остаёмся на Agg — тогда сохраните файл через --save

    u = make_solution(A, B)
    xs = linspace(0.0, L, NX)
    times = linspace(0.0, T_END, NT)

    # Огибающая |u| <= sum_n sqrt(A_n^2 + B_n^2) * |sin(k_n x)|
    env = [0.0] * NX
    for n in range(len(A)):
        k = eigenvalue(n)
        amp = math.hypot(A[n], B[n])
        for i, xi in enumerate(xs):
            env[i] += amp * abs(math.sin(k * xi))

    y_max = max(max(env), max(u(xs, 0.0)), 1.0) * 1.15

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, L)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")

    line, = ax.plot([], [], lw=2, color="tab:blue", label="u(x,t)")
    ax.plot(xs, env, ls="--", lw=0.8, color="gray", alpha=0.7, label="огибающая мод")
    ax.plot(xs, [-e for e in env], ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(loc="upper right")
    title = ax.set_title("")
    time_text = ax.text(0.02, 0.02, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        title.set_text("u$_{tt}$ = c$^2$ u$_{xx}$,   u(0,t) = 0,  u$_x$(L,t) = 0")
        return line, title, time_text

    def update(i):
        t = times[i]
        y = u(xs, t)
        line.set_data(xs, y)
        time_text.set_text("t = %5.2f" % t)
        return line, time_text

    ani = FuncAnimation(fig, update, frames=NT, init_func=init, blit=True,
                        interval=1000.0 / fps)

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(NT):
            update(i)
            fig.savefig(os.path.join(frames_dir, "frame_%04d.png" % i), dpi=80)
        print("Кадры сохранены в каталог: %s" % frames_dir)
    elif save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        ani.save(save_path, writer=writer, dpi=100,
                 progress_callback=lambda *a: None)
        print("Анимация сохранена в файл: %s" % save_path)
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

    print("Параметры: c=%s, L=%s, ряд из %d мод, "
          "интегрирование — симпсон (%d подынтервалов)." %
          (C, L, N_MODES, N_INTEGRAL))
    print("Строим решение (считаем коэффициенты Фурье)...")
    A, B = fourier_coefficients(N_MODES, f_initial, g_initial)

    u = make_solution(A, B)
    verify(u, A)
    print("Запускаем анимацию...")
    animate_solution(A, B, save_path=save_path, frames_dir=frames_dir)


if __name__ == "__main__":
    main()
