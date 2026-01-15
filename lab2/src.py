import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Pillow - опционально. Если не установлен или нет файла, создадим случайный массив.
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


# Часть 1. Изображение как массив

def load_image_or_random(path="image.jpeg", h=256, w=256):
    if PIL_OK and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return arr
    # fallback: "картинка" из случайных пикселей
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def show(img, title=""):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def task_image_ops(img):
    print("Форма изображения:", img.shape)  # (H, W, 3)

    # 2.1 Каналы
    red = img.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0

    green = img.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0

    blue = img.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0

    show(red, "Красный канал")
    show(green, "Зеленый канал")
    show(blue, "Синий канал")

    # 3.1 Grayscale (весовое среднее)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    gray = gray.astype(np.uint8)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")
    plt.show()

    # 4.1 Уменьшение в 2 раза по каждой оси (даунсэмплинг срезом)
    small = img[::2, ::2]
    show(small, "Уменьшено в 2 раза")

    # 5.1 Затирание прямоугольника
    erased = img.copy()
    # координаты прямоугольника (пример)
    y1, y2 = 60, 140
    x1, x2 = 80, 180
    erased[y1:y2, x1:x2] = [0, 0, 0]
    show(erased, "Затерли прямоугольник")

    # 6.1-6.3 Перевороты
    flip_h = np.flip(img, axis=1)   # горизонтально
    flip_v = np.flip(img, axis=0)   # вертикально
    rot180 = np.flip(img, axis=(0, 1))  # 180 градусов
    show(flip_h, "Горизонтальный переворот")
    show(flip_v, "Вертикальный переворот")
    show(rot180, "Поворот на 180")

    # 7.1 Broadcasting - "теплее" (прибавляем к красному каналу)
    filt = np.array([50, 0, 0], dtype=np.int16)          # (3,)
    warmer = img.astype(np.int16) + filt                  # расширение (broadcasting)
    warmer = np.clip(warmer, 0, 255).astype(np.uint8)     # ограничиваем 0..255
    show(warmer, "Теплее (R + 50)")

    # 8.1 Градиентная маска и смешивание с инверсией
    H, W = img.shape[:2]
    gradient = np.linspace(0, 1, H)[:, None, None]       # (H, 1, 1)
    inverted = 255 - img
    blended = (img * (1 - gradient) + inverted * gradient).astype(np.uint8)
    show(blended, "Переход: оригинал -> инверсия")


# Часть 2. kNN - Python vs NumPy

def make_data(n=1000, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, dim))
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


def knn_python(X_train, y_train, x_test, k=5):
    dists = []
    for i in range(len(X_train)):
        # евклидово расстояние
        s = 0.0
        for j in range(len(x_test)):
            s += (X_train[i][j] - x_test[j]) ** 2
        d = math.sqrt(s)
        dists.append((d, y_train[i]))

    dists.sort(key=lambda t: t[0])
    k_labels = [lab for _, lab in dists[:k]]
    # голосование
    return 1 if sum(k_labels) > (k / 2) else 0


def knn_numpy(X_train, y_train, x_test, k=5):
    dists = np.linalg.norm(X_train - x_test, axis=1)     # (n,)
    idx = np.argsort(dists)[:k]
    votes = y_train[idx]
    # если больше половины единиц - класс 1
    return int(votes.mean() > 0.5)


def time_call(fn, *args, repeats=1, **kwargs):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def knn_experiment():
    # базовые данные
    X, y = make_data(n=1000, dim=3, seed=1)
    x_test = np.array([0.2, -0.1, 0.05])

    print("\nkNN base (n=1000)")
    for k in [1, 3, 5, 11]:
        pred_py, t_py = time_call(knn_python, X, y, x_test, k=k, repeats=1)
        pred_np, t_np = time_call(knn_numpy, X, y, x_test, k=k, repeats=1)
        print("k =", k, "| python:", pred_py, "time:", round(t_py, 4), "sec",
              "| numpy:", pred_np, "time:", round(t_np, 4), "sec")

    # масштабирование
    print("\nScaling test (k=5):")
    for n in [1000, 10_000, 100_000]:
        Xs, ys = make_data(n=n, dim=3, seed=2)
        pred_np, t_np = time_call(knn_numpy, Xs, ys, x_test, k=5, repeats=1)
        # python для 100000 может быть очень медленно, поэтому ограничим
        if n <= 10_000:
            pred_py, t_py = time_call(knn_python, Xs, ys, x_test, k=5, repeats=1)
            print("n =", n, "| python:", round(t_py, 4), "sec | numpy:", round(t_np, 4), "sec")
        else:
            print("n =", n, "| python: (пропущено, слишком долго) | numpy:", round(t_np, 4), "sec")


def main():
    img = load_image_or_random("image.jpeg")
    task_image_ops(img)
    knn_experiment()


main()
