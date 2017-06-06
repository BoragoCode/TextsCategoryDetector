# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np
import scipy.spatial


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """

    X = X.toarray()
    X_train = X_train.toarray()
    #
    # result = np.zeros((X.shape[0], X_train.shape[0]))
    #
    # for i in range(X.shape[0]):
    #     for j in range(X_train.shape[0]):
    #         dist = 0
    #
    #         for d in range(X.shape[1]):
    #             if X[i, d] != X_train[j, d]:
    #                 dist += 1
    #
    #         result[i, j] = dist
    #
    # return result

    return scipy.spatial.distance.cdist(X, X_train, metric='hamming') * X.shape[1]


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """

    # Dist - Odległości między danymi treningowymi a nowymi danymi
    # y - nazwy kategorii w danych treningowych


    return y[Dist.argsort(kind='mergesort')]


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszych sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    M = 4
    N1 = y.shape[0]
    result = np.zeros((N1, M + 1))
    deleted = np.delete(y, range(k, y.shape[1]), axis=1)  # usuwamy kolumny z niepotrzebnymi sąsiadami
    print(deleted)
    for i in range(N1):
        result[i] = np.bincount(deleted[i], minlength=M + 1)  # bincount zlicza ile razy występuje jaka kategoria

    return np.divide(np.delete(result, 0, axis=1), k)


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    errors = 0
    N = y_true.shape[0]

    for i in range(N):
        rev = p_y_x[i][::-1]  # Odwracamy, żeby wziąć ostatnie maksymalne prawdopodobieństwo

        if y_true[i] != (len(rev) - np.argmax(rev)):
            errors += 1

    return errors / N


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    errors = np.zeros(len(k_values))

    dist = hamming_distance(Xval, Xtrain)
    dist_sort = sort_train_labels_knn(dist, ytrain)

    for i in range(len(k_values)):
        pyx = p_y_x_knn(dist_sort, k_values[i])
        errors[i] = classification_error(pyx, yval)

    index = np.argmin(errors)

    best_error = errors[index]
    best_k = k_values[index]

    return best_error, best_k, errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """

    # Etykiety - ciąg kolejnych wystąpień poszczególnych kategorii
    # Trzeba obliczyć ile razy która kategoria występuje

    categories = 4
    result = np.zeros(categories)

    for i in range(1, categories + 1):
        for j in range(ytrain.shape[0]):
            if ytrain[j] == i:
                result[i - 1] += 1

    return result / ytrain.shape[0]


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    # Wynik - theta
    Xtrain = Xtrain.toarray()

    M = 4
    D = Xtrain.shape[1]
    result = np.zeros((M, D))

    for k in range(1, M + 1):
        y_equal_k = np.equal(ytrain, k)  # true - dokument jest danej kategorii
        denominator = np.sum(y_equal_k) + a + b - 2  # liczba dokumentów danej kategorii

        for d in range(D):
            and_in_doc = np.bitwise_and(y_equal_k, Xtrain[:, d])  # true - dokument danej kategorii i zawiera dany wyraz
            numerator = np.sum(and_in_doc) + a - 1  # liczba wyrazów spełniających w/w warunek + hiperparametry

            result[k - 1, d] = np.divide(numerator, denominator)

    return result


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """

    # Prawdopodobieństwo, że dany dokument jest w danej kategorii
    X = X.toarray()
    M = p_x_1_y.shape[0]
    N = X.shape[0]
    p_x_l_y_neg = np.ones(p_x_1_y.shape) - p_x_1_y

    result = np.zeros((N, M))

    for n in range(N):
        for k in range(M):
            result[n, k] = p_y[k] * np.prod(np.where(X[n], p_x_1_y[k], p_x_l_y_neg[k]))

        result[n, :] /= np.sum(result[n, :])

    return result


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    errors = np.zeros((len(a_values), len(b_values)))
    p_y = estimate_a_priori_nb(ytrain)

    for a_index in range(len(a_values)):
        for b_index in range(len(b_values)):
            p_x_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a_index], b_values[b_index])
            p_y_x = p_y_x_nb(p_y, p_x_y, Xval)
            errors[a_index, b_index] = classification_error(p_y_x, yval)

    index = np.unravel_index(errors.argmin(), errors.shape)

    best_error = errors[index]
    best_a = a_values[index[0]]
    best_b = b_values[index[1]]

    return best_error, best_a, best_b, errors
