#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import array
import statistics
from collections import deque

from matplotlib import rc
rc('font', family='Droid Sans', weight='normal', size=14)
import matplotlib.pyplot as plt


class WikiGraph:
    def load_from_file(self, filename):
        print('Загружаю граф из файла: ' + filename)
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            (n, nlinks) = map(int, line.split())
            
            # Подготовка структур данных для хранения графа в формате CSR
            self._titles = []
            self._sizes = array.array('L', [0] * n)
            self._redirect = array.array('B', [0] * n)      # Флаги перенаправления (0 или 1)
            self._offset = array.array('L', [0] * (n + 1))
            self._links = array.array('L', [0] * nlinks)
            
            # Словарь для быстрого поиска ID статьи по её названию
            self._title_to_id = {}
            
            current_link_index = 0
            for i in range(n):
                title = f.readline().rstrip('\n')
                self._titles.append(title)
                self._title_to_id[title] = i

                size_line = f.readline().strip().split()
                size_bytes = int(size_line[0])
                redirect_flag = int(size_line[1])
                outdegree = int(size_line[2])
                
                self._sizes[i] = size_bytes
                self._redirect[i] = redirect_flag

                self._offset[i] = current_link_index
                
                for _ in range(outdegree):
                    link_id = int(f.readline().strip())
                    self._links[current_link_index] = link_id
                    current_link_index += 1

            # Последний элемент массива offset – граница массива ссылок для последней статьи
            self._offset[n] = current_link_index
        print('Граф загружен')

    def get_number_of_links_from(self, _id):
        """Возвращает количество исходящих ссылок из статьи с номером _id."""
        return self._offset[_id + 1] - self._offset[_id]

    def get_links_from(self, _id):
        """Возвращает список ID статей, на которые ссылается статья _id."""
        start = self._offset[_id]
        end = self._offset[_id + 1]
        return self._links[start:end]

    def get_id(self, title):
        """Возвращает ID статьи по её названию."""
        return self._title_to_id[title]

    def get_number_of_pages(self):
        """Возвращает общее количество статей в графе."""
        return len(self._titles)

    def is_redirect(self, _id):
        """Возвращает True, если статья _id является перенаправлением."""
        return self._redirect[_id] == 1

    def get_title(self, _id):
        """Возвращает название статьи по её ID."""
        return self._titles[_id]

    def get_page_size(self, _id):
        """Возвращает размер статьи в байтах по её ID."""
        return self._sizes[_id]


def hist(fname, data, bins, xlabel, ylabel, title, facecolor='green', alpha=0.5, transparent=True, **kwargs):
    """Строит гистограмму и сохраняет её в файл."""
    plt.clf()
    plt.hist(data, bins=bins, facecolor=facecolor, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(fname, transparent=transparent, **kwargs)


def bfs_path(graph, start_title, target_title):
    """Ищет путь от статьи start_title до target_title с помощью BFS."""
    try:
        start = graph.get_id(start_title)
        target = graph.get_id(target_title)
    except KeyError:
        print("Одна из указанных статей не найдена")
        return None

    queue = deque([start])
    predecessor = {start: None}
    
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for neighbor in graph.get_links_from(current):
            if neighbor not in predecessor:
                predecessor[neighbor] = current
                queue.append(neighbor)
    else:
        # Если очередь опустела и цель не найдена
        return None

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    return path


if __name__ == '__main__':
    filename = input("Введите имя файла с графом статей: ") # Сделано сразу для упражнения 4
    if os.path.isfile(filename):
        wg = WikiGraph()
        wg.load_from_file(filename)
    else:
        print('Файл с графом не найден')
        sys.exit(-1)

    # Поиск пути от статьи 1 до статьи 2. Сделано сразу для упражнения 4
    article1 = input("Введите название статьи 1: ")
    article2 = input("Введите название статьи 2: ")
    print("Запускаем поиск в ширину")
    path = bfs_path(wg, article1, article2)
    if path is not None:
        print("Поиск закончен. Найден путь:")
        for node in path:
            print(wg.get_title(node))
    else:
        print("Путь не найден")

    n_pages = wg.get_number_of_pages()

    # Статистика по исходящим ссылкам (без редиректов)
    outdegrees = [wg.get_number_of_links_from(i) for i in range(n_pages)]
    min_out = min(outdegrees)
    max_out = max(outdegrees)
    count_min_out = outdegrees.count(min_out)
    count_max_out = outdegrees.count(max_out)
    id_max_out = outdegrees.index(max_out)
    avg_out = statistics.mean(outdegrees)
    stdev_out = statistics.stdev(outdegrees) if n_pages > 1 else 0

    # Количество статей с перенаправлением
    count_redirect = sum(1 for i in range(n_pages) if wg.is_redirect(i))
    perc_redirect = (count_redirect / n_pages) * 100

    # Статистика по входящим ссылкам
    incoming_external = [0] * n_pages
    incoming_redirect = [0] * n_pages
    for j in range(n_pages):
        for target_id in wg.get_links_from(j):
            if wg.is_redirect(j):
                incoming_redirect[target_id] += 1
            else:
                incoming_external[target_id] += 1

    # Статистика для внешних входящих ссылок (перенаправления не учитываются)
    min_in_ext = min(incoming_external)
    max_in_ext = max(incoming_external)
    count_min_in_ext = incoming_external.count(min_in_ext)
    count_max_in_ext = incoming_external.count(max_in_ext)
    id_max_in_ext = incoming_external.index(max_in_ext)
    avg_in_ext = statistics.mean(incoming_external)
    stdev_in_ext = statistics.stdev(incoming_external) if n_pages > 1 else 0

    # Статистика для входящих ссылок от перенаправлений
    min_in_redir = min(incoming_redirect)
    max_in_redir = max(incoming_redirect)
    count_min_in_redir = incoming_redirect.count(min_in_redir)
    count_max_in_redir = incoming_redirect.count(max_in_redir)
    id_max_in_redir = incoming_redirect.index(max_in_redir)
    avg_in_redir = statistics.mean(incoming_redirect)
    stdev_in_redir = statistics.stdev(incoming_redirect) if n_pages > 1 else 0

    # Вывод результатов
    print(f"Количество статей с перенаправлением: {count_redirect} ({perc_redirect:.2f}%)")
    print(f"Минимальное количество ссылок из статьи: {min_out}")
    print(f"Количество статей с минимальным количеством ссылок: {count_min_out}")
    print(f"Максимальное количество ссылок из статьи: {max_out}")
    print(f"Количество статей с максимальным количеством ссылок: {count_max_out}")
    print(f"Статья с наибольшим количеством ссылок: {wg.get_title(id_max_out)}")
    print(f"Среднее количество ссылок в статье: {avg_out:.2f} (ср. откл. {stdev_out:.2f})")
    print(f"Минимальное количество ссылок на статью: {min_in_ext}")
    print(f"Количество статей с минимальным количеством внешних ссылок: {count_min_in_ext}")
    print(f"Максимальное количество ссылок на статью: {max_in_ext}")
    print(f"Количество статей с максимальным количеством внешних ссылок: {count_max_in_ext}")
    print(f"Статья с наибольшим количеством внешних ссылок: {wg.get_title(id_max_in_ext)}")
    print(f"Среднее количество внешних ссылок на статью: {avg_in_ext:.2f} (ср. откл. {stdev_in_ext:.2f})")
    print(f"Минимальное количество перенаправлений на статью: {min_in_redir}")
    print(f"Количество статей с минимальным количеством внешних перенаправлений: {count_min_in_redir}")
    print(f"Максимальное количество перенаправлений на статью: {max_in_redir}")
    print(f"Количество статей с максимальным количеством внешних перенаправлений: {count_max_in_redir}")
    print(f"Статья с наибольшим количеством внешних перенаправлений: {wg.get_title(id_max_in_redir)}")
    print(f"Среднее количество внешних перенаправлений на статью: {avg_in_redir:.2f} (ср. откл. {stdev_in_redir:.2f})")
