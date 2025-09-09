import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, dim=5, goal=0, start=24, alfa=0.1, gamma=0.8, epsilon=0.2, episodes=500):
        self.dim = dim           # Розмір сітки (dim x dim)
        self.goal = goal         # Цільова клітинка (індекс)
        self.start = start       # Початкова клітинка (індекс)
        self.alfa = alfa         # Швидкість навчання (learning rate)
        self.gamma = gamma       # Коефіцієнт дисконтної винагороди
        self.epsilon = epsilon   # Імовірність вибору випадкової дії (exploration)
        self.episodes = episodes # Кількість епізодів тренування

        # Побудова матриці винагород (R) та ініціалізація Q-матриці
        self.R = self._build_R()             
        self.Q = np.zeros_like(self.R, dtype=float)  # Матриця оцінок Q, спочатку нулі

    def _state2coord(self, s):
        return divmod(s, self.dim)

    def _coord2state(self, i, j):
        return i * self.dim + j

    def _available_actions(self, state):
        # Повертає список допустимих переходів (дій) із заданого стану
        i, j = self._state2coord(state)
        actions = []
        # Перевіряємо 4 напрямки: вгору, вниз, вліво, вправо
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            # Якщо координати в межах сітки, додаємо дію
            if 0 <= ni < self.dim and 0 <= nj < self.dim:
                actions.append(self._coord2state(ni, nj))
        return actions

    def _build_R(self):
        # Створення матриці винагород R
        size = self.dim * self.dim
        R = np.full((size, size), -1)  # Спочатку всі переходи -1 (негативна винагорода)
        for s in range(size):
            for a in self._available_actions(s):
                R[s, a] = -1              # Дозволені переходи отримують -1 (рух)
        # Встановлюємо позитивну винагороду за перехід у цільову клітинку
        for a in self._available_actions(self.goal):
            R[a, self.goal] = 100
        return R

    def train(self):
        for _ in range(self.episodes):
            # Випадковий початковий стан
            state = np.random.randint(0, self.dim * self.dim)
            while state == self.goal:
                state = np.random.randint(0, self.dim * self.dim)

            # Поки не досягнемо цілі, робимо кроки
            while state != self.goal:
                actions = self._available_actions(state)  # Доступні дії

                # Вибір дії: випадкова з ймовірністю epsilon (exploration)
                if np.random.rand() < self.epsilon:
                    next_state = np.random.choice(actions)
                else:
                    # Вибір кращої дії згідно з Q (exploitation)
                    qs = [self.Q[state, a] for a in actions]
                    max_q = max(qs)
                    best = [a for a, q in zip(actions, qs) if q == max_q]
                    next_state = np.random.choice(best)

                # Отримуємо винагороду за перехід
                reward = self.R[state, next_state]
                old_q = self.Q[state, next_state]

                # Оновлення Q-значення за формулою
                self.Q[state, next_state] = old_q + self.alfa * (
                    reward + self.gamma * np.max(self.Q[next_state]) - old_q)

                # Переходимо до наступного стану
                state = next_state

    def get_optimal_path(self):
        # Відновлення оптимального шляху із Q-матриці
        state = self.start
        path = [state]
        visited = set()

        while state != self.goal:
            actions = self._available_actions(state)
            qs = [self.Q[state, a] for a in actions]
            max_q = max(qs)
            best = [a for a, q in zip(actions, qs) if q == max_q]
            next_state = np.random.choice(best)

            path.append(next_state)

            if next_state in visited:
                break
            visited.add(next_state)

            state = next_state

            if len(path) > 100:
                break
        return path

    def visualize_path(self, path):
        fig, ax = plt.subplots(figsize=(6, 6))

        for idx in range(len(path)):
            ax.clear()

            for i in range(self.dim + 1):
                ax.plot([-0.5, self.dim - 0.5], [i - 0.5, i - 0.5], color='gray')  
                ax.plot([i - 0.5, i - 0.5], [-0.5, self.dim - 0.5], color='gray')  

            ax.set_xlim(-0.5, self.dim - 0.5)
            ax.set_ylim(self.dim - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.axis('off')  
            gi, gj = self._state2coord(self.goal)
            ax.add_patch(plt.Rectangle((gj - 0.5, gi - 0.5), 1, 1, color='lightgreen'))

            si, sj = self._state2coord(self.start)
            ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, color='lightblue'))

            for s in path[:idx + 1]:
                r, c = self._state2coord(s)
                ax.plot(c, r, 'o', color='red')

            ci, cj = self._state2coord(path[idx])
            ax.plot(cj, ci, 'o', color='black', markersize=15)

            ax.set_title(f"Крок {idx} / {len(path) - 1}")

            plt.pause(0.4)

        plt.ioff()
        plt.show()


agent = QLearningAgent()
agent.train()
optimal_path = agent.get_optimal_path()
agent.visualize_path(optimal_path)
