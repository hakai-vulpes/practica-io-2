"""
Este es el archivo cerebro.

Según el estado que reciba, devuelve una acción calculada por el modelo.
"""


import numpy as np

class MonteCarloAgent:
    """
    Agente que utiliza Control Monte Carlo (On-Policy / Epsilon-Greedy) 
    para aprender la política óptima.
    """

    def __init__(self, env, gamma=0.9):
        """
        Inicializa el agente y sus tablas de conocimiento.

        Args:
            env: El entorno (necesario para conocer dimensiones y acciones).
            gamma (float): Factor de descuento (0 a 1). Cuánto valora el futuro.
                           0.9 es estándar: valora el futuro pero prioriza lo inmediato.
        """
        self.env = env
        self.gamma = gamma
        
        # Tabla Q (Action-Value Function):
        # Guarda el valor estimado de tomar una acción 'a' en un estado 's'.
        # Dimensiones: (Alto, Ancho, Vel_X, Vel_Y, Num_Acciones)
        # Vel_X y Vel_Y tienen tamaño 5 (velocidades 0,1,2,3,4).
        self.q_table = np.zeros((env.height, env.width, 5, 5, len(env.actions)))
        
        # Tabla C (Conteo):
        # Guarda cuántas veces hemos visitado cada par (estado, acción).
        # Se usa para calcular la media incremental de forma eficiente.
        self.c_table = np.zeros((env.height, env.width, 5, 5, len(env.actions)))

    def get_action(self, state, epsilon=0.1):
        """
        Selecciona una acción usando una política Epsilon-Greedy.
        
        Args:
            state (tuple): El estado actual (x, y, vx, vy).
            epsilon (float): Probabilidad de exploración (0.0 a 1.0).
            
        Returns:
            int: El índice de la acción seleccionada.
        """
        x, y, vx, vy = state
        
        # EXPLORACIÓN: Con probabilidad epsilon, elegimos al azar.
        # Esto permite descubrir nuevas rutas que quizás sean mejores.
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.env.actions))
        
        # EXPLOTACIÓN: Elegimos la mejor acción conocida (Greedy).
        else:
            # Miramos los valores Q para este estado concreto
            state_q_values = self.q_table[y, x, vx, vy, :]
            
            # Buscamos el valor máximo
            max_val = np.max(state_q_values)
            
            # Identificamos TODAS las acciones que tienen ese valor máximo (por si hay empate)
            # np.where devuelve una tupla de arrays, tomamos el primero [0]
            best_actions = np.where(state_q_values == max_val)[0]
            
            # Si hay empate (común al inicio cuando todo es 0), elegimos una al azar.
            return np.random.choice(best_actions)

    def update(self, episode):
        """
        Actualiza la Tabla Q usando el método Monte Carlo al final de un episodio.
        
        Args:
            episode (list): Lista de tuplas (estado, acción, recompensa).
        """
        G = 0 # Retorno acumulado (Ganancia)
        visited_sa = set() # Conjunto para controlar "First-Visit" (opcional, pero eficiente)

        # Recorremos el episodio HACIA ATRÁS (desde la meta hacia la salida)
        # Esto es clave: propagamos la recompensa final hacia las decisiones previas.
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            x, y, vx, vy = state
            
            # Fórmula del Retorno: G = Recompensa + gamma * G_futuro
            G = self.gamma * G + reward
            
            # Identificador único del par Estado-Acción
            sa_pair = (x, y, vx, vy, action)

            # Lógica First-Visit MC: 
            # Si pasamos por el mismo estado 2 veces en un episodio, solemos contar 
            # solo la primera para evitar sesgos de ciclos, aunque Every-Visit también vale.
            if sa_pair not in visited_sa:
                visited_sa.add(sa_pair)
                
                # Incrementamos el contador de visitas (N)
                self.c_table[y, x, vx, vy, action] += 1
                N = self.c_table[y, x, vx, vy, action]
                
                # Actualizamos el valor Q (Media Incremental)
                # Q_nuevo = Q_viejo + (1/N) * (G - Q_viejo)
                old_q = self.q_table[y, x, vx, vy, action]
                self.q_table[y, x, vx, vy, action] = old_q + (1.0 / N) * (G - old_q)