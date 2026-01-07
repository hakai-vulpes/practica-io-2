"""
Este archivo se ocupa de dónde está el coche y las situaciones que desencadena
al hacer sus acciones.

Raliza las siguientes tareas:

- Convertir el mapa en una matriz NumPy.
- Según la acción tomada por el modelo, relizarla.
- Comprobar qué ha sucedido (choque, meta o movimiento válido)


Toma en consideración una validación de trayectoria: No basta con mirar el punto
inicial y final de un movimiento, hay que mirar también los puntos intermedios,
ya que el coche da "saltos", así aseguramos que no haya atravesado alguna pared
delgada.
"""

import numpy as np

class RaceTrackEnv:
    """
    Entorno del coche.
    
    Representa una pista en una cuadrícula discreta.
    - Estado: (x, y, vx, vy)
    - Acciones: 9 posibles combinaciones de incrementos de velocidad (-1, 0, +1).
    - Objetivo: Ir de la Salida ('S') a la Meta ('F') en el menor tiempo posible.
    """

    def __init__(self, track_str):
        """
        Inicializa el entorno procesando el mapa de texto.
        
        Args:
            track_str (list of str): Lista de strings que dibujan el mapa.
                                     'S'=Salida, 'F'=Meta, '#'=Muro, '.'=Pista.
        """
        self.track = self._parse_track(track_str)
        self.height, self.width = self.track.shape
        
        # Generamos las 9 acciones posibles
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        self.state = None # Se inicializa en reset()

    def _parse_track(self, track_str):
        """Convierte el mapa visual de caracteres a una matriz numérica de Numpy."""
        mapping = {'#': 1, '.': 0, 'S': 2, 'F': 3}
        track_list = []
        for line in track_str:
            row = [mapping[char] for char in line]
            track_list.append(row)
        return np.array(track_list)

    def reset(self):
        """
        Reinicia el coche en una posición aleatoria de la línea de salida.
        
        Returns:
            tuple: Estado inicial (x, y, vx=0, vy=0)
        """
        # Buscamos todas las coordenadas 'S' (Salida)
        start_y, start_x = np.where(self.track == 2)
        idx = np.random.randint(len(start_x))
        
        # Estado: (x, y, vx, vy)
        self.state = (int(start_x[idx]), int(start_y[idx]), 0, 0)
        return self.state

    def step(self, action_idx):
        """
        Ejecuta un paso de simulación.
        
        Args:
            action_idx (int): Índice de la acción a tomar (0-8).
            
        Returns:
            tuple: (next_state, reward, done)
        """
        ax, ay = self.actions[action_idx]
        x, y, vx, vy = self.state

        # 1. Actualizar Velocidad (Limitada entre 0 y 4)
        # Nota: Un diseño más universal permitiría velocidades negativas,
        # pero para este circuito específico se asume movimiento positivo.
        new_vx = int(np.clip(vx + ax, 0, 4))
        new_vy = int(np.clip(vy + ay, 0, 4))

        # Evitar parada total si no estamos en salida (opcional, pero ayuda a no atascarse)
        if new_vx == 0 and new_vy == 0:
            # En la práctica, si el agente aprende a parar, pierde tiempo.
            pass

        # 2. Calcular nueva posición teórica
        # EJE X: Positivo hacia la derecha
        new_x = x + new_vx
        # EJE Y: En matrices, fila 0 es ARRIBA. Para ir "arriba" restamos índice.
        new_y = y - new_vy 

        # 3. Comprobar colisiones en la trayectoria proyectada
        collision = self._check_path(x, y, new_x, new_y)

        reward = -1 # Penalización constante por paso de tiempo
        done = False

        if collision == 'wall':
            # CHOQUE: vuelta a la salida, el episodio NO termina
            self.reset()
            
        elif collision == 'finish':
            # META: El episodio termina con éxito
            self.state = (int(new_x), int(new_y), new_vx, new_vy)
            done = True
            reward = 0 # Recompensa final (o dejar de restar -1)
            
        else:
            # MOVIMIENTO VÁLIDO: Actualizamos posición
            self.state = (int(new_x), int(new_y), new_vx, new_vy)

        return self.state, reward, done

    def _check_path(self, x0, y0, x1, y1):
        """
        Verifica si la línea recta entre (x0,y0) y (x1,y1) cruza un muro o la meta.
        Usa interpolación lineal para simular la trayectoria continua.
        """
        points_num = max(abs(x1 - x0), abs(y1 - y0)) + 1
        
        for i in range(1, points_num + 1): 
            t = i / points_num
            # Interpolación redondeada al entero más cercano
            xi = int(round(x0 + t * (x1 - x0)))
            yi = int(round(y0 + t * (y1 - y0)))

            # Chequeo de límites del array
            if not (0 <= yi < self.height and 0 <= xi < self.width):
                return 'wall'

            cell = self.track[yi, xi]
            
            if cell == 1:  # '#' Muro
                return 'wall'
            elif cell == 3:  # 'F' Meta
                return 'finish'
        
        return None