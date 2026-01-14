"""
Modularización del entrenamiento y la visualización de los resultados.
"""



import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def train_agent(env, agent, num_episodes=10000, max_steps=2000, snapshot_interval=500):
    """
    Ejecuta el bucle de entrenamiento completo.
    
    Args:
        env: El entorno de carrera.
        agent: El agente Monte Carlo.
        num_episodes (int): Total de episodios.
        max_steps (int): Límite de pasos por episodio para evitar bucles infinitos.
        snapshot_interval (int): Cada cuántos episodios guardamos datos para el vídeo.
        
    Returns:
        list: evolution_history (datos para generar el GIF).
    """
    print(f"--- Iniciando entrenamiento en mapa {env.width}x{env.height} ({num_episodes} eps) ---")
    start_time = time.time()
    
    evolution_history = []
    epsilon = 0.1
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        # ¿Grabamos vídeo? (Episodio 1 o múltiplos del intervalo)
        record_video = (i == 0 or (i + 1) % snapshot_interval == 0)
        
        # Estructuras temporales para grabación
        current_segments = []       
        current_segment_path = []   
        if record_video:
            current_segment_path.append((state[0], state[1]))
        
        # Decaimiento lineal de epsilon
        current_epsilon = max(0.01, epsilon * (1 - i/num_episodes)) 
        episode = [] 

        while not done and steps < max_steps:
            # --- RAMA A: GRABACIÓN (Lenta, calcula física para líneas grises) ---
            if record_video:
                x, y, vx, vy = state
                action_idx = agent.get_action(state, current_epsilon)
                
                # Predicción física para detectar choques/resets
                ax_val, ay_val = env.actions[action_idx]
                pred_vx = int(np.clip(vx + ax_val, 0, 4))
                pred_vy = int(np.clip(vy + ay_val, 0, 4))
                pred_x = x + pred_vx
                pred_y = y - pred_vy
                
                next_state, reward, done = env.step(action_idx)
                
                has_reset = (next_state[0] != pred_x) or (next_state[1] != pred_y)
                
                if has_reset:
                    current_segment_path.append((pred_x, pred_y)) 
                    current_segments.append(current_segment_path)
                    current_segment_path = [(next_state[0], next_state[1])] 
                else:
                    current_segment_path.append((next_state[0], next_state[1]))

            # --- RAMA B: VELOCIDAD MÁXIMA (Normal) ---
            else:
                action_idx = agent.get_action(state, current_epsilon)
                next_state, reward, done = env.step(action_idx)

            episode.append((state, action_idx, reward))
            state = next_state
            steps += 1

        # Cierre de grabación
        if record_video:
            current_segments.append(current_segment_path)
            evolution_history.append({
                'episode': i + 1,
                'segments': current_segments,
                'result': 'Meta' if done else 'Tiempo'
            })

        if done:
            agent.update(episode)
        
        # Log de progreso
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"   > Progreso: {(i+1)/num_episodes:.0%} | Tiempo: {elapsed:.0f}s")
            
    print(f"--- Entrenamiento finalizado en {time.time() - start_time:.1f}s ---\n")
    return evolution_history



def show_track_layouts(env_list, title_list=None):
    """
    Visualiza múltiples circuitos en una sola fila de subgráficos.
    
    Args:
        env_list (list): Lista de objetos RaceTrackEnv.
        title_list (list): Lista de títulos (strings). Si es None, usa "Mapa 1", "Mapa 2"...
    """
    n_maps = len(env_list)
    
    # Si no hay títulos, generamos genéricos
    if title_list is None:
        title_list = [f"Mapa {i+1}" for i in range(n_maps)]
        
    # Crear la figura con N columnas (una por mapa)
    # Ajustamos el ancho (figsize) dinámicamente: 6 pulgadas por cada mapa
    fig, axes = plt.subplots(1, n_maps, figsize=(6 * n_maps, 10))
    
    # Caso especial: Si solo hay 1 mapa, 'axes' no es una lista, es un objeto único.
    # Lo convertimos en lista para que el bucle funcione igual.
    if n_maps == 1:
        axes = [axes]
        
    # Iteramos sobre los entornos, los títulos y los ejes a la vez
    for env, title, ax in zip(env_list, title_list, axes):
        ax.imshow(env.track)
        ax.set_title(f"{title}\n(Verde=S, Amarillo=F)", fontsize=12)
        ax.axis('off') # Quitar ejes
        
    plt.tight_layout() # Ajusta automáticamente los espacios para que no se solapen
    plt.show()



def create_animation(env, data_subset, filename, target_duration_sec=10, title_prefix=""):
    """
    Genera un GIF con la evolución del aprendizaje sobre el mapa del entorno dado.
    """
    if not data_subset:
        print(f"⚠️ No hay datos para generar {filename}")
        return

    # 1. Cálculo de frames
    total_raw_steps = 0
    for ep_data in data_subset:
        for segment in ep_data['segments']:
            total_raw_steps += len(segment)
            
    target_fps = 50
    target_total_frames = target_duration_sec * target_fps
    step_skip = max(1, total_raw_steps // target_total_frames)
    
    print(f"Generando GIF '{filename}'...")
    
    # 2. Setup Gráfico
    fig, ax = plt.subplots(figsize=(6, 10))
    # AQUÍ ESTÁ LA CLAVE: Usamos env.track, así sirve para cualquier mapa
    ax.imshow(env.track, cmap='gray_r') 
    ax.axis('off')

    line_red, = ax.plot([], [], 'r-', linewidth=2, label='Intento', zorder=10)
    point, = ax.plot([], [], 'ro', markersize=6, markeredgecolor='black', zorder=11)
    gray_lines_artists = [] 
    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", fontsize=10, weight='bold')

    def init():
        line_red.set_data([], [])
        point.set_data([], [])
        for art in gray_lines_artists: art.remove()
        gray_lines_artists.clear()
        return line_red, point, title_text

    def data_gen():
        # Limpieza inicial
        yield 'CLEAR_CANVAS', None, None, None

        for ep_data in data_subset:
            ep_num = ep_data['episode']
            segments = ep_data['segments']
            
            yield 'NEW_EPISODE', None, None, ep_num
            
            for seg_idx, segment in enumerate(segments):
                xs = [p[0] for p in segment]
                ys = [p[1] for p in segment]
                is_last = (seg_idx == len(segments) - 1)
                
                # Dibujo con salto de frames
                for k in range(0, len(xs), step_skip):
                    idx = min(k, len(xs)-1)
                    if k + step_skip >= len(xs): idx = len(xs) - 1
                    yield 'DRAW', xs[:idx+1], ys[:idx+1], (ep_num, seg_idx+1)
                
                if not is_last:
                    yield 'FREEZE_GRAY', xs, ys, None

    def run(data):
        action, xs, ys, info = data
        
        if action == 'CLEAR_CANVAS':
            for art in gray_lines_artists: art.remove()
            gray_lines_artists.clear()
            title_text.set_text(f"{title_prefix} Iniciando...")

        elif action == 'NEW_EPISODE':
            title_text.set_text(f"{title_prefix} Episodio: {info}")
            line_red.set_data([], [])

        elif action == 'FREEZE_GRAY':
            l, = ax.plot(xs, ys, color='gray', linewidth=1, alpha=0.3, zorder=1)
            gray_lines_artists.append(l)
            line_red.set_data([], [])
            
        elif action == 'DRAW':
            ep_num, try_num = info
            title_text.set_text(f"{title_prefix} Ep: {ep_num} | Intento: {try_num}")
            line_red.set_data(xs, ys)
            point.set_data([xs[-1]], [ys[-1]])
            
        return line_red, point, title_text

    frames_est = (total_raw_steps // step_skip) + (len(data_subset) * 5)
    
    anim = animation.FuncAnimation(fig, run, data_gen, init_func=init,
                                   save_count=frames_est, interval=1000//target_fps, 
                                   blit=False, repeat=False)
    try:
        anim.save(filename, writer='pillow', fps=target_fps)
        print(f"✅ GIF guardado: {filename}")
    except Exception as e:
        print(f"❌ Error: {e}")
    plt.close(fig)



# ctrl + k, ctrl + c
# ctrl + k, ctrl + u

# def create_static_trajectory_image(env, data_subset, filename, title="Trayectoria Final"):
#     """
#     Genera una imagen estática (PNG) mostrando la última trayectoria exitosa en rojo,
#     y todos los intentos fallidos previos en gris.
#     """
#     if not data_subset:
#         print(f"⚠️ No hay datos para generar la imagen {filename}")
#         return

#     print(f"Generando imagen estática: {filename}...")

#     fig, ax = plt.subplots(figsize=(6, 10))
#     ax.imshow(env.track, cmap='gray_r')
#     ax.axis('off')
#     ax.set_title(title, fontsize=12, weight='bold')

#     # --- 1. DIBUJAR LOS "FANTASMAS" (Gris) ---
#     # Recorremos todos los episodios EXCEPTO el último del subconjunto
#     for i in range(len(data_subset) - 1):
#         ep_data = data_subset[i]
#         for segment in ep_data['segments']:
#             xs = [p[0] for p in segment]
#             ys = [p[1] for p in segment]
#             ax.plot(xs, ys, color='gray', linewidth=1, alpha=0.3, zorder=1)

#     # Ahora miramos el ÚLTIMO episodio del subconjunto
#     last_ep_data = data_subset[-1]
#     segments = last_ep_data['segments']

#     # Dibujamos en gris sus intentos fallidos (todos menos el último segmento)
#     for i in range(len(segments) - 1):
#         segment = segments[i]
#         xs = [p[0] for p in segment]
#         ys = [p[1] for p in segment]
#         ax.plot(xs, ys, color='gray', linewidth=1, alpha=0.3, zorder=1)

#     # --- 2. DIBUJAR LA TRAYECTORIA FINAL (Rojo) ---
#     # Es el último segmento del último episodio
#     final_segment = segments[-1]
#     fxs = [p[0] for p in final_segment]
#     fys = [p[1] for p in final_segment]

#     ax.plot(fxs, fys, 'r-', linewidth=3, label='Ruta Exitosa Final', zorder=10)
#     # Marcamos el inicio (círculo verde) y el final (estrella roja) de esta ruta
#     ax.plot(fxs[0], fys[0], 'go', markersize=8, zorder=11, label='Inicio')
#     ax.plot(fxs[-1], fys[-1], 'r*', markersize=12, markeredgecolor='black', zorder=11, label='Meta')

#     ax.legend(loc='lower right')
#     plt.tight_layout()
    
#     # Guardar como PNG
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     print(f"✅ Imagen guardada: {filename}")
#     plt.close(fig)



def create_static_trajectory_image(env, data_subset, filename, title="Trayectoria Final"):
    """
    Genera una imagen estática (PNG) inteligente.
    - Busca y resalta en ROJO el último trayecto que llegó a META.
    - Si ninguno llegó, resalta el último intento y lo marca como "Fin (Tiempo)".
    - El resto de intentos se dibujan en GRIS.
    """
    if not data_subset:
        print(f"⚠️ No hay datos para generar la imagen {filename}")
        return

    print(f"Generando imagen estática inteligente: {filename}...")

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(env.track, cmap='gray_r')
    ax.axis('off')
    ax.set_title(title, fontsize=12, weight='bold')

    # --- 1. DIBUJAR EPISODIOS PREVIOS ENTEROS (Si hay) EN GRIS ---
    for i in range(len(data_subset) - 1):
        ep_data = data_subset[i]
        for segment in ep_data['segments']:
            xs = [p[0] for p in segment]
            ys = [p[1] for p in segment]
            ax.plot(xs, ys, color='gray', linewidth=1, alpha=0.3, zorder=1)

    # --- 2. PROCESAR EL EPISODIO OBJETIVO (El último del subset) ---
    target_ep_data = data_subset[-1]
    segments = target_ep_data['segments']

    # A) Identificar cuál es el segmento "ganador" (el que dibujaremos en rojo)
    successful_segments_indices = []
    for i, segment in enumerate(segments):
        # Miramos el último punto del segmento
        lx, ly = segment[-1]
        # Verificamos si está dentro del mapa y si es una casilla de META (valor 3)
        if 0 <= ly < env.height and 0 <= lx < env.width and env.track[ly, lx] == 3:
            successful_segments_indices.append(i)

    if successful_segments_indices:
        # CASO ÉXITO: Si hubo algún trayecto que llegó a meta, elegimos el último de ellos.
        final_idx = successful_segments_indices[-1]
        is_successful_run = True
    else:
        # CASO TIMEOUT: Si ninguno llegó, el "final" es simplemente el último intento que se hizo.
        final_idx = len(segments) - 1
        is_successful_run = False

    # B) Dibujar los segmentos de este episodio
    for i, segment in enumerate(segments):
        xs = [p[0] for p in segment]
        ys = [p[1] for p in segment]

        if i == final_idx:
            # --- ESTE ES EL SEGMENTO A RESALTAR EN ROJO ---
            label_ruta = 'Ruta Exitosa' if is_successful_run else 'Último Intento (Fallido)'
            ax.plot(xs, ys, 'r-', linewidth=3, label=label_ruta, zorder=10)
            ax.plot(xs[0], ys[0], 'go', markersize=8, zorder=11, label='Inicio')

            if is_successful_run:
                # Si llegó a meta: Estrella Roja y label "Meta"
                ax.plot(xs[-1], ys[-1], 'r*', markersize=12, markeredgecolor='black', zorder=11, label='Meta')
            else:
                # Si NO llegó a meta (timeout): Una 'X' roja y label distinto
                ax.plot(xs[-1], ys[-1], 'rX', markersize=10, markeredgecolor='black', zorder=11, label='Fin (Tiempo)')
        else:
            # --- RESTO DE SEGMENTOS EN GRIS ---
            ax.plot(xs, ys, color='gray', linewidth=1, alpha=0.3, zorder=1)

    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Imagen guardada: {filename}")
    plt.close(fig)