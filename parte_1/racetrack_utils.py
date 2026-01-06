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