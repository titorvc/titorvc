import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

def main():
    print("=== Starting 3D Rendering (Web-Safe Cinematic Optimizations) ===")

    # 1. Environment and Paths Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(script_dir, 'dist_local')
    os.makedirs(dist_dir, exist_ok=True)
    print(f"Output Directory: {dist_dir}")

    # Set up global typography (Montserrat Bold) and light theme defaults
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Montserrat', 'DejaVu Sans', 'Arial']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['text.color'] = '#1E293B'      # Dark Slate
    plt.rcParams['axes.labelcolor'] = '#475569'  # Mid Slate
    plt.rcParams['xtick.color'] = '#64748B'
    plt.rcParams['ytick.color'] = '#64748B'

    # Performance & Frame Rate Optimization variables
    num_frames = 400  # More frames for super smooth progressive build-up
    interval_ms = 33  # ~30 FPS to bypass browser throttling
    fps_output = 30   # Target 30 FPS playback rate
    dpi_val = 80      # Optimized DPI for speed and compact GIF file sizes

    # Helper function to style 3D axes
    def style_3d_axes(ax):
        ax.set_facecolor('#FFFFFF')
        
        # Completely invisible 3D panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove hard spines (borders)
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Soft, thin, dotted grid lines
        ax.grid(True, color='#E2E8F0', linestyle=':', linewidth=0.5)
        
        # Tick parameters
        ax.tick_params(colors='#64748B', labelsize=8)
        
        # Force Montserrat Bold font on all axis tick labels
        for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels() + ax.zaxis.get_ticklabels():
            label.set_family('Montserrat')
            label.set_fontweight('bold')

    # =========================================================================
    # GRAPH 1: AI Cognitive Filter Analysis
    # =========================================================================
    csv_path = "/home/titorvc/Documents/job_hunter_ai/dags/data_lake/reporte_diario.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing AI dataset file: {csv_path}")

    df_ai = pd.read_csv(csv_path)
    df_ai['desc_len'] = df_ai['descripcion_full'].fillna('').str.len()
    df_ai['reason_len'] = df_ai['razon'].fillna('').str.len()

    # Fixed bounds for the dataset
    x1_min, x1_max = df_ai['score'].min() - 2, df_ai['score'].max() + 2
    y1_min, y1_max = df_ai['desc_len'].min() - 200, df_ai['desc_len'].max() + 200
    z1_min, z1_max = df_ai['reason_len'].min() - 20, df_ai['reason_len'].max() + 20

    # Initialize white figure
    fig1 = plt.figure(figsize=(9, 6.5), facecolor='#FFFFFF', dpi=dpi_val)
    fig1.subplots_adjust(top=0.82, bottom=0.08, left=0.05, right=0.95)
    
    # Title & English Context block
    fig1.text(0.06, 0.93, "AI Cognitive Filter Analysis", fontsize=14, fontweight='bold', color='#1E293B', family='Montserrat')
    fig1.text(0.06, 0.86, "Evaluating technical roles.\nAxes: Match Score vs. Job Complexity vs. Analysis Depth.", fontsize=9, color='#64748B', family='Montserrat', fontweight='bold')

    ax1 = fig1.add_subplot(projection='3d')

    def update_fig1(frame):
        ax1.clear()
        style_3d_axes(ax1)
        
        # Proportional data build-up slice
        N = len(df_ai)
        num_points = max(1, int((frame + 1) * N / num_frames))
        df_frame = df_ai.iloc[:num_points]
        
        X = df_frame['score']
        Y = df_frame['desc_len']
        Z = df_frame['reason_len']
        
        # Organic variable bubble sizing
        sizes = df_frame['reason_len'] * 0.5 + 50
        
        # 3D bubble scatter (winter colormap, clean transparency, no borders)
        sc = ax1.scatter(X, Y, Z, c=Z, cmap='winter', s=sizes, alpha=0.35, depthshade=True, edgecolors='none', vmin=z1_min, vmax=z1_max)
        
        # Keep limits fixed
        ax1.set_xlim(x1_min, x1_max)
        ax1.set_ylim(y1_min, y1_max)
        ax1.set_zlim(z1_min, z1_max)
        
        # Set Labels in English using Montserrat Bold
        ax1.set_xlabel("Match Score", labelpad=10, family='Montserrat', fontweight='bold')
        ax1.set_ylabel("Job Complexity", labelpad=10, family='Montserrat', fontweight='bold')
        ax1.set_zlabel("Analysis Depth", labelpad=10, family='Montserrat', fontweight='bold')
        
        # Slower, cinematic rotation step (0.15 degrees per frame)
        azim = (frame * 0.15) % 360
        ax1.view_init(elev=22, azim=azim)
        
        return fig1,

    # =========================================================================
    # GRAPH 2: n8n Multi-Agent Telemetry Stream
    # =========================================================================
    db_path = "/home/titorvc/Documents/Agencia_core/data/n8n/database.sqlite"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing n8n database file: {db_path}")

    conn = sqlite3.connect(db_path)
    df_n8n = pd.read_sql_query("SELECT periodStart, type, value FROM insights_by_period", conn)
    conn.close()

    df_n8n['periodStart_dt'] = pd.to_datetime(df_n8n['periodStart'])
    df_n8n = df_n8n.sort_values('periodStart_dt')
    df_n8n['periodStart_num'] = mdates.date2num(df_n8n['periodStart_dt'])

    # Fixed limits for telemetry
    x2_min, x2_max = df_n8n['periodStart_num'].min() - 0.05, df_n8n['periodStart_num'].max() + 0.05
    y2_min, y2_max = -0.5, 3.5
    z2_min, z2_max = 0, df_n8n['value'].max() + 1000

    # Extract 4 clean colors from the cool colormap
    cool_colors = [cm.cool(i) for i in np.linspace(0.15, 0.85, 4)]
    type_colors = {0: cool_colors[0], 1: cool_colors[1], 2: cool_colors[2], 3: cool_colors[3]}
    type_names = {0: 'Executions', 1: 'Active Workflows', 2: 'Errors', 3: 'Trigger Events'}

    # Initialize white figure
    fig2 = plt.figure(figsize=(9, 6.5), facecolor='#FFFFFF', dpi=dpi_val)
    fig2.subplots_adjust(top=0.82, bottom=0.08, left=0.05, right=0.95)
    
    # Title & English Context block
    fig2.text(0.06, 0.93, "n8n Multi-Agent Telemetry Stream", fontsize=14, fontweight='bold', color='#1E293B', family='Montserrat')
    fig2.text(0.06, 0.86, "Automated pipeline executions over time.\nAxes: Timeline vs. Task Type vs. Volume.", fontsize=9, color='#64748B', family='Montserrat', fontweight='bold')

    ax2 = fig2.add_subplot(projection='3d')

    def update_fig2(frame):
        ax2.clear()
        style_3d_axes(ax2)
        
        # Proportional data build-up slice
        N = len(df_n8n)
        num_points = max(1, int((frame + 1) * N / num_frames))
        df_frame = df_n8n.iloc[:num_points]
        
        # Plot each series
        for t, color in type_colors.items():
            df_t = df_frame[df_frame['type'] == t]
            if len(df_t) == 0:
                continue
            
            # Flow line with clean translucency (alpha=0.5)
            ax2.plot(df_t['periodStart_num'], [t]*len(df_t), df_t['value'], color=color, linewidth=2.0, alpha=0.5)
            
            # Dynamic sized bubbles proportional to value
            sizes = df_t['value'] * 0.005 + 40
            
            # Scatter bubbles (alpha=0.35, edgecolors='none')
            ax2.scatter(df_t['periodStart_num'], [t]*len(df_t), df_t['value'], color=color, s=sizes, alpha=0.35, depthshade=True, edgecolors='none')
            
            # Drop lines to grid floor
            for x, z in zip(df_t['periodStart_num'], df_t['value']):
                ax2.plot([x, x], [t, t], [0, z], color=color, alpha=0.1, linewidth=0.8, linestyle=':')
        
        # Lock limits
        ax2.set_xlim(x2_min, x2_max)
        ax2.set_ylim(y2_min, y2_max)
        ax2.set_zlim(z2_min, z2_max)
        
        # English Labels using Montserrat Bold
        ax2.set_xlabel("Timeline", labelpad=12, family='Montserrat', fontweight='bold')
        ax2.set_ylabel("Task Type", labelpad=12, family='Montserrat', fontweight='bold')
        ax2.set_zlabel("Volume", labelpad=10, family='Montserrat', fontweight='bold')
        
        # X-axis date styling
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        ax2.tick_params(axis='x', rotation=15)
        
        # Y-axis task labels
        ax2.set_yticks(list(type_colors.keys()))
        ax2.set_yticklabels([type_names[k] for k in type_colors.keys()], color='#64748B', fontsize=8, fontweight='bold', family='Montserrat')
        
        # Slower, cinematic rotation step (0.15 degrees per frame)
        azim = (frame * 0.15) % 360
        ax2.view_init(elev=25, azim=azim)
        
        return fig2,

    # =========================================================================
    # RENDER ANIMATIONS & SAVE GIFS
    # =========================================================================
    gif_path1 = os.path.join(dist_dir, 'ai_cognitive_filter.gif')
    print("Rendering Graphic 1 (AI Filter) at 30 FPS with cinematic slow rotation...")
    anim1 = FuncAnimation(fig1, update_fig1, frames=num_frames, interval=interval_ms)
    anim1.save(gif_path1, writer='pillow', fps=fps_output)
    print(f"  [SUCCESS] Saved: {gif_path1}")

    gif_path2 = os.path.join(dist_dir, 'n8n_telemetry.gif')
    print("Rendering Graphic 2 (n8n Telemetry) at 30 FPS with cinematic slow rotation...")
    anim2 = FuncAnimation(fig2, update_fig2, frames=num_frames, interval=interval_ms)
    anim2.save(gif_path2, writer='pillow', fps=fps_output)
    print(f"  [SUCCESS] Saved: {gif_path2}")

    print("\n=== All animations successfully updated at web-safe 30 FPS! ===")

if __name__ == "__main__":
    main()
