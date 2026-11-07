"""
Figures of the laser-ablation model (laser_ablation_a..d.png): engagement geometry,
HP depletion for 1-3 drones, pointing cone, damage rate vs pointing error.

No trained model needed:  python plot_laser_ablation.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

LASER_COL  = '#ff0040'
PLUME_COL  = '#ff8c00'
DRONE_COLS_A = ['#0077aa', '#cc4400', '#229922']

L_BG    = 'white'
L_GRID  = '#dddddd'
L_TEXT  = '#111111'
L_SPINE = '#888888'
CURVE_COLS = ['#1f77b4', '#e07020', '#2ca02c']

# keep in sync with ADRSwarmEnv in swarm.py
FORM_SIDE     = 1.5
STANDOFF      = 3.0
LASER_RANGE   = 8.0
CONE_COS      = 0.97
CONE_HALF_DEG = np.degrees(np.arccos(CONE_COS))  # ~14.07 deg
DPS           = 0.5
DEBRIS_HP     = 200.0

_s3 = np.sqrt(3.)
FORM_OFFSETS = np.array([
    [ 0.0,         0.0,  FORM_SIDE / _s3],
    [-FORM_SIDE/2, 0.0, -FORM_SIDE/(2*_s3)],
    [ FORM_SIDE/2, 0.0, -FORM_SIDE/(2*_s3)],
])
DEBRIS_POS = np.array([0., 0., 0.])
CENTROID   = DEBRIS_POS + np.array([0., -STANDOFF, 0.])
DRONE_POS  = np.array([CENTROID + off for off in FORM_OFFSETS])

# ideal pointing: every laser axis aimed straight at the debris

def nose_dir(i):
    d = DEBRIS_POS - DRONE_POS[i]
    return d / np.linalg.norm(d)
NOSE = np.array([nose_dir(i) for i in range(3)])


def draw_cone(ax3d, apex, direction, half_angle_deg, length, color,
              n_rings=3, n_spokes=24, alpha=0.10, lw=0.6):
    """Wireframe pointing-tolerance cone (rim, spokes, inner rings)."""
    ha  = np.radians(half_angle_deg)
    nd  = direction / np.linalg.norm(direction)
    perp = np.array([1., 0., 0.]) if abs(nd[0]) < 0.9 else np.array([0., 1., 0.])
    e1   = np.cross(nd, perp); e1 /= np.linalg.norm(e1)
    e2   = np.cross(nd, e1)
    theta = np.linspace(0., 2*np.pi, n_spokes, endpoint=True)
    rim   = (apex[:,None] + length*nd[:,None] +
             length*np.tan(ha)*(e1[:,None]*np.cos(theta[None,:]) +
                                e2[:,None]*np.sin(theta[None,:])))
    ax3d.plot(rim[0], rim[1], rim[2], color=color, lw=lw, alpha=alpha*3)
    for t in theta[::max(1, n_spokes//8)]:
        tip = apex + length*nd + length*np.tan(ha)*(e1*np.cos(t)+e2*np.sin(t))
        ax3d.plot([apex[0],tip[0]],[apex[1],tip[1]],[apex[2],tip[2]],
                  color=color, lw=lw, alpha=alpha*2)
    for frac in np.linspace(0.2, 0.8, n_rings):
        rr = (apex[:,None] + frac*length*nd[:,None] +
              frac*length*np.tan(ha)*(e1[:,None]*np.cos(theta[None,:]) +
                                      e2[:,None]*np.sin(theta[None,:])))
        ax3d.plot(rr[0], rr[1], rr[2], color=color, lw=lw*0.7, alpha=alpha)


def draw_spacecraft(ax3d, pos, nose_vec, color, scale=0.18):
    """Box-shaped spacecraft aligned with nose_vec, plus a nose stick."""
    nd    = nose_vec / np.linalg.norm(nose_vec)
    perp  = np.array([1., 0., 0.]) if abs(nd[0]) < 0.9 else np.array([0., 1., 0.])
    e1    = np.cross(nd, perp); e1 /= np.linalg.norm(e1)
    e2    = np.cross(nd, e1)
    dx, dy, dz = scale*0.6, scale, scale*0.6
    corners = np.array([pos + sx*dx*e1 + sy*dy*nd + sz*dz*e2
                        for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])
    idx   = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    faces = [[corners[i] for i in face] for face in idx]
    poly  = Poly3DCollection(faces, alpha=0.55, facecolor=color,
                             edgecolor='black', linewidth=0.3)
    ax3d.add_collection3d(poly)
    tip = pos + nd*(dy + scale*0.7)
    ax3d.plot([pos[0],tip[0]],[pos[1],tip[1]],[pos[2],tip[2]],
              color='black', lw=1.5, solid_capstyle='round')
    ax3d.scatter(*tip, color=color, s=18, zorder=5)


def ablation_plume(ax3d, origin, rng=None):
    """Particle plume ejected from the debris (35 deg cone around -V-bar);
    the ablation reaction pushes the debris along -V-bar (retrograde)."""
    rng   = rng or np.random.default_rng(42)
    retro = np.array([0.,-1.,0.])
    n_p   = 360
    az    = rng.uniform(0, 2*np.pi, n_p)
    el    = rng.uniform(0, np.radians(35), n_p)
    vecs  = (np.cos(el)[:,None]*retro[None,:] +
             np.sin(el)[:,None]*(np.cos(az)[:,None]*np.array([[1,0,0]])+
                                 np.sin(az)[:,None]*np.array([[0,0,1]])))
    dist  = rng.uniform(0.05, 0.9, n_p)
    pts   = origin[None,:] + vecs*dist[:,None]
    sizes = (1.-dist/0.9)*14+1
    alpha = (1.-dist/0.9)*0.85
    order = np.argsort(dist)[::-1]
    for s in range(0, n_p, 30):
        sl = order[s:s+30]
        ax3d.scatter(pts[sl,0], pts[sl,1], pts[sl,2],
                     c=PLUME_COL, s=sizes[sl],
                     alpha=float(np.mean(alpha[sl])),
                     depthshade=False, zorder=4)


def style_light(ax, xlabel='', ylabel='', title=''):
    """Shared light style for the 2-D panels."""
    ax.set_facecolor(L_BG)
    ax.grid(True, color=L_GRID, lw=0.7, linestyle='--', zorder=0)
    ax.tick_params(colors=L_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(L_SPINE)
        sp.set_linewidth(0.8)
    ax.xaxis.label.set_color(L_TEXT)
    ax.yaxis.label.set_color(L_TEXT)
    ax.title.set_color(L_TEXT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=L_TEXT)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=L_TEXT)
    if title:  ax.set_title(title, fontsize=12, color=L_TEXT, fontweight='bold', pad=8)


# panel a: 3-D engagement geometry (LVLH)
fig_a = plt.figure(figsize=(9, 8), facecolor='white')
fig_a.patch.set_facecolor('white')
ax3d = fig_a.add_subplot(111, projection='3d')

ax3d.set_facecolor('white')
ax3d.grid(True, color='#cccccc', lw=0.5, linestyle='--')
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = True
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.set_facecolor('white')
    pane.set_edgecolor('#cccccc')

# debris shown at 35% HP
u_s, v_s = np.mgrid[0:2*np.pi:32j, 0:np.pi:16j]
r_deb = 0.22; HP_FRAC = 0.35
ax3d.plot_surface(r_deb*np.cos(u_s)*np.sin(v_s),
                  r_deb*np.sin(u_s)*np.sin(v_s),
                  r_deb*np.cos(v_s),
                  color=((1-HP_FRAC), HP_FRAC*0.5, 0.0),
                  alpha=0.90, linewidth=0.3, antialiased=True, zorder=2)
for hr, ha in [(0.30, 0.06), (0.40, 0.03)]:
    ax3d.plot_surface(hr*np.cos(u_s)*np.sin(v_s), hr*np.sin(u_s)*np.sin(v_s),
                      hr*np.cos(v_s), color=(1., 0.4, 0.), alpha=ha,
                      linewidth=0, antialiased=False)
ax3d.text(0, 0, r_deb+0.35, "DEBRIS\n(HP 35 %)",
          color='#cc2200', fontsize=9, ha='center', fontweight='bold')

ablation_plume(ax3d, DEBRIS_POS)

ax3d.quiver(0, 0, 0, 0, -1.1, 0, length=1.0, arrow_length_ratio=0.25,
            color='#333333', lw=1.5, alpha=0.85)
ax3d.text(0, -1.2, 0.12, r'$\Delta\mathbf{v}_{abl}$',
          color='#333333', fontsize=9, ha='center')

labels3d = ['Drone 0', 'Drone 1', 'Drone 2']
for i in range(3):
    p = DRONE_POS[i]; nd = NOSE[i]; c = DRONE_COLS_A[i]
    draw_spacecraft(ax3d, p, nd, c)
    for lw_i, al_i in [(4, 0.10), (2, 0.25), (1.2, 0.90)]:
        ax3d.plot([p[0], 0], [p[1], 0], [p[2], 0],
                  color=LASER_COL, lw=lw_i, alpha=al_i,
                  solid_capstyle='round', zorder=5)
    draw_cone(ax3d, p, nd, CONE_HALF_DEG, np.linalg.norm(DEBRIS_POS - p),
              LASER_COL, n_rings=2, n_spokes=20, alpha=0.14)
    ax3d.text(p[0]+0.05, p[1]-0.35, p[2]+0.12,
              labels3d[i], color=c, fontsize=9, fontweight='bold')

for i in range(3):
    j = (i+1) % 3
    ax3d.plot([DRONE_POS[i,0], DRONE_POS[j,0]],
              [DRONE_POS[i,1], DRONE_POS[j,1]],
              [DRONE_POS[i,2], DRONE_POS[j,2]],
              color='#999999', lw=0.8, ls='--', alpha=0.6)

ax3d.scatter(*CENTROID, color='#444444', s=30, marker='+', linewidths=1.4,
             zorder=6, alpha=0.8)
ax3d.plot([0, 0], [0, CENTROID[1]], [0, 0],
          color='#444444', lw=0.8, ls=':', alpha=0.6)
ax3d.text(0.08, CENTROID[1]/2, 0.05, f'{STANDOFF:.1f} m',
          color='#333333', fontsize=8, alpha=0.9)

ax3d.set_xlabel('R-bar [m]', labelpad=4, fontsize=9, color='black')
ax3d.set_ylabel('V-bar [m]', labelpad=4, fontsize=9, color='black')
ax3d.set_zlabel('H-bar [m]', labelpad=4, fontsize=9, color='black')
ax3d.tick_params(colors='black', labelsize=7)
ax3d.set_title('Laser Engagement Geometry — LVLH Frame',
               color='black', fontsize=12, pad=10, fontweight='bold')
ax3d.set_xlim(-1.8, 1.8); ax3d.set_ylim(-4.5, 1.5); ax3d.set_zlim(-1.5, 1.5)
ax3d.view_init(elev=22, azim=-50)

legend_handles = ([mpatches.Patch(color=c, label=labels3d[i])
                   for i, c in enumerate(DRONE_COLS_A)] +
                  [mpatches.Patch(color=LASER_COL, label='Laser beam / cone'),
                   mpatches.Patch(color=PLUME_COL, label='Ablation plume')])
ax3d.legend(handles=legend_handles, loc='upper left', fontsize=8,
            facecolor='white', edgecolor='#888888', labelcolor='black',
            framealpha=0.92)

plt.tight_layout()
plt.savefig('laser_ablation_a.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Saved → laser_ablation_a.png")
plt.close(fig_a)


# panel b: HP depletion
fig_b, ax_hp = plt.subplots(figsize=(8, 5), facecolor='white')
fig_b.patch.set_facecolor('white')

style_light(ax_hp,
            xlabel='Time  [s]',
            ylabel='Debris HP',
            title='HP Depletion vs. Time — Cooperative Scaling')

t_max = 450
t_arr = np.linspace(0, t_max, 2000)

for n, (c, ls, lw) in enumerate(zip(CURVE_COLS, ['--','-.', '-'], [2.0, 2.0, 2.5]), start=1):
    rate   = DPS * n          # cos = 1, damage sums over drones
    t_kill = DEBRIS_HP / rate
    hp     = np.maximum(0., DEBRIS_HP - rate*t_arr)
    label  = (f'{n} drone{"s" if n>1 else ""}  '
              f'($t_{{\\rm kill}}$ = {t_kill:.0f} s)')
    ax_hp.plot(t_arr, hp, color=c, lw=lw, ls=ls, label=label, zorder=4)
    ax_hp.fill_between(t_arr, hp, alpha=0.10, color=c, zorder=3)
    ax_hp.axvline(t_kill, color=c, lw=1.0, ls=':', alpha=0.7, zorder=3)
    ax_hp.annotate(f'{t_kill:.0f} s',
                   xy=(t_kill, 0), xytext=(t_kill + 5, 22 - n*5),
                   fontsize=9, color=c, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=c, lw=0.9))

ax_hp.axhline(0, color='#cc0000', lw=1.2, ls='-', zorder=5)
ax_hp.fill_betweenx([0, 8], 0, t_max, color='#ff0000', alpha=0.07, zorder=2)
ax_hp.text(430, 4, 'DESTROYED', color='#cc0000', fontsize=9,
           ha='right', fontweight='bold', va='center')

ax_hp.set_xlim(0, t_max); ax_hp.set_ylim(-8, 215)
ax_hp.xaxis.set_minor_locator(MultipleLocator(25))
ax_hp.yaxis.set_minor_locator(MultipleLocator(20))
ax_hp.tick_params(which='minor', length=2, color=L_SPINE)

legend = ax_hp.legend(fontsize=10, framealpha=0.95, loc='upper right',
                      edgecolor=L_SPINE, facecolor='white')
for text in legend.get_texts(): text.set_color(L_TEXT)

ax_hp.text(230, 65,
           r'$\dot{HP} = N_{drones} \times \mathrm{DPS} \times \cos\theta$',
           fontsize=10, color='#333333', ha='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff',
                     edgecolor='#8888cc', lw=0.8))

plt.tight_layout()
plt.savefig('laser_ablation_b.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Saved → laser_ablation_b.png")
plt.close(fig_b)


# panel c: pointing cone cross-section
fig_c, ax_cn = plt.subplots(figsize=(6, 7), facecolor='white')
fig_c.patch.set_facecolor('white')

style_light(ax_cn,
            xlabel='Lateral offset  [m]',
            ylabel='Range  [m]',
            title='Pointing-Tolerance Cone Geometry')

ax_cn.set_aspect('equal')
ranges   = np.linspace(0, LASER_RANGE, 400)
ha_rad   = np.radians(CONE_HALF_DEG)
cone_r   = ranges * np.tan(ha_rad)

ax_cn.fill_betweenx(ranges, -cone_r, cone_r,
                    color='#ff4444', alpha=0.13, label='Engagement cone', zorder=2)
ax_cn.plot( cone_r, ranges, color='#cc0000', lw=1.8, alpha=0.85, zorder=4)
ax_cn.plot(-cone_r, ranges, color='#cc0000', lw=1.8, alpha=0.85, zorder=4)

ax_cn.axvline(0, color='#444444', lw=0.9, ls='--', alpha=0.6, label='Boresight axis', zorder=3)

ax_cn.scatter([0],[0], color=DRONE_COLS_A[0], s=120, zorder=7,
              marker='s', label='Drone (apex)')
ax_cn.annotate('Drone', xy=(0, 0), xytext=(-1.8, 0.6),
               fontsize=9, color=DRONE_COLS_A[0], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=DRONE_COLS_A[0], lw=0.9))

r_at_std = STANDOFF * np.tan(ha_rad)
ax_cn.scatter([0],[STANDOFF], color='#dd5500', s=120, zorder=7,
              marker='o', label='Debris (standoff)')
ax_cn.annotate('Debris\n(standoff)', xy=(0.02, STANDOFF),
               xytext=(0.5, STANDOFF+0.35),
               fontsize=9, color='#dd5500', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#dd5500', lw=0.9))

ax_cn.annotate('', xy=(r_at_std, STANDOFF), xytext=(-r_at_std, STANDOFF),
               arrowprops=dict(arrowstyle='<->', color='#2266aa', lw=1.4))
ax_cn.text(0, STANDOFF + 0.22, f'2 × {r_at_std:.2f} m = {2*r_at_std:.2f} m  (cone diameter)',
           color='#2266aa', fontsize=8.5, ha='center', fontweight='bold')

ax_cn.axhline(LASER_RANGE, color='#888888', lw=0.9, ls=':', alpha=0.8, zorder=3)
ax_cn.text(-2.3, LASER_RANGE+0.12, f'Max range  {LASER_RANGE:.0f} m',
           color='#666666', fontsize=9)

theta_arc = np.linspace(np.pi/2 - ha_rad, np.pi/2, 60)
arc_r_val = 1.0
ax_cn.plot(arc_r_val*np.cos(theta_arc), arc_r_val*np.sin(theta_arc),
           color='#aa6600', lw=2.0, alpha=0.9, zorder=5)
ax_cn.text(0.62, 0.98, f'{CONE_HALF_DEG:.1f}°',
           color='#aa6600', fontsize=11, fontweight='bold', zorder=6)

ax_cn.text(0, -0.65, rf'Firing condition: $\cos\theta \geq {CONE_COS}$',
           color='#cc0000', fontsize=9, ha='center', fontweight='bold')

ax_cn.set_xlim(-2.6, 2.6); ax_cn.set_ylim(-0.9, 9.5)
legend_c = ax_cn.legend(fontsize=9, framealpha=0.95, loc='upper right',
                        edgecolor=L_SPINE)
for t in legend_c.get_texts(): t.set_color(L_TEXT)

plt.tight_layout()
plt.savefig('laser_ablation_c.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Saved → laser_ablation_c.png")
plt.close(fig_c)


# panel d: damage rate vs pointing error
fig_d, ax_dm = plt.subplots(figsize=(8, 5), facecolor='white')
fig_d.patch.set_facecolor('white')

style_light(ax_dm,
            xlabel='Pointing error  θ  [°]',
            ylabel='Damage rate  [hp / s]',
            title='Damage Rate vs. Pointing Error')

theta_deg = np.linspace(0, 25, 600)
theta_rad = np.radians(theta_deg)
cos_t     = np.cos(theta_rad)
dmg_1 = DPS * cos_t
dmg_3 = 3 * DPS * cos_t

ax_dm.axvspan(CONE_HALF_DEG, 25, color='#ffcccc', alpha=0.55, zorder=1,
              label=f'Outside cone  (θ > {CONE_HALF_DEG:.1f}°)')
ax_dm.axvline(CONE_HALF_DEG, color='#cc0000', lw=1.4, ls='--', zorder=4)
ax_dm.text(CONE_HALF_DEG + 0.4, 0.52,
           f'θ = {CONE_HALF_DEG:.1f}°\n(cos θ = {CONE_COS})',
           color='#cc0000', fontsize=9, va='top', fontweight='bold')

ax_dm.fill_between(theta_deg, dmg_1, where=(theta_deg<=CONE_HALF_DEG),
                   alpha=0.12, color=CURVE_COLS[0], zorder=2)
ax_dm.fill_between(theta_deg, dmg_3, where=(theta_deg<=CONE_HALF_DEG),
                   alpha=0.07, color=CURVE_COLS[2], zorder=2)

ax_dm.plot(theta_deg, dmg_1, color=CURVE_COLS[0], lw=2.2, zorder=5,
           label=f'1 drone  (max {DPS:.1f} hp/s)')
ax_dm.plot(theta_deg, dmg_3, color=CURVE_COLS[2], lw=2.2, ls='--', zorder=5,
           label=f'3 drones  (max {3*DPS:.1f} hp/s)')

ax_dm.scatter([0],[DPS],   color=CURVE_COLS[0], s=70, zorder=6)
ax_dm.scatter([0],[3*DPS], color=CURVE_COLS[2], s=70, zorder=6)
ax_dm.text(0.35, DPS   + 0.015, f'{DPS:.2f} hp/s',   color=CURVE_COLS[0], fontsize=9, fontweight='bold')
ax_dm.text(0.35, 3*DPS + 0.015, f'{3*DPS:.2f} hp/s', color=CURVE_COLS[2], fontsize=9, fontweight='bold')

ax_dm.text(14, 0.38,
           r'$\dot{HP}_i = \mathrm{DPS} \cdot \cos\theta_i$',
           fontsize=10, color='#333333',
           bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0f4ff',
                     edgecolor='#8888cc', lw=0.8))

ax_dm.set_xlim(0, 25); ax_dm.set_ylim(-0.02, 0.64)
ax_dm.xaxis.set_minor_locator(MultipleLocator(2.5))
ax_dm.yaxis.set_minor_locator(MultipleLocator(0.05))
ax_dm.tick_params(which='minor', length=2, color=L_SPINE)

legend_d = ax_dm.legend(fontsize=10, framealpha=0.95, loc='upper right',
                        edgecolor=L_SPINE)
for t in legend_d.get_texts(): t.set_color(L_TEXT)

plt.tight_layout()
plt.savefig('laser_ablation_d.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Saved → laser_ablation_d.png")
plt.close(fig_d)

print("\n[DONE] All 4 panels saved as separate files.")
