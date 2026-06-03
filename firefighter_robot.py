# ============================================================
#   FIREFIGHTER ROBOT — FULLY ANIMATED VERSION
#   BS(AI) 6th Semester — Robotics Lab | Dr. Adil Khan
#   Maham Qaiser (01-136232-025) | Imama Tufail (01-136232-021)
#
#   pip install matplotlib numpy
#   python firefighter_robot.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import heapq, math, random

# ─────────────────────────────────────────────
#  GRID CONSTANTS
# ─────────────────────────────────────────────
ROWS, COLS = 20, 20
FREE, WALL, FIRE, VISITED, PATH, EXTINGUISHED = 0,1,2,3,4,5

# ─────────────────────────────────────────────
#  WORLD
# ─────────────────────────────────────────────
def create_world():
    grid = np.zeros((ROWS, COLS), dtype=int)
    walls = [
        (2,4),(2,5),(2,6),(2,7),
        (5,10),(6,10),(7,10),(8,10),
        (10,3),(10,4),(10,5),
        (4,15),(5,15),(6,15),
        (13,12),(13,13),(13,14),
        (15,2),(16,2),(17,2),
        (0,8),(1,8),(1,9),
        (12,7),(12,8),
    ]
    for r,c in walls:
        grid[r][c] = WALL
    fires = [(3,9),(3,10),(8,16),(8,17),(16,14),(16,15),(18,5),(18,6)]
    for r,c in fires:
        grid[r][c] = FIRE
    return grid, list(fires)

# ─────────────────────────────────────────────
#  A*
# ─────────────────────────────────────────────
def astar(grid, start, goal):
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    heap=[(h(start,goal),start)]; came={}
    g={start:0}; vis=set()
    while heap:
        _,cur=heapq.heappop(heap)
        if cur in vis: continue
        vis.add(cur)
        if cur==goal:
            path=[]
            while cur in came: path.append(cur); cur=came[cur]
            path.append(start); return path[::-1]
        r,c=cur
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb=(r+dr,c+dc)
            if not(0<=nb[0]<ROWS and 0<=nb[1]<COLS): continue
            if grid[nb[0]][nb[1]]==WALL: continue
            ng=g[cur]+1
            if ng<g.get(nb,99999):
                came[nb]=cur; g[nb]=ng
                heapq.heappush(heap,(ng+h(nb,goal),nb))
    return []

# ─────────────────────────────────────────────
#  PID
# ─────────────────────────────────────────────
class PID:
    def __init__(self,kp=1.2,ki=0.04,kd=0.25):
        self.kp=kp;self.ki=ki;self.kd=kd
        self.prev=0;self.intg=0;self.history=[]
    def compute(self,target,current,dt=0.1):
        e=target-current; self.intg+=e*dt
        d=(e-self.prev)/dt; self.prev=e
        out=self.kp*e+self.ki*self.intg+self.kd*d
        self.history.append((round(e,3),round(out,3))); return out
    def reset(self): self.prev=0;self.intg=0

# ─────────────────────────────────────────────
#  SENSOR FUSION
# ─────────────────────────────────────────────
def sensor_fusion(grid,pos):
    LIDAR_R,HEAT_R=3,8
    r,c=pos; obstacles,fires=[],[]
    for dr in range(-HEAT_R,HEAT_R+1):
        for dc in range(-HEAT_R,HEAT_R+1):
            nr,nc=r+dr,c+dc
            if not(0<=nr<ROWS and 0<=nc<COLS): continue
            dist=math.sqrt(dr*dr+dc*dc)
            cell=grid[nr][nc]
            if dist<=LIDAR_R and cell==WALL: obstacles.append((nr,nc))
            if dist<=HEAT_R and cell==FIRE:
                conf=round(max(0.1,1.0-(dist/HEAT_R)),2)
                fires.append(((nr,nc),conf))
    fires.sort(key=lambda x:x[1],reverse=True)
    return obstacles,fires

# ─────────────────────────────────────────────
#  ROBOT
# ─────────────────────────────────────────────
class FirefighterRobot:
    def __init__(self,start,grid,fires):
        self.pos=start; self.grid=grid
        self.remaining=list(fires); self.extinguished=[]
        self.visited=set(); self.trail=[start]
        self.path=[]; self.target=None
        self.pid=PID(); self.state="PLANNING"
        self.status="Starting..."; self.direction=(0,1)
        self.ext_anim=0

    def _pick_target(self):
        best,bp=None,[]
        for fire in self.remaining:
            p=astar(self.grid,self.pos,fire)
            if p and(not bp or len(p)<len(bp)): best,bp=fire,p
        return best,bp

    def step(self):
        self.visited.add(self.pos)
        _,fused=sensor_fusion(self.grid,self.pos)

        if self.state=="PLANNING":
            if not self.remaining: self.state="DONE"; return self.state
            if fused:
                self.target=fused[0][0]
                self.path=astar(self.grid,self.pos,self.target)
            else:
                self.target,self.path=self._pick_target()
            if self.path:
                self.state="MOVING_TO_FIRE"
                self.status=f"A* path → {self.target} ({len(self.path)} steps)"
                print(f"[{self.pos}] Target:{self.target} Steps:{len(self.path)}")
            else:
                if self.remaining: self.remaining.pop(0)

        elif self.state=="MOVING_TO_FIRE":
            if not self.path or len(self.path)<2:
                self.state="EXTINGUISHING"; self.ext_anim=0; return self.state
            nxt=self.path[1]
            if self.grid[nxt[0]][nxt[1]]==WALL:
                self.path=astar(self.grid,self.pos,self.target); return self.state
            self.direction=(nxt[0]-self.pos[0],nxt[1]-self.pos[1])
            self.pid.compute(nxt[0],self.pos[0])
            self.pos=nxt; self.trail.append(self.pos); self.path.pop(0)
            self.status=f"Moving → {self.pos}"
            if self.pos==self.target: self.state="EXTINGUISHING"; self.ext_anim=0

        elif self.state=="EXTINGUISHING":
            self.ext_anim+=1
            if self.ext_anim>=4:
                if self.target and self.grid[self.target[0]][self.target[1]]==FIRE:
                    self.grid[self.target[0]][self.target[1]]=EXTINGUISHED
                    self.extinguished.append(self.target)
                    if self.target in self.remaining: self.remaining.remove(self.target)
                    self.status=f"Extinguished {self.target}!"
                    print(f"[{self.pos}] {self.status}")
                self.pid.reset(); self.target=None; self.path=[]; self.state="PLANNING"

        return self.state

# ─────────────────────────────────────────────
#  DRAW FUNCTIONS
# ─────────────────────────────────────────────
def draw_wall(ax, c, r):
    """Brick-style wall."""
    face=FancyBboxPatch((c-0.48,r-0.48),0.96,0.96,
        boxstyle="round,pad=0.03",
        facecolor='#3d3d4a',edgecolor='#1a1a2e',linewidth=0.8,zorder=3)
    ax.add_patch(face)
    # Brick lines
    for brow in [r-0.16,r+0.16]:
        ax.plot([c-0.48,c+0.48],[brow,brow],color='#1a1a2e',lw=0.5,zorder=4)
    for bcol in [c]:
        ax.plot([bcol,bcol],[r-0.48,r-0.16],color='#1a1a2e',lw=0.5,zorder=4)
    for bcol in [c-0.24,c+0.24]:
        ax.plot([bcol,bcol],[r+0.16,r+0.48],color='#1a1a2e',lw=0.5,zorder=4)
    # Shine
    ax.fill([c-0.48,c+0.48,c+0.38,c-0.38],
            [r-0.48,r-0.48,r-0.38,r-0.38],
            color='white',alpha=0.08,zorder=4)

def draw_fire(ax, c, r, pulse):
    """Animated flickering fire."""
    s=1.0+0.2*math.sin(pulse)
    # Glow background
    glow=Circle((c,r),0.48*s,facecolor='#FF6F00',alpha=0.15,zorder=3)
    ax.add_patch(glow)
    # Outer flame
    ax.fill([c-0.32*s,c,c+0.32*s,c+0.18*s,c,-0.0+c,-0.18*s+c,-0.32*s+c],
            [r+0.42*s,r-0.42*s,r+0.42*s,r+0.08*s,r+0.32*s,r+0.32*s,r+0.08*s,r+0.42*s],
            color='#FF6F00',alpha=0.95,zorder=4)
    # Mid flame
    ax.fill([c-0.18*s,c,c+0.18*s,c],
            [r+0.18*s,r-0.32*s,r+0.18*s,r+0.38*s],
            color='#FFCA28',alpha=0.95,zorder=5)
    # Core
    ax.fill([c-0.07*s,c,c+0.07*s,c],
            [r+0.05*s,r-0.18*s,r+0.05*s,r+0.18*s],
            color='white',alpha=0.7,zorder=6)
    # Sparks
    for _ in range(3):
        sx=c+random.uniform(-0.3,0.3)*s
        sy=r+random.uniform(-0.48,0.48)*s
        ax.plot(sx,sy,'.',color='#FFEB3B',
                markersize=random.uniform(1,3),alpha=0.7,zorder=7)

def draw_robot(ax, c, r, state, tick, direction=(0,1)):
    """Proper firefighter robot with helmet, body, wheels, hose."""
    # Shadow
    shadow=Circle((c+0.05,r+0.08),0.42,facecolor='black',alpha=0.18,zorder=8)
    ax.add_patch(shadow)

    # Body (chassis)
    body=FancyBboxPatch((c-0.35,r-0.30),0.70,0.60,
        boxstyle="round,pad=0.05",
        facecolor='#1565C0',edgecolor='#0D47A1',linewidth=1.2,zorder=9)
    ax.add_patch(body)

    # Chest stripe (yellow)
    stripe=FancyBboxPatch((c-0.35,r-0.04),0.70,0.08,
        boxstyle="square,pad=0",
        facecolor='#FFEB3B',edgecolor='none',alpha=0.9,zorder=10)
    ax.add_patch(stripe)

    # Helmet
    helmet=Circle((c,r+0.30),0.20,
        facecolor='#C62828',edgecolor='#B71C1C',linewidth=1.0,zorder=11)
    ax.add_patch(helmet)

    # Visor
    visor=FancyBboxPatch((c-0.13,r+0.22),0.26,0.12,
        boxstyle="round,pad=0.02",
        facecolor='#80DEEA',edgecolor='#006064',linewidth=0.7,alpha=0.95,zorder=12)
    ax.add_patch(visor)

    # Water tank (green, on back)
    tank=FancyBboxPatch((c+0.26,r-0.20),0.16,0.40,
        boxstyle="round,pad=0.02",
        facecolor='#388E3C',edgecolor='#1B5E20',linewidth=0.8,zorder=9)
    ax.add_patch(tank)

    # Wheels
    for wx,wy in [(c-0.20,r-0.36),(c+0.20,r-0.36)]:
        w=Circle((wx,wy),0.10,facecolor='#212121',edgecolor='#757575',linewidth=0.8,zorder=9)
        ax.add_patch(w)
        ax.plot(wx,wy,'o',color='#9E9E9E',markersize=2.5,zorder=10)

    # Hose arm
    if state=="EXTINGUISHING":
        # Spray water
        ax.plot([c+0.35,c+0.55],[r+0.05,r+0.05],
                color='#546E7A',linewidth=3,zorder=11,solid_capstyle='round')
        for i in range(7):
            angle=random.uniform(-0.5,0.5)
            dist=random.uniform(0.2,0.65)
            wx2=c+0.55+dist*math.cos(angle)
            wy2=r+0.05+dist*math.sin(angle)
            ax.plot(wx2,wy2,'o',color='#29B6F6',
                    markersize=random.uniform(2,6),alpha=0.8,zorder=14)
        # Water arc
        t=np.linspace(0,1,10)
        wx_arc=c+0.55+0.6*t
        wy_arc=r+0.05+0.3*t-0.5*t**2
        ax.plot(wx_arc,wy_arc,'-',color='#29B6F6',lw=2,alpha=0.7,zorder=13)
        ax.text(c+0.9,r+0.35,'💧',fontsize=9,zorder=15,ha='center',va='center')
    else:
        # Arm at rest
        ax.plot([c+0.35,c+0.50],[r+0.05,r+0.05],
                color='#546E7A',linewidth=2.5,zorder=11,solid_capstyle='round')

    # Siren (blinking red/yellow)
    siren_col='#FF1744' if tick%2==0 else '#FFEA00'
    ax.plot(c,r+0.50,'o',color=siren_col,markersize=5,
            markeredgecolor='white',markeredgewidth=0.5,zorder=13)
    if tick%2==0:
        glow2=Circle((c,r+0.50),0.12,facecolor='#FF1744',alpha=0.25,zorder=12)
        ax.add_patch(glow2)

    # Label
    ax.text(c,r+0.08,'FF',fontsize=5,color='white',
            fontweight='bold',ha='center',va='center',zorder=13)

def draw_extinguished(ax,c,r):
    """Wet ash — extinguished fire."""
    bg=FancyBboxPatch((c-0.42,r-0.42),0.84,0.84,
        boxstyle="round,pad=0.05",
        facecolor='#37474F',edgecolor='#263238',linewidth=0.8,alpha=0.8,zorder=3)
    ax.add_patch(bg)
    ax.text(c,r,'✓',fontsize=12,color='#80CBC4',
            ha='center',va='center',fontweight='bold',zorder=4)

def draw_path(ax,path):
    """Yellow dotted path arrows."""
    if len(path)<2: return
    for i in range(min(len(path)-1,10)):
        r1,c1=path[i]; r2,c2=path[i+1]
        ax.annotate('',xy=(c2,r2),xytext=(c1,r1),
            arrowprops=dict(arrowstyle='->',color='#FFEB3B',
                            lw=1.0,alpha=0.55),zorder=6)

def draw_lidar_ring(ax,pos):
    """LiDAR detection ring."""
    r,c=pos
    ring=Circle((c,r),3.0,fill=False,edgecolor='#29B6F6',
        linewidth=0.7,linestyle='--',alpha=0.25,zorder=5)
    ax.add_patch(ring)
    ring2=Circle((c,r),8.0,fill=False,edgecolor='#FF7043',
        linewidth=0.5,linestyle=':',alpha=0.15,zorder=5)
    ax.add_patch(ring2)

def draw_trail(ax,trail):
    if len(trail)<2: return
    n=len(trail)
    for i,(tr,tc) in enumerate(trail[:-1]):
        a=0.05+0.3*(i/max(n,1))
        ax.plot(tc,tr,'o',color='#29B6F6',markersize=2,alpha=a,zorder=4)
    rs=[p[0] for p in trail]; cs=[p[1] for p in trail]
    ax.plot(cs,rs,'-',color='#29B6F6',alpha=0.2,linewidth=0.8,zorder=4)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    grid,fires=create_world()
    robot=FirefighterRobot((0,0),grid,fires)

    fig=plt.figure(figsize=(16,8),facecolor='#0d1117')
    fig.suptitle(
        '  Firefighter Robot  |  BS(AI) 6th Semester  |  Dr. Adil Khan  ',
        color='white',fontsize=12,fontweight='bold',y=0.99)

    ax_g =fig.add_axes([0.01,0.05,0.57,0.90])
    ax_p =fig.add_axes([0.63,0.54,0.35,0.38])
    ax_st=fig.add_axes([0.63,0.08,0.35,0.38])

    for ax in [ax_g,ax_p,ax_st]:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e',labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

    steps=[0]; p_err=[]; p_out=[]; p_x=[]; pulse=[0.0]

    def frame(_):
        state=robot.step()
        steps[0]+=1; pulse[0]+=0.35

        # ── GRID ─────────────────────────────────────
        ax_g.cla(); ax_g.set_facecolor('#161b22')
        ax_g.set_xlim(-0.5,COLS-0.5); ax_g.set_ylim(ROWS-0.5,-0.5)
        ax_g.set_aspect('equal')

        # Checkerboard floor
        for r in range(ROWS):
            for c in range(COLS):
                col='#1c2128' if (r+c)%2==0 else '#161b22'
                ax_g.fill([c-0.5,c+0.5,c+0.5,c-0.5],
                          [r-0.5,r-0.5,r+0.5,r+0.5],color=col,zorder=0)

        # Grid lines
        for i in range(ROWS+1):
            ax_g.axhline(i-0.5,color='#21262d',lw=0.3,zorder=1)
        for j in range(COLS+1):
            ax_g.axvline(j-0.5,color='#21262d',lw=0.3,zorder=1)

        # Visited
        for vr,vc in robot.visited:
            if robot.grid[vr][vc]==FREE:
                ax_g.fill([vc-0.5,vc+0.5,vc+0.5,vc-0.5],
                          [vr-0.5,vr-0.5,vr+0.5,vr+0.5],
                          color='#0e4429',alpha=0.55,zorder=2)

        # All cells
        for r in range(ROWS):
            for c in range(COLS):
                cell=robot.grid[r][c]
                if cell==WALL:
                    draw_wall(ax_g,c,r)
                elif cell==FIRE:
                    ax_g.fill([c-0.5,c+0.5,c+0.5,c-0.5],
                              [r-0.5,r-0.5,r+0.5,r+0.5],
                              color='#3d0000',zorder=2)
                    draw_fire(ax_g,c,r,pulse[0])
                elif cell==EXTINGUISHED:
                    draw_extinguished(ax_g,c,r)

        # A* path
        if robot.path: draw_path(ax_g,robot.path)

        # Target highlight
        if robot.target:
            tr2,tc2=robot.target
            ring=Circle((tc2,tr2),0.55,fill=False,edgecolor='#FFEB3B',
                lw=1.5,linestyle='-',alpha=0.85,zorder=7)
            ax_g.add_patch(ring)
            ax_g.text(tc2,tr2-0.75,'TARGET',fontsize=5,
                color='#FFEB3B',ha='center',alpha=0.85,zorder=7)

        # Sensor rings
        draw_lidar_ring(ax_g,robot.pos)

        # Trail
        draw_trail(ax_g,robot.trail)

        # Robot
        rr,rc=robot.pos
        draw_robot(ax_g,rc,rr,state,steps[0],robot.direction)

        # Title
        sc={'PLANNING':'#58a6ff','MOVING_TO_FIRE':'#FFEB3B',
            'EXTINGUISHING':'#FF6F00','DONE':'#3fb950'}.get(state,'white')
        ax_g.set_title(
            f'State: {state}   |   Fires left: {len(robot.remaining)}   |   '
            f'Extinguished: {len(robot.extinguished)}/8   |   Step: {steps[0]}',
            color=sc,fontsize=9,fontweight='bold',pad=5)
        ax_g.set_xticks(range(COLS)); ax_g.set_yticks(range(ROWS))
        ax_g.set_xticklabels(range(COLS),fontsize=4,color='#484f58')
        ax_g.set_yticklabels(range(ROWS),fontsize=4,color='#484f58')

        patches=[
            mpatches.Patch(color='#3d3d4a',label='🧱 Wall'),
            mpatches.Patch(color='#FF6F00',label='🔥 Fire'),
            mpatches.Patch(color='#1565C0',label='🤖 Robot (FF)'),
            mpatches.Patch(color='#FFEB3B',label='🟡 A* Path'),
            mpatches.Patch(color='#0e4429',label='🟢 Visited'),
            mpatches.Patch(color='#37474F',label='✅ Extinguished'),
            mpatches.Patch(color='#29B6F6',label='📡 LiDAR range'),
        ]
        ax_g.legend(handles=patches,loc='lower right',fontsize=6,
            framealpha=0.7,labelcolor='white',
            facecolor='#0d1117',edgecolor='#30363d')

        # ── PID PLOT ─────────────────────────────────
        if robot.pid.history:
            e,o=robot.pid.history[-1]
            p_err.append(e); p_out.append(o); p_x.append(steps[0])
        ax_p.cla(); ax_p.set_facecolor('#0d1117')
        ax_p.set_title(' PID Controller',color='#8b949e',fontsize=8,pad=4)
        ax_p.set_xlabel('Step',color='#8b949e',fontsize=7)
        ax_p.set_ylabel('Value',color='#8b949e',fontsize=7)
        ax_p.tick_params(colors='#8b949e',labelsize=6)
        if p_x:
            ax_p.plot(p_x,p_err,'#FF6B6B',label='Error',lw=1.3,alpha=0.9)
            ax_p.plot(p_x,p_out,'#29B6F6',label='PID Output',lw=1.3,alpha=0.9)
            ax_p.fill_between(p_x,p_err,alpha=0.08,color='#FF6B6B')
            ax_p.fill_between(p_x,p_out,alpha=0.08,color='#29B6F6')
            ax_p.axhline(0,color='#30363d',lw=0.7,linestyle='--')
            ax_p.legend(fontsize=7,labelcolor='white',
                facecolor='#161b22',framealpha=0.6,edgecolor='#30363d')
        for sp in ax_p.spines.values(): sp.set_edgecolor('#30363d')

        # ── STATS PANEL ──────────────────────────────
        ax_st.cla(); ax_st.set_facecolor('#0d1117')
        ax_st.set_title('📋 Mission Control',color='#8b949e',fontsize=8,pad=4)
        ax_st.axis('off')
        stats=[
            ('Steps',         str(steps[0]),              '#58a6ff'),
            ('Fires Out',     f"{len(robot.extinguished)}/8",'#3fb950'),
            ('Fires Left',    str(len(robot.remaining)),  '#FF6F00'),
            ('State',         state,                      sc),
            ('Robot Pos',     str(robot.pos),             '#d2a8ff'),
            ('Kp / Ki / Kd', '1.2 / 0.04 / 0.25',       '#ffa657'),
            ('Algorithm',     'A* + PID + Fusion',        '#79c0ff'),
        ]
        for i,(lbl,val,col) in enumerate(stats):
            y=0.93-i*0.134
            ax_st.text(0.03,y,lbl+':',fontsize=8,color='#8b949e',
                transform=ax_st.transAxes,va='top')
            ax_st.text(0.97,y,val,fontsize=8,fontweight='bold',
                color=col,transform=ax_st.transAxes,va='top',ha='right')
            if i<len(stats)-1:
                ax_st.plot([0,1],[y-0.05,y-0.05],color='#21262d',
                    lw=0.5,transform=ax_st.transAxes,clip_on=False)
        for sp in ax_st.spines.values(): sp.set_edgecolor('#30363d')

        fig.texts.clear()
        fig.text(0.30,0.005,
            'Maham Qaiser (01-136232-025)  |  Imama Tufail (01-136232-021)  |  '
            'BS(AI) 6(A)  |  Dr. Adil Khan',
            ha='center',color='#484f58',fontsize=7)

        if state=="DONE":
            ani.event_source.stop()
            ax_g.set_title('  ALL FIRES EXTINGUISHED — MISSION COMPLETE!  ',
                color='#3fb950',fontsize=11,fontweight='bold')
            print("\n"+"="*50)
            print("   MISSION COMPLETE!")
            print(f"  Total steps   : {steps[0]}")
            print(f"  Fires put out : {len(robot.extinguished)}/8")
            print("="*50)

    ani=animation.FuncAnimation(fig,frame,frames=600,interval=120,repeat=False)
    plt.show()

if __name__=="__main__":
    print("="*50)
    print("   FIREFIGHTER ROBOT — Animated Version")
    print("  Maham Qaiser  |  Imama Tufail  |  BS(AI) 6(A)")
    print("="*50)
    run()
