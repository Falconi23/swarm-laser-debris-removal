"""
Three-drone swarm for laser-based active debris removal, trained with one shared PPO policy.

ADRSwarmEnv is a PettingZoo parallel env: two-body translation in ECI, rigid-body
attitude with gravity-gradient and drag torques, rocket-equation fuel use and a
simulated star tracker. Relative quantities are in the LVLH frame of the base
(r_ost). Phases: 1.0 approach, 2.0 laser engagement, 2.5 return, 3.0 done.

python swarm.py trains with fictitious self-play (checkpoints in ./swarm_checkpoints/)
and renders a deterministic test rollout to swarm_mission.gif.
"""

import os, gc, time, glob, random, resource, warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from skyfield.api import load
from skyfield.data import hipparcos

warnings.filterwarnings("ignore", category=UserWarning)

def q_to_rot_mat(q):
    """Scalar-first unit quaternion [w,x,y,z] -> rotation matrix (body -> ECI)."""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*qy**2-2*qz**2,  2*qx*qy-2*qw*qz,   2*qx*qz+2*qw*qy],
        [2*qx*qy+2*qw*qz,    1-2*qx**2-2*qz**2,  2*qy*qz-2*qw*qx],
        [2*qx*qz-2*qw*qy,    2*qy*qz+2*qw*qx,    1-2*qx**2-2*qy**2]
    ], dtype=np.float64)

def rot_mat_to_q(R):
    """Rotation matrix -> unit quaternion (Shepperd's method, branch on largest diagonal term)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.)*2.; qw,qx = .25*S,(R[2,1]-R[1,2])/S
        qy,qz = (R[0,2]-R[2,0])/S,(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S = np.sqrt(1.+R[0,0]-R[1,1]-R[2,2])*2.; qw,qx = (R[2,1]-R[1,2])/S,.25*S
        qy,qz = (R[0,1]+R[1,0])/S,(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S = np.sqrt(1.+R[1,1]-R[0,0]-R[2,2])*2.; qw,qx = (R[0,2]-R[2,0])/S,(R[0,1]+R[1,0])/S
        qy,qz = .25*S,(R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.+R[2,2]-R[0,0]-R[1,1])*2.; qw,qx = (R[1,0]-R[0,1])/S,(R[0,2]+R[2,0])/S
        qy,qz = (R[1,2]+R[2,1])/S,.25*S
    q = np.array([qw,qx,qy,qz]); return q/np.linalg.norm(q)

def q_dot(q, omega):
    """Quaternion kinematics: q_dot = 1/2 * Omega(omega) * q, omega in body frame."""
    qw,qx,qy,qz = q; wx,wy,wz = omega
    W = np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])
    return 0.5*(W@q)

def rk4_step(mu, r, v, dt):
    """One RK4 step of the two-body problem r_ddot = -mu r / |r|^3."""
    def d(r_,v_): return v_,-mu/np.linalg.norm(r_)**3*r_
    k1r,k1v = d(r,v); k2r,k2v = d(r+.5*dt*k1r,v+.5*dt*k1v)
    k3r,k3v = d(r+.5*dt*k2r,v+.5*dt*k2v); k4r,k4v = d(r+dt*k3r,v+dt*k3v)
    return r+(dt/6.)*(k1r+2*k2r+2*k3r+k4r), v+(dt/6.)*(k1v+2*k2v+2*k3v+k4v)

def circular_state(r_orb, angle, mu):
    """Position/velocity on an equatorial circular orbit of radius r_orb at a given angle."""
    c,s = np.cos(angle),np.sin(angle); vc = np.sqrt(mu/r_orb)
    return r_orb*np.array([c,s,0.]), vc*np.array([-s,c,0.])

class StarTrackerModel:
    """Star tracker: Hipparcos stars (mag <= 4.5), 15 deg FOV on body +Y, 35 deg Sun exclusion.

    With two or more stars in view the attitude comes from TRIAD on noisy star
    vectors, otherwise from a noisier direct quaternion measurement.
    """
    def __init__(self):
        print("[StarTracker] Loading ephemeris + Hipparcos catalog...")
        ts = load.timescale()
        t  = ts.utc(2024, 5, 15, 12, 0, 0)
        eph = load('de421.bsp')
        sun_pos = eph['earth'].at(t).observe(eph['sun']).position.km
        self.sun_dir = sun_pos / np.linalg.norm(sun_pos)
        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        bright = df[df['magnitude'] <= 4.5]
        ra  = np.radians(bright['ra_degrees'].values)
        dec = np.radians(bright['dec_degrees'].values)
        self.catalog = np.column_stack((np.cos(dec)*np.cos(ra),
                                        np.cos(dec)*np.sin(ra),
                                        np.sin(dec)))
        self.fov_cos = np.cos(np.radians(15.))
        self.sun_cos = np.cos(np.radians(35.))
        print(f"[StarTracker] {len(self.catalog)} stars (mag<=4.5).")

    def measure(self, q_true):
        """Return (measured quaternion, measured ECI->body rotation matrix)."""
        R_b2e = q_to_rot_mat(q_true); R_e2b = R_b2e.T
        bore  = R_b2e @ np.array([0.,1.,0.])
        blind = np.dot(self.sun_dir, bore) > self.sun_cos
        dots  = np.dot(self.catalog, bore)
        stars = self.catalog[dots > self.fov_cos]
        if len(stars) >= 2 and not blind:
            # TRIAD: triads from two stars in ECI and body, R_e2b = M_body @ M_ref.T
            s1_e,s2_e = stars[0],stars[1]
            s1 = R_e2b@s1_e + np.random.normal(0,.002,3)
            s2 = R_e2b@s2_e + np.random.normal(0,.002,3)
            s1/=np.linalg.norm(s1); s2/=np.linalg.norm(s2)
            t1=s1_e; t2=np.cross(s1_e,s2_e); t2/=np.linalg.norm(t2); t3=np.cross(t1,t2)
            M_ref=np.column_stack((t1,t2,t3))
            s2o=np.cross(s1,s2); n=np.linalg.norm(s2o)
            s2o=s2o/n if n>1e-6 else np.array([0.,1.,0.])
            s3=np.cross(s1,s2o); M_body=np.column_stack((s1,s2o,s3))
            R_e2b_m = M_body@M_ref.T; q_m = rot_mat_to_q(R_e2b_m.T)
        else:
            # sun-blinded or too few stars: coarse estimate from the true attitude
            q_m = q_true+np.random.normal(0,.02,4); q_m/=np.linalg.norm(q_m)
            R_e2b_m = q_to_rot_mat(q_m).T
        return q_m, R_e2b_m


class DroneState:
    def __init__(self, idx: int, r0: np.ndarray, v0: np.ndarray, mass: float, base_I: np.ndarray):
        self.idx          = idx
        self.r            = r0.copy()
        self.v            = v0.copy()
        self.mass         = mass
        self.I            = base_I.copy()
        self.q            = np.array([1.,0.,0.,0.])
        self.omega        = np.zeros(3)
        self.prev_dist_goal = 20.0
        self.dm_step      = 0.0   # propellant used in the last step, for the fuel reward
        self.ep_start_mass = mass


class ADRSwarmEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "adr_swarm_v0"}

    # SI units throughout
    MU           = 3.986004418e14
    RE           = 6_371e3
    DT           = 1.0
    MAX_THRUST   = 30.0   # N, phase 1.0 (approach)
    MAX_TORQUE   = 2.0    # Nm
    MED_THRUST   = 10.0   # N, phase 2.5 (return)
    MED_TORQUE   = 0.5    # Nm
    FINE_THRUST  = 3.0    # N, phase 2.0 (laser)
    FINE_TORQUE  = 0.2    # Nm
    DRONE_MASS   = 150.0
    DRY_MASS     = 110.0
    ISP          = 220.0
    G0           = 9.81
    RHO          = 1e-13  # atmospheric density at ~700 km [kg/m^3]
    CD           = 2.2
    DT_RK4       = 5.0
    R_PARK       = RE + 600e3
    R_TARGET     = RE + 700e3
    OST_R        = 2.0    # keep-out radius around the base [m]
    GOAL_RADIUS  = 1.0
    MAX_STEPS    = 3000
    FORM_SIDE    = 1.5    # side of the equilateral formation [m]
    N_DRONES     = 3
    OBS_DIM      = 28
    ACT_DIM      = [3]*6      # {-max, 0, +max} on 3 forces + 3 torques

    LASER_STANDOFF  = 3.0   # formation centroid distance from debris [m]
    LASER_RANGE     = 8.0   # [m]
    LASER_CONE_COS  = 0.97  # ~14 deg half-angle
    LASER_DPS       = 0.5   # per drone, at zero pointing error
    DEBRIS_HP       = 200.0
    FIRE_RADIUS     = 2.0   # goal tolerance to enter phase 2.0 [m]

    # equilateral triangle in the R-bar/H-bar plane, centred on the centroid
    _FORM_OFFSETS_LVLH = np.array([
        [ 0.0,  0.0,  FORM_SIDE / np.sqrt(3)],
        [-FORM_SIDE/2, 0.0, -FORM_SIDE/(2*np.sqrt(3))],
        [ FORM_SIDE/2, 0.0, -FORM_SIDE/(2*np.sqrt(3))],
    ], dtype=np.float64)

    def __init__(self, star_tracker: StarTrackerModel = None, render_mode=None):
        super().__init__()
        self.render_mode  = render_mode
        self.star_tracker = star_tracker
        # start mass already pays for the Hohmann transfer from the parking orbit
        a_t = .5*(self.R_PARK+self.R_TARGET)
        v1  = np.sqrt(self.MU/self.R_PARK); v2 = np.sqrt(self.MU/self.R_TARGET)
        dv1 = np.sqrt(self.MU*(2/self.R_PARK-1/a_t))-v1
        dv2 = v2-np.sqrt(self.MU*(2/self.R_TARGET-1/a_t))
        dv_tot = abs(dv1)+abs(dv2)
        self.START_MASS = self.DRONE_MASS * np.exp(-dv_tot/(self.ISP*self.G0))
        # uniform box [m]; inertia uses wet mass and is not updated as fuel burns
        self.dims = np.array([0.5, 0.8, 0.5])
        Ix = (1/12)*self.DRONE_MASS*(self.dims[1]**2+self.dims[2]**2)
        Iy = (1/12)*self.DRONE_MASS*(self.dims[0]**2+self.dims[2]**2)
        Iz = (1/12)*self.DRONE_MASS*(self.dims[0]**2+self.dims[1]**2)
        self.base_I = np.array([Ix,Iy,Iz])
        self.cp_offset = np.array([0.,0.1,0.])   # CoP - CoM, gives the drag torque arm
        self.possible_agents = [f"drone_{i}" for i in range(self.N_DRONES)]
        self.agents = list(self.possible_agents)
        obs_high = np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_spaces = {
            a: spaces.Box(-obs_high, obs_high, dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.MultiDiscrete(self.ACT_DIM)
            for a in self.possible_agents
        }
        self._traj = {a: {"out": ([],[],[]), "back": ([],[],[])}
                      for a in self.possible_agents}

    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent):      return self.action_spaces[agent]

    def _gravity(self, r): return -self.MU/np.linalg.norm(r)**3*r
    def _lvlh(self, r, v):
        """LVLH -> ECI rotation matrix (columns: radial, along-track, orbit normal)."""
        rh=r/np.linalg.norm(r); hv=np.cross(r,v); hh=hv/np.linalg.norm(hv)
        return np.column_stack((rh,np.cross(hh,rh),hh))

    def _disturbances(self, d: DroneState):
        """Disturbance torque in body frame: gravity gradient + aerodynamic drag."""
        R_b2e=q_to_rot_mat(d.q); R_e2b=R_b2e.T
        rb=R_e2b@d.r; I_mat=np.diag(d.I)
        tau_gg=(3*self.MU/np.linalg.norm(rb)**5)*np.cross(rb,I_mat@rb)
        vb=R_e2b@d.v; vm=np.linalg.norm(vb); vh=vb/(vm+1e-9)
        area=(self.dims[1]*self.dims[2]*abs(vh[0])+
              self.dims[0]*self.dims[2]*abs(vh[1])+
              self.dims[0]*self.dims[1]*abs(vh[2]))
        f_drag=-vh*.5*self.RHO*vm**2*self.CD*area
        return tau_gg+np.cross(self.cp_offset,f_drag)

    def _individual_goal(self, drone_idx: int, Q_e2l, L2E) -> np.ndarray:
        """Formation vertex of one drone: around the standoff point until the debris is gone, then the base waypoint."""
        offset_lvlh = self._FORM_OFFSETS_LVLH[drone_idx]
        if self.phase < 2.5:
            centroid_goal = self.r_debris + L2E @ np.array([0., -self.LASER_STANDOFF, 0.])
        else:
            centroid_goal = self.r_ost + L2E @ np.array([0., -8., 0.])
        return centroid_goal + L2E @ offset_lvlh

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.agents       = list(self.possible_agents)
        self.phase        = 1.0
        self.current_step = 0
        self.mission_done = False

        self.debris_hp        = self.DEBRIS_HP
        self.debris_destroyed = False
        self._firing          = [False]*self.N_DRONES
        self._aim             = [0.0]*self.N_DRONES
        self._destroy_lvlh    = None

        if self.star_tracker is None:
            self.star_tracker = StarTrackerModel()
        # base on a circular orbit, debris 8 m ahead on V-bar, swarm 8 m behind
        r0 = self.R_TARGET; v0 = np.sqrt(self.MU/r0)
        self.r_ost = np.array([r0,0.,0.]); self.v_ost = np.array([0.,v0,0.])
        Q0 = self._lvlh(self.r_ost,self.v_ost).T; L0 = Q0.T
        self.r_debris = self.r_ost + L0@np.array([0.,8.,0.])
        self.v_debris = np.array([0.,v0,0.])

        self.drones: list[DroneState] = []
        for i in range(self.N_DRONES):
            offset = self._FORM_OFFSETS_LVLH[i]
            r_i = self.r_ost + L0@(np.array([0.,-8.,0.])+offset)
            v_i = np.array([0.,v0,0.])
            d = DroneState(i, r_i, v_i, self.START_MASS, self.base_I)
            d.prev_dist_goal = np.linalg.norm(r_i - self._individual_goal(i,Q0,L0))
            self.drones.append(d)

        for i,a in enumerate(self.possible_agents):
            lp = Q0@(self.drones[i].r-self.r_ost)
            self._traj[a] = {"out":([lp[0]],[lp[1]],[lp[2]]), "back":([],[],[])}

        obs = self._build_all_obs()
        return obs, {a:{} for a in self.agents}

    def _sensor_single(self, d: DroneState, Q_e2l):
        R_b2e = q_to_rot_mat(d.q); R_e2b = R_b2e.T
        q_m, R_e2b_m = self.star_tracker.measure(d.q)
        omega_m = d.omega + np.random.normal(0,.002,3)
        return q_m, R_e2b_m, omega_m, R_b2e, R_e2b

    def _build_all_obs(self) -> dict:
        """Noisy 28-D observation for each drone, normalised to [-1, 1]."""
        Q_e2l = self._lvlh(self.r_ost,self.v_ost).T
        L2E   = Q_e2l.T
        obs = {}
        integ = np.clip(self.debris_hp/self.DEBRIS_HP, 0., 1.)

        for i, a in enumerate(self.possible_agents):
            d  = self.drones[i]
            j  = (i+1) % self.N_DRONES
            k  = (i+2) % self.N_DRONES
            dj = self.drones[j]; dk = self.drones[k]

            q_m, R_e2b_m, omega_m, R_b2e, R_e2b = self._sensor_single(d, Q_e2l)

            r_goal_i = self._individual_goal(i, Q_e2l, L2E)
            rel_goal  = r_goal_i - d.r
            nf = 1.+np.linalg.norm(rel_goal)/50.
            m_rg = rel_goal + np.random.normal(0,.02*nf,3)
            m_vg = (self.v_ost - d.v) + np.random.normal(0,.005,3)
            rg_body = R_e2b_m @ m_rg
            vg_body = R_e2b_m @ m_vg

            rj_lvlh  = Q_e2l@(dj.r - d.r) + np.random.normal(0,.05,3)
            rk_lvlh  = Q_e2l@(dk.r - d.r) + np.random.normal(0,.05,3)
            ost_lvlh = Q_e2l@(self.r_ost - d.r) + np.random.normal(0,.05,3)

            aim_vec = (self.r_debris - d.r) if not self.debris_destroyed else (r_goal_i - d.r)
            aim_b   = R_e2b_m @ (aim_vec / (np.linalg.norm(aim_vec) + 1e-9))

            vec = np.concatenate([
                np.clip(rg_body/20., -1.,1.),       # [0:3]
                np.clip(vg_body/5.,  -1.,1.),       # [3:6]
                np.clip(rj_lvlh/10., -1.,1.),       # [6:9]
                np.clip(rk_lvlh/10., -1.,1.),       # [9:12]
                np.clip(ost_lvlh/15.,-1.,1.),       # [12:15]
                np.clip(omega_m/0.5, -1.,1.),       # [15:18]
                np.clip(q_m,         -1.,1.),       # [18:22]
                [np.clip(d.mass/self.DRONE_MASS, 0., 1.)],  # [22]
                [self.phase/3.0],                   # [23]
                np.clip(aim_b,       -1.,1.),       # [24:27]
                [integ],                            # [27]
            ]).astype(np.float32)
            obs[a] = vec
        return obs

    def step(self, actions: dict):
        self.current_step += 1
        # actuator authority depends on phase: coarse approach, fine while firing, medium return
        if self.phase == 2.0:
            thr, trq = self.FINE_THRUST, self.FINE_TORQUE
        elif self.phase == 2.5:
            thr, trq = self.MED_THRUST,  self.MED_TORQUE
        else:
            thr, trq = self.MAX_THRUST,  self.MAX_TORQUE
        thrust_map = {0: -thr, 1: 0., 2: thr}
        torque_map = {0: -trq, 1: 0., 2: trq}

        Q_e2l = self._lvlh(self.r_ost,self.v_ost).T
        L2E   = Q_e2l.T
        goals = [self._individual_goal(i,Q_e2l,L2E) for i in range(self.N_DRONES)]

        rewards  = {a:0. for a in self.possible_agents}
        term_all = False

        for i, a in enumerate(self.possible_agents):
            d  = self.drones[i]
            ac = actions.get(a, np.full(6,1,dtype=int))

            F_body = np.array([thrust_map[ac[0]], thrust_map[ac[1]], thrust_map[ac[2]]])
            T_body = np.array([torque_map[ac[3]], torque_map[ac[4]], torque_map[ac[5]]])

            # mdot = |F|/(Isp g0); torques are charged as if they were thrust too
            dm = (np.sum(np.abs(F_body))+np.sum(np.abs(T_body)))/(self.ISP*self.G0)*self.DT
            d.dm_step = dm
            d.mass = max(d.mass-dm, self.DRY_MASS)

            # Euler's equations, explicit Euler integration
            tau_dist = self._disturbances(d)
            T_tot    = T_body + tau_dist
            w_dot    = (T_tot - np.cross(d.omega,d.I*d.omega))/d.I
            d.omega  = np.clip(d.omega+w_dot*self.DT,-0.5,0.5)
            d.q     += q_dot(d.q,d.omega)*self.DT
            d.q     /= np.linalg.norm(d.q)

            # thrust is commanded in body frame
            R_b2e = q_to_rot_mat(d.q)
            F_eci = R_b2e@F_body
            d.v += (self._gravity(d.r)+F_eci/d.mass)*self.DT
            d.r += d.v*self.DT

        self.v_ost += self._gravity(self.r_ost)*self.DT
        self.r_ost += self.v_ost*self.DT

        if not self.debris_destroyed:
            self.r_debris = self.r_ost + L2E@np.array([0.,8.,0.])
            self.v_debris = self.v_ost.copy()

        Q_e2l = self._lvlh(self.r_ost,self.v_ost).T
        L2E   = Q_e2l.T
        goals = [self._individual_goal(i,Q_e2l,L2E) for i in range(self.N_DRONES)]

        self._firing = [False]*self.N_DRONES
        self._aim    = [0.0]*self.N_DRONES
        total_dmg    = 0.0
        # a drone fires when the debris is in range and inside its cone;
        # damage scales with cos(pointing error) and sums over drones
        if self.phase == 2.0 and not self.debris_destroyed:
            for i, d in enumerate(self.drones):
                aim_vec  = self.r_debris - d.r
                dist     = np.linalg.norm(aim_vec) + 1e-9
                nose_eci = q_to_rot_mat(d.q) @ np.array([0.,1.,0.])
                cos_aim  = float(np.dot(nose_eci, aim_vec/dist))
                self._aim[i] = cos_aim
                if dist <= self.LASER_RANGE and cos_aim >= self.LASER_CONE_COS:
                    self._firing[i] = True
                    total_dmg += self.LASER_DPS * cos_aim
            self.debris_hp = max(0., self.debris_hp - total_dmg*self.DT)

        out_of_bounds_any = False
        collision_any     = False
        fuel_empty_any    = False

        centroid_current   = np.mean([d.r for d in self.drones], axis=0)
        dist_centroid_ost  = np.linalg.norm(centroid_current - self.r_ost)
        dists_to_ost       = [np.linalg.norm(d.r - self.r_ost) for d in self.drones]
        min_swarm_dist_ost = min(dists_to_ost + [dist_centroid_ost])

        for i, a in enumerate(self.possible_agents):
            d = self.drones[i]
            lp = Q_e2l@(d.r-self.r_ost)
            key = "out" if self.phase < 2.5 else "back"
            self._traj[a][key][0].append(lp[0])
            self._traj[a][key][1].append(lp[1])
            self._traj[a][key][2].append(lp[2])

            dist_goal = np.linalg.norm(d.r - goals[i])
            dist_ost  = np.linalg.norm(d.r - self.r_ost)
            vel_mag   = np.linalg.norm(d.v - self.v_ost)
            omega_mag = np.linalg.norm(d.omega)
            R_e2b     = q_to_rot_mat(d.q).T

            # navigation: small distance penalty, progress term does most of the shaping
            r_i  = -0.003*dist_goal
            r_i += (d.prev_dist_goal - dist_goal)*0.45
            d.prev_dist_goal = dist_goal

            r_i -= d.dm_step * 2.5
            if d.mass <= self.DRY_MASS: fuel_empty_any = True

            # pointing: body +Y is the laser axis; outside phase 2.0 a weaker term
            # keeps it on the debris (or the goal once the debris is gone)
            if self.phase == 2.0 and not self.debris_destroyed:
                aim_vec_pt  = self.r_debris - d.r
                vec_body_pt = R_e2b @ aim_vec_pt
                dist_pt     = max(np.linalg.norm(aim_vec_pt), 1e-6)
                point_acc   = (vec_body_pt / dist_pt)[1]
                r_i += point_acc * 0.08

                r_i += max(0., self._aim[i]) * 0.05
                if self._firing[i]:
                    r_i += 0.08
                    my_damage = self.LASER_DPS * self._aim[i] * self.DT
                    r_i += my_damage * 0.04
            else:
                if not self.debris_destroyed:
                    vec_target = self.r_debris - d.r
                else:
                    vec_target = goals[i] - d.r
                vec_body   = R_e2b @ vec_target
                dist_t_vec = max(np.linalg.norm(vec_target), 1e-6)
                point_acc  = (vec_body/dist_t_vec)[1]
                r_i += point_acc * 0.015

            # relative-velocity damping only near the goal, wider while firing
            damp_radius = 4.0 if self.phase == 2.0 else self.GOAL_RADIUS * 2.0
            if dist_goal < damp_radius:
                proximity = max(0., 1. - dist_goal/damp_radius)
                r_i -= vel_mag * proximity * 0.25
            r_i -= omega_mag * 0.15

            # potential field: exponential repulsion within 2x the keep-out radius
            if dist_ost < self.OST_R*2.0:
                p = (self.OST_R*2.0 - dist_ost)/self.OST_R
                r_i -= 0.08*(np.exp(3.0*p)-1.0)

            # formation cohesion: pairwise distances should equal FORM_SIDE
            j,k = (i+1)%3,(i+2)%3
            dev_j = abs(np.linalg.norm(d.r-self.drones[j].r) - self.FORM_SIDE)
            dev_k = abs(np.linalg.norm(d.r-self.drones[k].r) - self.FORM_SIDE)
            r_i -= 0.03*(dev_j+dev_k)

            # terminal penalties (box, keep-out zone), then a soft one below 0.8 m separation
            if (np.abs(lp)>25).any(): out_of_bounds_any=True; r_i-=50.
            if dist_ost <= self.OST_R: collision_any=True; r_i-=50.
            if self.phase>=2.0 and dist_centroid_ost<=self.OST_R:
                collision_any=True; r_i-=50.
            for jj in [j,k]:
                d_ij=np.linalg.norm(d.r-self.drones[jj].r)
                if d_ij<0.8: r_i-=0.5*(0.8-d_ij)**2

            rewards[a]=float(r_i)

        # phase transitions; prev_dist_goal is reset because the goals jump
        if self.phase == 1.0:
            in_pos = all(
                np.linalg.norm(self.drones[i].r - goals[i]) < self.FIRE_RADIUS and
                np.linalg.norm(self.drones[i].v - self.v_ost) < 0.3
                for i in range(self.N_DRONES))
            if in_pos:
                self.phase = 2.0
                for a in self.possible_agents: rewards[a] += 25.
                for i,d in enumerate(self.drones):
                    d.prev_dist_goal = np.linalg.norm(d.r-self._individual_goal(i,Q_e2l,L2E))

        elif self.phase == 2.0:
            if self.debris_hp <= 0. and not self.debris_destroyed:
                self.debris_destroyed = True
                self._destroy_lvlh = Q_e2l@(self.r_debris - self.r_ost)
                self.phase = 2.5
                print(f"[{self.current_step}] debris destroyed, returning to base")
                for a in self.possible_agents: rewards[a] += 60.
                for i,d in enumerate(self.drones):
                    d.prev_dist_goal = np.linalg.norm(d.r-self._individual_goal(i,Q_e2l,L2E))

        elif self.phase == 2.5:
            # MED_THRUST gives ~0.077 m/s per step, so a 0.5 m tolerance is reachable
            centroid     = np.mean([d.r for d in self.drones], axis=0)
            base_goal    = self.r_ost + L2E @ np.array([0., -8., 0.])
            vel_centroid = np.linalg.norm(
                np.mean([d.v for d in self.drones], axis=0) - self.v_ost)
            omega_mean   = float(np.mean([np.linalg.norm(d.omega) for d in self.drones]))
            base_ok = (np.linalg.norm(centroid - base_goal) < 0.5 and
                       vel_centroid < 0.2 and
                       omega_mean   < 0.15)
            if base_ok:
                print(f"[{self.current_step}] base reached, mission complete")
                self.phase = 3.0; self.mission_done = True; term_all = True
                for a in self.possible_agents: rewards[a] += 80.

        if out_of_bounds_any or collision_any or fuel_empty_any: term_all=True
        truncated = self.current_step >= self.MAX_STEPS
        if term_all or truncated: self.agents=[]

        terminations = {a: term_all  for a in self.possible_agents}
        truncations  = {a: truncated for a in self.possible_agents}
        infos = {a: {"phase": self.phase, "debris_hp": self.debris_hp}
                 for a in self.possible_agents}

        obs = self._build_all_obs()
        return obs, rewards, terminations, truncations, infos

    def render(self):
        """3-D LVLH view of the mission (MATLAB-like light style)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Sans'

        MATLAB = {
            'bg':       'white',
            'pane':     (0.94, 0.94, 0.96, 1.0),
            'edge':     '#b0b0b0',
            'grid':     '#d0d0d0',
            'text':     '#1a1a1a',
            'drone':    ['#0072BD', '#D95319', '#77AC30'],   # MATLAB blue / orange / green
            'traj_out': ['#0072BD', '#D95319', '#77AC30'],
            'traj_back':['#4DBEEE', '#EDB120', '#A2142F'],
            'nose':     '#7E2F8E',
            'obstacle': '#D95319',
            'base':     '#77AC30',
            'laser':    '#FF0000',
            'debris':   '#EDB120',
            'burst':    '#D95319',
            'form':     '#888888',
        }

        fig = plt.figure(figsize=(13, 9.5), facecolor=MATLAB['bg'])
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(MATLAB['bg'])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(MATLAB['pane'])
            axis.pane.set_edgecolor(MATLAB['edge'])
            axis.pane.set_linewidth(0.8)
        ax.grid(True, color=MATLAB['grid'], linewidth=0.6, linestyle='-')

        Q_e2l = self._lvlh(self.r_ost, self.v_ost).T

        # base keep-out zone
        u, vv = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        xs = self.OST_R*np.cos(u)*np.sin(vv)
        ys = self.OST_R*np.sin(u)*np.sin(vv)
        zs = self.OST_R*np.cos(vv)
        ax.plot_surface(xs, ys, zs, color=MATLAB['obstacle'],
                        alpha=0.18, linewidth=0, antialiased=True)
        ax.plot_wireframe(xs, ys, zs, color=MATLAB['obstacle'],
                          alpha=0.35, linewidth=0.4, rstride=4, cstride=4)

        ax.scatter(0, -8., 0, color=MATLAB['base'], s=180, marker='s',
                   edgecolors='#333333', linewidths=1.2,
                   depthshade=False, label='Base', zorder=20)
        theta = np.linspace(0, 2*np.pi, 60)
        ax.plot(4.*np.cos(theta), -8.*np.ones(60), 4.*np.sin(theta),
                color=MATLAB['base'], lw=0.8, alpha=0.4, ls='--')

        for i, a in enumerate(self.possible_agents):
            ox, oy, oz = self._traj[a]["out"]
            bx, by, bz = self._traj[a]["back"]
            if ox:
                ax.plot(ox, oy, oz, color=MATLAB['traj_out'][i],
                        lw=1.4, alpha=0.75,
                        label=f'D{i} outbound' if i == 0 else f'D{i}')
            if bx:
                ax.plot(bx, by, bz, color=MATLAB['traj_back'][i],
                        lw=1.4, alpha=0.65, ls='--',
                        label=f'D{i} return' if i == 0 else "")

        # drone body plus a nose stick along body +Y (laser axis)
        d_half = self.dims / 2.
        verts_body = np.array([
            [-d_half[0],-d_half[1],-d_half[2]], [ d_half[0],-d_half[1],-d_half[2]],
            [ d_half[0], d_half[1],-d_half[2]], [-d_half[0], d_half[1],-d_half[2]],
            [-d_half[0],-d_half[1], d_half[2]], [ d_half[0],-d_half[1], d_half[2]],
            [ d_half[0], d_half[1], d_half[2]], [-d_half[0], d_half[1], d_half[2]]])

        debris_lvlh    = Q_e2l @ (self.r_debris - self.r_ost)
        nose_tips_lvlh = []

        for i, d in enumerate(self.drones):
            R_b2e = q_to_rot_mat(d.q)
            lp    = Q_e2l @ (d.r - self.r_ost)

            def b2l(p_body, _r=d.r, _R=R_b2e):
                return Q_e2l @ (_r + _R @ p_body - self.r_ost)

            vp = np.array([b2l(vb) for vb in verts_body])
            faces = [
                [vp[0],vp[1],vp[2],vp[3]], [vp[4],vp[5],vp[6],vp[7]],
                [vp[0],vp[1],vp[5],vp[4]], [vp[2],vp[3],vp[7],vp[6]],
                [vp[0],vp[3],vp[7],vp[4]], [vp[1],vp[2],vp[6],vp[5]]]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=MATLAB['drone'][i],
                edgecolors='#333333', linewidths=0.6, alpha=0.88))

            nose  = b2l(np.array([0., d_half[1]+0.7, 0.]))
            nbase = b2l(np.array([0., d_half[1],      0.]))
            ax.plot([nbase[0],nose[0]], [nbase[1],nose[1]], [nbase[2],nose[2]],
                    color=MATLAB['nose'], lw=2.0, solid_capstyle='round')
            nose_tips_lvlh.append(b2l(np.array([0., d_half[1], 0.])))

            ax.text(lp[0], lp[1], lp[2]+1.0, f'D{i}',
                    color=MATLAB['text'], fontsize=8, fontweight='bold',
                    ha='center', va='bottom')

        beam_drawn = False
        if self.phase == 2.0 and not self.debris_destroyed:
            for i in range(self.N_DRONES):
                if not self._firing[i]: continue
                p0, p1 = nose_tips_lvlh[i], debris_lvlh
                ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                        color=MATLAB['laser'], lw=5, alpha=0.15)
                ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                        color=MATLAB['laser'], lw=1.5, alpha=0.90,
                        label='Laser' if not beam_drawn else "")
                beam_drawn = True

        if not self.debris_destroyed:
            integ = max(0.05, self.debris_hp / self.DEBRIS_HP)
            size  = 80 + 200*integ
            cmap_val = plt.cm.RdYlGn(integ)
            ax.scatter(debris_lvlh[0], debris_lvlh[1], debris_lvlh[2],
                       color=cmap_val, s=size, marker='*',
                       edgecolors='#333333', linewidths=0.6,
                       depthshade=False, label=f'Debris ({integ*100:.0f}%)', zorder=18)
        else:
            c = self._destroy_lvlh if self._destroy_lvlh is not None else debris_lvlh
            rng = np.random.RandomState(0)
            burst = c + rng.normal(0, 1.2, (60, 3))
            sizes = rng.uniform(8, 30, 60)
            ax.scatter(burst[:,0], burst[:,1], burst[:,2],
                       c=MATLAB['burst'], s=sizes, alpha=0.55, marker='*',
                       edgecolors='none', label='Debris destroyed', zorder=16)

        lps = [Q_e2l @ (d.r - self.r_ost) for d in self.drones]
        for ii in range(3):
            jj = (ii+1) % 3
            ax.plot([lps[ii][0],lps[jj][0]], [lps[ii][1],lps[jj][1]],
                    [lps[ii][2],lps[jj][2]],
                    color=MATLAB['form'], lw=0.9, ls=':', alpha=0.6)

        ax.set_xlim([-12, 12]); ax.set_ylim([-12, 12]); ax.set_zlim([-8, 8])
        ax.set_xlabel('R-Bar  X [m]', color=MATLAB['text'], labelpad=8, fontsize=10)
        ax.set_ylabel('V-Bar  Y [m]', color=MATLAB['text'], labelpad=8, fontsize=10)
        ax.set_zlabel('H-Bar  Z [m]', color=MATLAB['text'], labelpad=8, fontsize=10)
        ax.tick_params(colors=MATLAB['text'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(MATLAB['edge'])

        phase_labels = {1.0:'Approach', 2.0:'Laser Engagement',
                        2.5:'Return to Base', 3.0:'Mission Complete'}
        phase_str = phase_labels.get(self.phase, f'{self.phase:.1f}')
        integ_pct = 100. * self.debris_hp / self.DEBRIS_HP
        fuel_used = sum((d.ep_start_mass - d.mass) for d in self.drones)

        ax.set_title(
            f'ADR Swarm — Laser Ablation   |   Phase: {phase_str}   |   '
            f'Debris: {integ_pct:.0f}%   |   ΔMass: {fuel_used:.1f} kg   |   '
            f'Step: {self.current_step}',
            color=MATLAB['text'], fontsize=10, fontweight='bold', pad=12)

        handles, labels = ax.get_legend_handles_labels()
        for i in range(self.N_DRONES):
            handles.append(mpatches.Patch(color=MATLAB['drone'][i], label=f'Drone {i}'))
            labels.append(f'Drone {i}')
        ax.legend(handles, labels,
                  loc='upper left', fontsize=7.5,
                  framealpha=0.9, edgecolor='#aaaaaa',
                  facecolor='white', labelcolor=MATLAB['text'])

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            plt.close(fig); return img
        return None

    def close(self): pass


_PARTNER_RELOAD_INTERVAL = 150_000   # partner calls between reloads of the shared policy

class SwarmAgentWrapper(gym.Env):
    """Single-agent Gymnasium view of the swarm for fictitious self-play.

    One drone (rotating every episode) is controlled by the PPO learner; the
    other two act with the latest saved copy of the shared policy (random
    actions until a checkpoint exists), reloaded periodically from disk.
    """
    OBS_DIM = ADRSwarmEnv.OBS_DIM
    ACT_DIM = ADRSwarmEnv.ACT_DIM

    def __init__(self, shared_model_path: str | None = None, seed: int = 0):
        super().__init__()
        self.shared_model_path = shared_model_path
        self._seed             = seed
        self._model            = None
        self._model_loaded     = False
        self._env: ADRSwarmEnv | None = None
        self._st: StarTrackerModel | None = None
        self._trainee_idx      = seed % ADRSwarmEnv.N_DRONES
        self._ep_count         = 0
        self._partner_calls    = 0
        self.observation_space = spaces.Box(
            low=-1., high=1., shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(self.ACT_DIM)
        self._last_partner_obs: dict = {}

    def _lazy_init(self):
        if self._st is None:  self._st  = StarTrackerModel()
        if self._env is None: self._env = ADRSwarmEnv(star_tracker=self._st)

    def _load_model(self):
        if self._model_loaded: return
        self._model_loaded = True
        if self.shared_model_path and os.path.exists(self.shared_model_path):
            obs_sp = spaces.Box(low=-1.,high=1.,shape=(self.OBS_DIM,),dtype=np.float32)
            act_sp = spaces.MultiDiscrete(self.ACT_DIM)
            self._model = PPO.load(
                self.shared_model_path, device="cpu",
                custom_objects={"observation_space":obs_sp,"action_space":act_sp})

    def _partner_action(self, agent_name: str) -> np.ndarray:
        self._partner_calls += 1
        if self._partner_calls % _PARTNER_RELOAD_INTERVAL == 0:
            self._model_loaded = False
        self._load_model()
        obs = self._last_partner_obs.get(agent_name,
                  np.zeros(self.OBS_DIM, dtype=np.float32))
        if self._model is None:
            return self.action_space.sample()
        act, _ = self._model.predict(obs, deterministic=False)
        return act

    def reset(self, seed=None, options=None):
        self._lazy_init()
        self._trainee_idx = self._ep_count % ADRSwarmEnv.N_DRONES
        self._ep_count   += 1
        obs_dict, _ = self._env.reset(seed=seed or self._seed)
        self._last_partner_obs = obs_dict.copy()
        return obs_dict[f"drone_{self._trainee_idx}"], {}

    def step(self, action):
        trainee_name = f"drone_{self._trainee_idx}"
        actions = {}
        for a in self._env.possible_agents:
            actions[a] = action if a == trainee_name else self._partner_action(a)
        obs, rews, terms, truncs, infos = self._env.step(actions)
        for a,o in obs.items(): self._last_partner_obs[a] = o
        return (obs.get(trainee_name,   np.zeros(self.OBS_DIM,dtype=np.float32)),
                rews.get(trainee_name,  0.),
                terms.get(trainee_name, True),
                truncs.get(trainee_name,False),
                infos.get(trainee_name, {}))

    def close(self):
        if self._env: self._env.close()


def make_swarm_vec_env(shared_model_path: str | None, num_envs: int, seed: int):
    _path = shared_model_path; _seed = seed
    def make_fn(idx):
        def _init():
            return SwarmAgentWrapper(shared_model_path=_path, seed=_seed+idx)
        return _init
    vec = SubprocVecEnv([make_fn(i) for i in range(num_envs)])
    return VecMonitor(vec)

def load_or_create_swarm(ckpt_dir: str, vec_env, seed: int):
    files = glob.glob(os.path.join(ckpt_dir,"swarm_shared*.zip"))
    if files:
        latest = max(files,key=os.path.getmtime)
        print(f"  [LOAD] {latest}")
        model = PPO.load(latest, env=vec_env, device="auto")
        # the checkpoint restores its own lr schedule; override it on resume
        model.learning_rate = 1e-4
        model._setup_lr_schedule()
        from stable_baselines3.common.utils import update_learning_rate
        update_learning_rate(model.policy.optimizer, 1e-4)
        print(f"  [LR]   forced to 1e-4")
        return model
    print("  [NEW] New shared swarm model.")
    return PPO(
        "MlpPolicy", vec_env,
        learning_rate=1e-4, gamma=0.99,
        n_steps=4096, batch_size=512,
        n_epochs=10, ent_coef=0.02,
        clip_range=0.20, gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[256,256,128]),
        verbose=1, seed=seed, device="cpu")

def _close_vec(vec):
    try: vec.close()
    except: pass
    gc.collect(); time.sleep(1.5)


class PartnerSyncCallback(BaseCallback):
    """Periodically saves the learner so that partner drones pick up the latest policy."""
    def __init__(self, save_path: str, sync_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.sync_freq == 0:
            self.model.save(self.save_path.replace(".zip", ""))
        return True


if __name__ == "__main__":
    import resource
    # Raise the open-file limit (SubprocVecEnv workers open many file descriptors)
    try:
        soft,hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft  = max(hard,8192) if hard>0 else 8192
        resource.setrlimit(resource.RLIMIT_NOFILE,(new_soft,hard))
        print(f"[FD LIMIT] {soft} → {new_soft}")
    except Exception as e:
        print(f"[FD LIMIT] {e}")

    SEED            = 42
    NUM_ENVS        = 8
    STEPS_PER_ROUND = 2_500_000
    NUM_ROUNDS      = 4
    CKPT_DIR        = "./swarm_checkpoints/"
    os.makedirs(CKPT_DIR,exist_ok=True)

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    set_random_seed(SEED)

    print("="*70)
    print(" ADR SWARM - cooperative laser debris removal, shared-policy IPPO training")
    print(" Checkpoint: ./swarm_checkpoints/")
    print("="*70)
    print("\n[INIT] Pre-loading astronomical data files...")
    load('de421.bsp'); load.open(hipparcos.URL)
    print("[INIT] OK\n")

    shared_latest = os.path.join(CKPT_DIR,"swarm_shared_latest.zip")
    shared_path   = shared_latest if os.path.exists(shared_latest) else None

    SYNC_FREQ = max(STEPS_PER_ROUND // 10, 100_000)

    for rnd in range(NUM_ROUNDS):
        print(f"\n{'='*70}  ROUND {rnd+1}/{NUM_ROUNDS}")
        print(f"  Model path: {shared_path or 'new (random)'}")
        vec = make_swarm_vec_env(shared_path, NUM_ENVS, SEED)
        model = load_or_create_swarm(CKPT_DIR, vec, SEED)
        cb_ckpt = CheckpointCallback(
            save_freq = STEPS_PER_ROUND // NUM_ENVS // 4,
            save_path = CKPT_DIR, name_prefix="swarm_shared")
        cb_sync = PartnerSyncCallback(
            save_path=shared_latest, sync_freq=SYNC_FREQ // NUM_ENVS)
        model.learn(total_timesteps=STEPS_PER_ROUND,
                    callback=CallbackList([cb_ckpt, cb_sync]),
                    reset_num_timesteps=(rnd == 0))
        model.save(shared_latest.replace(".zip",""))
        del model
        _close_vec(vec)
        shared_path = shared_latest
        print(f"  [Round {rnd+1}] Checkpoint saved: {shared_latest}")

    print("\n"+"="*70)
    print(" TRAINING COMPLETE - visual test rollout")
    print("="*70)
    import imageio

    final_model = PPO.load(shared_latest, device="cpu")
    st_test     = StarTrackerModel()
    test_env    = ADRSwarmEnv(star_tracker=st_test, render_mode="rgb_array")
    obs, _      = test_env.reset(seed=SEED)
    frames      = []

    for step_i in range(test_env.MAX_STEPS):
        actions = {}
        for a in test_env.possible_agents:
            act,_ = final_model.predict(obs[a], deterministic=True)
            actions[a] = act
        obs, rews, terms, truncs, _ = test_env.step(actions)
        if step_i % 3 == 0:
            frame = test_env.render()
            if frame is not None: frames.append(frame)
        if all(terms.values()) or all(truncs.values()):
            status = "mission success" if test_env.mission_done else "failed"
            print(f"{status} — Step {step_i+1} — Phase {test_env.phase:.1f}")
            frame = test_env.render()
            if frame is not None: frames.append(frame)
            break

    if frames:
        imageio.mimsave("swarm_mission.gif", frames, fps=20)
        print("GIF saved: swarm_mission.gif")

    test_env.render_mode = "human"
    test_env.render()
    test_env.close()
