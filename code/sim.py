"""
Hierarchical UAV-FANET / IoV / MEC simulator.

Every candidate solution is evaluated with a vectorised model (population axis P).
Decision vector per epoch (all genes normalised to [0,1)):
  gt[N]   offloading target (rank in heuristic-ordered option list)
  ga[N]   access node (rank in SNR-ordered feasible access list)
  w[N]    bandwidth weight            -> b_k = B_ap * w_k / sum w
  phi[N]  CPU weight                  -> f_k = F_s * phi_k / sum phi
  p[N]    vehicle uplink power
  mv[2U]  UAV micro-repositioning (dx, dy)
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path

# ----------------------------------------------------------------- constants
AREA = 1000.0
ROADS = np.array([125.0, 375.0, 625.0, 875.0])
DT = 5.0                       # control interval (s)
MISSION = 300.0                # UAV mission horizon (s)
M, U = 5, 10                   # MEC/RSU, UAV
RSU_POS = np.array([[125, 375], [625, 125], [625, 625], [375, 875], [875, 875]], float)
RSU_H = 10.0
R_RSU = 300.0                  # V2I range (m)
R_UAV = 500.0                  # A2G horizontal range (m)
FC = 7.0                       # GHz (FR3 upper mid-band)
B_RSU = 40e6                   # Hz per RSU (orthogonal channels, 5 x 40 = 200 MHz)
B_UAV = 50e6                   # Hz per UAV sub-band (aerial band 200 MHz, reuse 4)
N_SB = 4
B_BH = 20e6                    # Hz per wireless backhaul link
N0 = 10 ** ((-174 + 9 - 30) / 10)   # W/Hz incl. 9 dB noise figure
G_RSU, G_UAV, G_VEH = 10 ** 0.8, 10 ** 0.3, 1.0
P_MIN, P_MAX = 0.02, 0.2       # vehicle uplink (W)
P_AP_RSU, P_AP_UAV, P_BH = 1.0, 0.5, 1.0
F_LOC, F_MEC, F_UAV, F_CLOUD = 1.0e9, 40e9, 8e9, 8e9
K_VEH, K_MEC, K_UAV = 1e-27, 1e-28, 1e-28
E_CLOUD_PER_CYCLE = 1e-9
R_WIRED, R_CLOUD, PROP_CLOUD = 1e9, 2e8, 0.060
GAMMA_TH = 2.0                 # 3 dB decoding threshold
HARQ = 3
V_F = 12.0                     # UAV transit speed (m/s)
EPS_FLY = 127.8 / V_F          # J/m, P(v_f)/v_f from rotary-wing model
P_HOVER = 168.5
D_MAX = 60.0                   # max repositioning per interval (m)
D_MIN = 5.0
E_SAFE = 20e3
CLASSES = {  # prob, d[Mb], c[Gcyc], deadline[s], R target
    0: (0.40, (0.1, 0.3), (0.02, 0.08), 0.100, 0.95),   # safety
    1: (0.35, (1.0, 2.5), (0.3, 0.8), 0.300, 0.95),     # cooperative perception
    2: (0.25, (2.0, 6.0), (0.1, 0.4), 1.000, 0.90),     # infotainment / map
}
N_OPT = 1 + M + U + 1
W_MIN = 0.25                   # minimum share weight (max 4:1 ratio) -> no starvation
L_CAP = 5.0
DZ = 0.3


def rotor_power(V):
    P0, Pi, Ut, v0, d0, rho, s, A = 79.86, 88.63, 120, 4.03, 0.6, 1.225, 0.05, 0.503
    return P0 * (1 + 3 * V ** 2 / Ut ** 2) + Pi * np.sqrt(np.sqrt(1 + V ** 4 / (4 * v0 ** 4)) - V ** 2 / (2 * v0 ** 2)) + 0.5 * d0 * rho * s * A * V ** 3


def fspl_db(d):
    return 32.45 + 20 * np.log10(np.maximum(d, 1.0)) + 20 * np.log10(FC * 1e3) - 60.0  # d in m, f in MHz -> 32.45+20log d_km+20log f_MHz


def v2i_gain(dist3d, shadow_db):
    pl = 32.4 + 21 * np.log10(np.maximum(dist3d, 1.0)) + 20 * np.log10(FC)   # 3GPP UMi-SC LoS
    return G_VEH * G_RSU * 10 ** (-(pl + shadow_db) / 10)


def a2g_gain(horiz, h, shadow_db, g_other=G_VEH):
    d = np.sqrt(horiz ** 2 + h ** 2)
    theta = np.degrees(np.arctan2(h, np.maximum(horiz, 1e-3)))
    plos = 1.0 / (1 + 9.61 * np.exp(-0.16 * (theta - 9.61)))
    pl = fspl_db(d) + plos * 1.0 + (1 - plos) * 20.0
    return g_other * G_UAV * 10 ** (-(pl + shadow_db) / 10)


def a2a_gain(d):
    return G_UAV * G_UAV * 10 ** (-fspl_db(d) / 10)


def contact_time(pos, vel, centre, radius):
    """time until |pos + vel t - centre| > radius (broadcast); 0 if already outside"""
    rel = pos - centre
    a = np.sum(vel ** 2, -1) + 1e-9
    b = 2 * np.sum(rel * vel, -1)
    c = np.sum(rel ** 2, -1) - radius ** 2
    disc = np.maximum(b ** 2 - 4 * a * c, 0)
    t = (-b + np.sqrt(disc)) / (2 * a)
    return np.where(c > 0, 0.0, np.maximum(t, 0.0))


# ----------------------------------------------------------------- mobility
class World:
    """Vehicle mobility is decision-independent (own RNG) so every algorithm sees the same
    trajectories, tasks and channels; UAV positions and batteries evolve with each algorithm's decisions."""

    def __init__(self, n_veh, seed, sub=0.25):
        self.seed = seed
        self.rng = np.random.default_rng([seed, 0])
        r = self.rng
        self.n_veh = n_veh
        horiz = r.random(n_veh) < 0.5
        lane = r.choice(ROADS, n_veh)
        along = r.uniform(0, AREA, n_veh)
        self.pos = np.where(horiz[:, None], np.c_[along, lane], np.c_[lane, along])
        sgn = r.choice([-1.0, 1.0], n_veh)
        self.speed = r.uniform(60, 80, n_veh) / 3.6
        self.dir = np.where(horiz[:, None], np.c_[sgn, 0 * sgn], np.c_[0 * sgn, sgn])
        self.uav = r.uniform(100, 900, (U, 2))
        self.uav_h = r.uniform(100, 150, U)
        self.battery = r.uniform(75e3, 150e3, U)
        self.load = r.uniform(0.1, 0.6, M)
        self.t = 0.0
        self.epoch = 0
        self.sub = sub
        self.hist = []
        for _ in range(int(4.0 / sub)):           # warm-up so that 4 s of history exist
            self._substep()

    def _substep(self):
        r, sub = self.rng, self.sub
        self.hist.append((self.pos.copy(), (self.dir * self.speed[:, None]).copy()))
        self.hist = self.hist[-int(4.0 / sub) - 1:]
        new = self.pos + self.dir * self.speed[:, None] * sub
        for i in range(self.n_veh):
            ax = 0 if self.dir[i, 0] != 0 else 1
            o = 1 - ax
            lo, hi = sorted([self.pos[i, ax], new[i, ax]])
            cross = ROADS[(ROADS > lo) & (ROADS <= hi)] if self.dir[i, ax] > 0 else ROADS[(ROADS >= lo) & (ROADS < hi)]
            if new[i, ax] < 0 or new[i, ax] > AREA:
                self.dir[i] *= -1
                new[i, ax] = np.clip(new[i, ax], 0, AREA)
            elif len(cross) and r.random() < 0.5:
                new[i, ax] = cross[0]
                d = np.zeros(2); d[o] = r.choice([-1.0, 1.0])
                self.dir[i] = d
            if self.dir[i, o] == 0:
                new[i, o] = self.pos[i, o]
        self.pos = new

    def step_vehicles(self, dt=DT):
        for _ in range(int(dt / self.sub)):
            self._substep()
        self.t += dt
        self.epoch += 1
        self.load = np.clip(self.load + self.rng.normal(0, 0.05, M), 0.05, 0.8)

    def snapshot(self, lam=1.2, tau_max=2.0, age_aware=True, rank_order=True):
        return Scenario(self, lam, tau_max, age_aware, rank_order)

    def apply(self, detail):
        self.uav = detail["uav_pos"]
        self.battery = self.battery - P_HOVER * DT - detail["E_uav"]


# ----------------------------------------------------------------- scenario (one epoch)
class Scenario:
    def __init__(self, w, lam, tau_max, age_aware, rank_order=True):
        r = np.random.default_rng([w.seed, 1, w.epoch])      # paired across algorithms
        self.rank_order = rank_order
        cnt = r.poisson(lam, w.n_veh)
        veh = np.repeat(np.arange(w.n_veh), cnt)
        N = len(veh)
        self.N, self.veh, self.n_veh = N, veh, w.n_veh
        probs = [CLASSES[c][0] for c in range(3)]
        cls = r.choice(3, N, p=probs)
        self.cls = cls
        d = np.empty(N); c = np.empty(N); Lmax = np.empty(N); Rt = np.empty(N)
        for k in range(3):
            m = cls == k
            _, dr, cr, dl, rt = CLASSES[k]
            d[m] = r.uniform(*dr, m.sum()) * 1e6
            c[m] = r.uniform(*cr, m.sum()) * 1e9
            Lmax[m] = dl; Rt[m] = rt
        self.d, self.c, self.o, self.Lmax, self.Rt = d, c, 0.05 * d, Lmax, Rt
        # true state
        self.pos = w.pos[veh]
        self.vel = (w.dir * w.speed[:, None])[veh]
        # controller view: last report has age tau (FANET/MEC aggregation delay)
        tau_v = r.uniform(0, tau_max, w.n_veh) if tau_max > 0 else np.zeros(w.n_veh)
        lag = np.minimum(np.round(tau_v / w.sub).astype(int), len(w.hist))
        ps = np.array([w.hist[-l][0][i] if l > 0 else w.pos[i] for i, l in enumerate(lag)])
        vs = np.array([w.hist[-l][1][i] if l > 0 else (w.dir[i] * w.speed[i]) for i, l in enumerate(lag)])
        tau_eff = lag * w.sub
        self.tau = tau_eff[veh]
        self.age_aware = age_aware
        self.pos_k = (ps + vs * tau_eff[:, None])[veh] if age_aware else ps[veh]
        self.vel_k = vs[veh]
        self.uav0 = w.uav.copy()
        self.uav_h = w.uav_h.copy()
        remaining = max(MISSION - w.t, DT)
        self.bud = (w.battery - E_SAFE) / (remaining / DT) - P_HOVER * DT   # per-epoch budget beyond hover
        self.F_mec = F_MEC * (1 - w.load)
        self.W_mec = r.uniform(0, 1.5e9, M)
        self.sh_vr = r.normal(0, 4, (N, M))
        self.sh_vu = r.normal(0, 3, (N, U))
        for tag, P_, V_ in (("t", self.pos, self.vel), ("k", self.pos_k, self.vel_k)):
            hr = np.linalg.norm(P_[:, None, :] - RSU_POS[None], axis=-1)
            setattr(self, "g_vr_" + tag, v2i_gain(np.sqrt(hr ** 2 + RSU_H ** 2), self.sh_vr))
            setattr(self, "tc_vr_" + tag, contact_time(P_[:, None, :], V_[:, None, :], RSU_POS[None], R_RSU))
        # access lists from the controller's view
        hr = np.linalg.norm(self.pos_k[:, None, :] - RSU_POS[None], axis=-1)
        hu = np.linalg.norm(self.pos_k[:, None, :] - self.uav0[None], axis=-1)
        g_u = a2g_gain(hu, self.uav_h[None], self.sh_vu)
        snr = np.concatenate([np.where(hr <= R_RSU, self.g_vr_k * P_MAX / (N0 * B_RSU / 8), -1),
                              np.where(hu <= R_UAV, g_u * P_MAX / (N0 * B_UAV / 8), -1)], 1)
        order = np.argsort(-snr, 1)
        self.n_acc = (snr > 0).sum(1)
        self.acc_list = np.where(np.arange(M + U)[None] < self.n_acc[:, None], order, -1)
        self._build_backhaul()
        self._heuristic_order()

    # ------------------------------------------------------------- backhaul graph
    def _build_backhaul(self):
        nodes = M + U + 1
        edges = []  # (a, b, type)  type 0 wired, 1 cloud, 2 uav-rsu, 3 uav-uav
        for i in range(M):
            for j in range(i + 1, M):
                edges.append((i, j, 0))
            edges.append((i, M + U, 1))
        for u in range(U):
            for m in range(M):
                if np.linalg.norm(self.uav0[u] - RSU_POS[m]) <= 600:
                    edges.append((M + u, m, 2))
            for v in range(u + 1, U):
                if np.linalg.norm(self.uav0[u] - self.uav0[v]) <= 600:
                    edges.append((M + u, M + v, 3))
        self.edges = np.array(edges, int)
        E = len(edges)
        self.E = E
        rate0 = self.edge_rates(self.uav0[None])[0]
        wgt = 1e6 / rate0 + np.where(self.edges[:, 2] == 1, PROP_CLOUD, 1e-3)
        W = np.full((nodes, nodes), np.inf)
        for e, (a, b, _) in enumerate(edges):
            W[a, b] = W[b, a] = wgt[e]
        np.fill_diagonal(W, 0)
        dist, pred = shortest_path(np.where(np.isinf(W), 0, W), directed=False, return_predecessors=True)
        eid = {}
        for e, (a, b, _) in enumerate(edges):
            eid[(a, b)] = eid[(b, a)] = e
        A = M + U
        self.inc = np.zeros((A, nodes, E), bool)
        self.reach = np.zeros((A, nodes), bool)
        for a in range(A):
            for s in range(nodes):
                if a == s:
                    self.reach[a, s] = True
                    continue
                if np.isinf(dist[a, s]):
                    continue
                self.reach[a, s] = True
                x = s
                while x != a:
                    px = pred[a, x]
                    self.inc[a, s, eid[(px, x)]] = True
                    x = px
        # UAV endpoints of edges (for energy attribution)
        self.edge_uav = np.zeros((E, U))
        for e, (a, b, t) in enumerate(edges):
            if a >= M and a < M + U:
                self.edge_uav[e, a - M] = 1
            if b >= M and b < M + U:
                self.edge_uav[e, b - M] = 1
        self.prop = np.where(self.edges[:, 2] == 1, PROP_CLOUD, 0.0)

    def edge_rates(self, uavpos):
        """uavpos (P,U,2) -> rates (P,E) bit/s"""
        Pn = uavpos.shape[0]
        R = np.empty((Pn, len(self.edges)))
        for e, (a, b, t) in enumerate(self.edges):
            if t == 0:
                R[:, e] = R_WIRED
            elif t == 1:
                R[:, e] = R_CLOUD
            elif t == 2:
                hz = np.linalg.norm(uavpos[:, a - M] - RSU_POS[b], axis=-1)
                g = a2g_gain(hz, self.uav_h[a - M] - RSU_H, 0.0, G_RSU)
                R[:, e] = B_BH * np.log2(1 + P_BH * g / (N0 * B_BH))
            else:
                dd = np.linalg.norm(uavpos[:, a - M] - uavpos[:, b - M], axis=-1)
                dd = np.sqrt(dd ** 2 + (self.uav_h[a - M] - self.uav_h[b - M]) ** 2)
                g = a2a_gain(dd)
                R[:, e] = B_BH * np.log2(1 + P_BH * g / (N0 * B_BH))
        return R

    # ------------------------------------------------------------- heuristic option ordering
    def _heuristic_order(self):
        N = self.N
        est = np.full((N, N_OPT), np.inf)
        est[:, 0] = self.c / F_LOC
        rate0 = self.edge_rates(self.uav0[None])[0]
        share = max(N / (M + U), 1.0)
        best_acc = self.acc_list[:, 0]
        hu = np.linalg.norm(self.pos_k[:, None, :] - self.uav0[None], axis=-1)
        g_u = a2g_gain(hu, self.uav_h[None], self.sh_vu)
        g_r = self.g_vr_k
        for k in range(N):
            a = best_acc[k]
            if a < 0:
                continue
            if a < M:
                b = B_RSU / share; g = g_r[k, a]
            else:
                b = B_UAV / share; g = g_u[k, a - M]
            r_up = b * np.log2(1 + P_MAX * g / (b * N0))
            tup = self.d[k] / r_up
            for s in range(M + U + 1):
                if not self.reach[a, s]:
                    continue
                bh = np.sum(self.inc[a, s] * ((self.d[k] + self.o[k]) / rate0 * share / 4 + self.prop))
                if s < M:
                    cmp = self.W_mec[s] / self.F_mec[s] + self.c[k] / (self.F_mec[s] / share)
                elif s < M + U:
                    cmp = self.c[k] / (F_UAV / share)
                else:
                    cmp = self.c[k] / F_CLOUD
                est[k, 1 + s] = tup + bh + cmp
        self.est = est
        if self.rank_order:
            self.order = np.argsort(est, 1, kind="stable")      # option ids by rank
        else:                                                   # ablation: arbitrary order, unreachable last
            rr = np.random.default_rng(12345 + N)
            key = rr.random(est.shape) + np.isinf(est) * 10
            self.order = np.argsort(key, 1)
            for k in range(N):
                na = self.n_acc[k]
                if na > 1:
                    self.acc_list[k, :na] = rr.permutation(self.acc_list[k, :na])
        self.n_valid = np.isfinite(est).sum(1)

    # ------------------------------------------------------------- reference energy (fixed, algorithm-independent)
    @property
    def E_task_ref(self):          # all-local execution energy (fixed per epoch)
        return np.sum(K_VEH * self.c * F_LOC ** 2)

    @property
    def E_move_ref(self):          # every UAV flies the maximum repositioning distance
        return U * EPS_FLY * D_MAX

    @property
    def dim(self):
        return 5 * self.N + 2 * U

    # ------------------------------------------------------------- seeds
    def greedy_vector(self):
        N = self.N
        x = np.empty(self.dim)
        x[:N] = 0.5 / N_OPT
        x[N:2 * N] = 0.5 / np.maximum(self.n_acc, 1)
        x[2 * N:4 * N] = 1.0 - 1e-9
        x[4 * N:5 * N] = 1.0 - 1e-9
        x[5 * N:] = 0.5
        return x

    def nearest_edge_vector(self):
        """conventional greedy: best-SNR access node, execute on that node's own
        server (RSU-MEC or UAV CPU), equal shares, max power, UAVs hover."""
        N = self.N
        x = self.greedy_vector()
        a = self.acc_list[:, 0]
        opt_id = np.where(a >= 0, a + 1, 0)
        rank = np.argmax(self.order == opt_id[:, None], 1)
        x[:N] = (rank + 0.5) / N_OPT
        return x

    def local_vector(self):
        N = self.N
        x = self.greedy_vector()
        rank_local = np.argmax(self.order == 0, 1)
        x[:N] = (rank_local + 0.5) / N_OPT
        x[4 * N:5 * N] = 0.0
        return x


# ----------------------------------------------------------------- evaluator
WEIGHTS = {"balanced": (0.3, 0.2, 0.2, 0.3), "equal": (0.25, 0.25, 0.25, 0.25),
           "latency": (0.5, 0.1, 0.1, 0.3), "energy": (0.15, 0.55, 0.1, 0.2)}


class Evaluator:
    """evaluate(X) for a population; `true` uses the actual channel, otherwise the
    controller's estimate (CSI error). contact_aware toggles link-lifetime awareness."""

    def __init__(self, scen, weights="balanced", true=False, contact_aware=True, allow_move=True):
        self.s = scen
        self.w = WEIGHTS[weights] if isinstance(weights, str) else weights
        self.true = true
        self.contact_aware = contact_aware or true
        self.allow_move = allow_move
        self.nfe = 0
        s = scen
        self.g_vr = s.g_vr_t if true else s.g_vr_k
        self.tc_vr = s.tc_vr_t if true else s.tc_vr_k
        self.pos = s.pos if true else s.pos_k
        self.vel = s.vel if true else s.vel_k
        self.sh_u = s.sh_vu

    def decode(self, X):
        s = self.s; N = s.N
        X = np.clip(X, 0, 1 - 1e-9)
        rt = np.minimum((X[:, :N] * N_OPT).astype(int), s.n_valid[None] - 1)
        opt = np.take_along_axis(np.broadcast_to(s.order, (X.shape[0], N, N_OPT)), rt[..., None], 2)[..., 0]
        ra = (X[:, N:2 * N] * np.maximum(s.n_acc, 1)[None]).astype(int)
        acc = np.take_along_axis(np.broadcast_to(s.acc_list, (X.shape[0], N, M + U)), ra[..., None], 2)[..., 0]
        noacc = (acc < 0)
        srv = opt - 1                                     # server node id; -1 local
        unreach = np.zeros_like(noacc)
        off = srv >= 0
        aa = np.where(acc < 0, 0, acc); ss = np.where(srv < 0, 0, srv)
        unreach = off & ~s.reach[aa, ss]
        local = (~off) | noacc | unreach                  # decode-time repair -> local
        return opt, np.where(local, -1, acc), np.where(local, -1, srv), local

    def __call__(self, X, detail=False):
        s = self.s; N = s.N; P = X.shape[0]
        self.nfe += P
        X = np.clip(X, 0, 1 - 1e-9)
        opt, acc, srv, local = self.decode(X)
        w = W_MIN + (1 - W_MIN) * X[:, 2 * N:3 * N]
        phi = W_MIN + (1 - W_MIN) * X[:, 3 * N:4 * N]
        p = P_MIN + (P_MAX - P_MIN) * X[:, 4 * N:5 * N]
        z = X[:, 5 * N:].reshape(P, U, 2) * 2 - 1              # dead-zone encoding: |z|<DZ -> 0
        mv = np.sign(z) * np.maximum(np.abs(z) - DZ, 0) / (1 - DZ) * D_MAX
        dist = np.linalg.norm(mv, axis=-1)
        scale = np.where(dist > D_MAX, D_MAX / np.maximum(dist, 1e-9), 1.0)
        mv = mv * scale[..., None]
        dist = np.minimum(dist, D_MAX)
        if not self.allow_move:
            mv[:] = 0; dist[:] = 0
        moving = dist >= D_MIN
        mv = np.where(moving[..., None], mv, 0.0); dist = np.where(moving, dist, 0.0)
        uav = np.clip(s.uav0[None] + mv, 0, AREA)
        E_move = EPS_FLY * dist                                     # (P,U)

        # ---------------- access channel
        hu = np.linalg.norm(self.pos[None, :, None, :] - uav[:, None, :, :], axis=-1)   # P,N,U
        g_vu = a2g_gain(hu, s.uav_h[None, None], self.sh_u[None])
        off = ~local
        a_r = off & (acc < M)
        a_u = off & (acc >= M)
        ar = np.where(a_r, acc, 0); au = np.where(a_u, acc - M, 0)
        g_srv = np.where(a_r, np.take_along_axis(np.broadcast_to(self.g_vr, (P, N, M)), ar[..., None], 2)[..., 0],
                         np.take_along_axis(g_vu, au[..., None], 2)[..., 0])
        # RSUs use orthogonal channels -> no inter-RSU interference.
        # UAV tier: 4 sub-bands (UAV u uses sub-band u % 4); co-sub-band users of other UAVs interfere.
        pu = p * a_u
        sb_of_acc = np.where(a_u, au % N_SB, 0)
        oh = np.eye(N_SB)[sb_of_acc] * a_u[..., None]                 # P,N,SB
        S = np.einsum("pn,pnu,pns->pus", pu, g_vu, oh)                # P,U,SB
        own = p * g_srv
        flu = (np.arange(P)[:, None] * U + au).ravel()
        own_cell = np.bincount(flu, (own * a_u).ravel(), P * U).reshape(P, U)
        I_cell = (S[:, np.arange(U), np.arange(U) % N_SB] - own_cell) / B_UAV
        I = np.where(a_u, np.take_along_axis(I_cell, au, 1), 0.0)
        I = np.maximum(I, 0)
        # bandwidth shares
        grp = np.where(off, acc, M + U)                               # AP index; M+U = none
        flat = (np.arange(P)[:, None] * (M + U + 1) + grp).ravel()
        wsum = np.bincount(flat, (w * off).ravel(), P * (M + U + 1)).reshape(P, M + U + 1)
        Btier = np.where(acc < M, B_RSU, B_UAV)
        b = Btier * w / np.maximum(np.take_along_axis(wsum, grp, 1), 1e-12)
        sinr = own / (b * (N0 + I))
        r_up = b * np.log2(1 + sinr)
        ps = np.exp(-GAMMA_TH / np.maximum(sinr, 1e-9))
        att = sum((1 - ps) ** i for i in range(HARQ))
        rel_up = 1 - (1 - ps) ** HARQ
        T_up = np.where(off, s.d / np.maximum(r_up, 1) * att, 0)
        Pap = np.where(acc < M, P_AP_RSU, P_AP_UAV)
        r_dn = b * np.log2(1 + Pap * g_srv / (b * N0))
        T_dn = np.where(off, s.o / np.maximum(r_dn, 1), 0)

        # ---------------- backhaul
        R_e = s.edge_rates(uav)                                        # P,E
        aa = np.where(off, acc, 0); ss = np.where(off, srv, 0)
        inc = s.inc[aa, ss] & off[..., None]                           # P,N,E
        load = inc.sum(1)                                              # tasks per edge
        per_edge = (s.d + s.o)[None, :, None] * (load / R_e)[:, None, :] + s.prop[None, None, :]
        T_bh = np.where(inc, per_edge, 0).sum(-1)
        airtime = np.where(inc, ((s.d + s.o)[None, :, None] / R_e[:, None, :]), 0)   # P,N,E
        E_bh_uav = np.einsum("pe,eu->pu", airtime.sum(1) * P_BH, s.edge_uav)
        # ---------------- computation
        G = s.n_veh + M + U + N                                       # groups: local(veh) | mec | uav | cloud(each)
        gid = np.where(local, s.veh[None], np.where(srv < M, s.n_veh + srv,
                       np.where(srv < M + U, s.n_veh + srv, s.n_veh + M + U + np.arange(N)[None])))
        gid = np.where(local, s.veh[None], gid)
        flat = (np.arange(P)[:, None] * G + gid).ravel()
        psum = np.bincount(flat, phi.ravel(), P * G).reshape(P, G)
        Fcap = np.where(local, F_LOC, np.where(srv < M, s.F_mec[np.clip(srv, 0, M - 1)],
                        np.where(srv < M + U, F_UAV, F_CLOUD)))
        cloud = (~local) & (srv == M + U)
        f = np.where(cloud, F_CLOUD, Fcap * phi / np.take_along_axis(psum, gid, 1))
        T_cmp = s.c / f
        T_q = np.where((~local) & (srv < M), s.W_mec[np.clip(srv, 0, M - 1)] / s.F_mec[np.clip(srv, 0, M - 1)], 0)
        L = T_up + T_bh + T_q + T_cmp + T_dn

        # ---------------- reliability & link lifetime
        hu_new = hu
        tc_u = contact_time(self.pos[None, :, None, :], self.vel[None, :, None, :], uav[:, None, :, :], R_UAV)  # P,N,U
        tc = np.where(a_r, np.take_along_axis(np.broadcast_to(self.tc_vr, (P, N, M)), ar[..., None], 2)[..., 0],
                      np.take_along_axis(tc_u, au[..., None], 2)[..., 0])
        alive = np.where(off, L <= tc, True)
        bh_snr_ok = 1.0  # dedicated backhaul links operate far above threshold
        R = np.where(off, rel_up, 1.0)
        if self.contact_aware:
            R = R * alive
        Q = (L <= s.Lmax) & (R >= s.Rt)

        # ---------------- energy
        E_tx = np.where(off, p * T_up, 0)
        k_eff = np.where(local, K_VEH, np.where(srv < M, K_MEC, K_UAV))
        E_cmp = np.where(cloud, E_CLOUD_PER_CYCLE * s.c, k_eff * s.c * f ** 2)
        E_dn = np.where(off, Pap * T_dn, 0)
        uav_srv = (~local) & (srv >= M) & (srv < M + U)
        E_uav_cmp = np.zeros((P, U)); E_uav_dn = np.zeros((P, U))
        fl = (np.arange(P)[:, None] * U + np.clip(srv - M, 0, U - 1)).ravel()
        E_uav_cmp = np.bincount(fl, (E_cmp * uav_srv).ravel(), P * U).reshape(P, U)
        fl2 = (np.arange(P)[:, None] * U + au).ravel()
        E_uav_dn = np.bincount(fl2, (E_dn * a_u).ravel(), P * U).reshape(P, U)
        E_uav = E_move + E_uav_cmp + E_uav_dn + E_bh_uav
        E_total = E_tx.sum(1) + E_cmp.sum(1) + E_dn.sum(1) + E_bh_uav.sum(1) + E_move.sum(1)

        # ---------------- objective
        wl, we, wr, wq = self.w
        Lh = np.mean(np.minimum(L / s.Lmax, L_CAP) / L_CAP, 1)
        Eh = 0.5 * (E_total - E_move.sum(1)) / s.E_task_ref + 0.5 * E_move.sum(1) / s.E_move_ref
        Rh = R.mean(1)
        Qh = Q.mean(1)
        F = wl * Lh + we * Eh + wr * (1 - Rh) + wq * (1 - Qh)
        safety = s.cls == 0
        g1 = np.sum(np.maximum(0, L / s.Lmax - 1) * safety, 1)
        g2 = np.sum(np.maximum(0, E_uav - s.bud[None]) / np.maximum(np.abs(s.bud[None]), 1.0), 1)
        viol = g1 + g2
        key = F + (viol > 0) * (10.0 + viol)
        if not detail:
            return key, F, viol
        i = 0
        assert P == 1
        kind = np.where(local[i], 0, np.where(srv[i] < M, 1, np.where(srv[i] < M + U, 2, 3)))
        return dict(key=key[i], F=F[i], viol=viol[i], L=L[i], R=R[i], Q=Q[i], cls=s.cls, kind=kind,
                    E_total=E_total[i], E_tx=E_tx[i].sum(), E_cmp=E_cmp[i].sum(), E_move=E_move[i].sum(),
                    E_uav=E_uav[i], E_bh=E_bh_uav[i].sum() + E_dn[i].sum(), uav_pos=uav[i],
                    T_up=T_up[i], T_bh=T_bh[i], T_q=T_q[i], T_cmp=T_cmp[i], T_dn=T_dn[i],
                    contact_fail=(off[i] & ~alive[i]), safety_viol=(safety & (L[i] > s.Lmax)),
                    uav_budget_viol=(E_uav[i] > s.bud).sum())
