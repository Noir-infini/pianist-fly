"""DLS Jacobian IK + reachability for the fly's four front legs.

Drives the claw TIP site of T1L/T1R/T2L/T2R onto an (x, y, z) target in the
piano scene (world frame). Joints are the limb servos of the native 4-leg
action layout (stand_native_4legs.LEG_SLOTS): coxa, femur_twist, femur,
tibia, tarsus are solved; coxa_abduct/coxa_twist stay at their rest values.

The solver uses MuJoCo's analytic site Jacobian (mj_jacSite) with damped
least squares, so it is exact for the real model (root pose affects it, and
joint limits are respected by clamping qpos). The returned joint targets are
ACTION values (rad) ready to drop into the 59-action used by the live loop.
"""
import numpy as np
import mujoco

import stand_native_4legs as s4

_KEYSPECS = None


def _key(j):
    global _KEYSPECS
    if _KEYSPECS is None:
        import json
        import os
        _KEYSPECS = json.load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "piano", "key_positions.json")))["keys"]
    return _KEYSPECS[j]


def press_target(j):
    """World target that places the claw tip exactly ON key j's top face.

    Uses the physical box surface (top+0.0005, where the collider sits) so a
    press makes real contact without commanding penetration.
    """
    k = _key(j)
    return [k["center_xyz"][0], k["center_xyz"][1],
            k["key_top_z"] + 0.0005]

SOLVE_JOINTS = ("coxa", "femur_twist", "femur", "tibia", "tarsus", "tarsus2")
_LOCK = ("coxa_abduct", "coxa_twist")  # stay at rest (no sideways twist)
_LIMB_CLAW = {
    "T1_left": "walker/claw_T1_left",
    "T1_right": "walker/claw_T1_right",
    "T2_left": "walker/claw_T2_left",
    "T2_right": "walker/claw_T2_right",
    "T3_left": "walker/claw_T3_left",
    "T3_right": "walker/claw_T3_right",
}


class LegIK:
    """IK for one front limb in a built piano/free-floor env."""

    def __init__(self, env, leg):
        self.m = getattr(env.physics.model, "_model", env.physics.model)
        self.d = env.physics.data._data
        self.leg = leg
        self.base = s4.LEG_SLOTS[leg]
        self.site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE,
                                         _LIMB_CLAW[leg])
        # joint ids, qpos addrs, dof addrs for the SOLVED joints
        self.joint_ids = []
        self.qadr = []
        self.dof = []
        for jname in SOLVE_JOINTS:
            full = "walker/%s_%s" % (jname, leg)
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, full)
            if jid < 0:
                raise KeyError("joint %s not in model" % full)
            self.joint_ids.append(jid)
            self.qadr.append(self.m.jnt_qposadr[jid])
            self.dof.append(self.m.jnt_dofadr[jid])
        # ALL joints of this limb (incl locked + tip tarsus3/4/5) for rigid
        # pinning, so a claw wedged among keys cannot deflect the chain
        # off-command from the planned pose.
        tail = "_" + leg
        self.all_joint_names = []
        self.all_joint_ids = []
        self.all_qadr = []
        self.all_dof = []
        for jid in range(self.m.njnt):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if nm.endswith(tail) and nm.startswith("walker/"):
                self.all_joint_names.append(nm)
                self.all_joint_ids.append(jid)
                self.all_qadr.append(self.m.jnt_qposadr[jid])
                self.all_dof.append(self.m.jnt_dofadr[jid])
        # rest values for the locked twist/abduct + tip joints
        self.lock_val = {}
        for jname in _LOCK:
            full = "walker/%s_%s" % (jname, leg)
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, full)
            self.lock_val[full] = float(self.d.qpos[self.m.jnt_qposadr[jid]])
        for full in self.all_joint_names:
            if full not in self.lock_val and not any(
                    full.endswith("_%s_%s" % (j, leg)) for j in SOLVE_JOINTS):
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, full)
                self.lock_val[full] = float(self.d.qpos[self.m.jnt_qposadr[jid]])

    def full_q(self, qsolve):
        """Per-limb joint vector: solve joints at qsolve, all others at rest."""
        suf = "_" + self.leg
        out = []
        for full in self.all_joint_names:
            short = full[len("walker/"):-len(suf)]
            if short in SOLVE_JOINTS:
                out.append(float(qsolve[SOLVE_JOINTS.index(short)]))
            else:
                out.append(self.lock_val[full])
        return np.array(out)

    def tip(self, q=None):
        """Claw tip site world position. If q (radians for solved joints) is
        given, write it into the limb qpos first (caller re-forwards)."""
        self.write(q)
        self._fwd()
        return self.d.site_xpos[self.site_id].copy()

    def _fwd(self):
        mujoco.mj_forward(self.m, self.d)

    def jac(self):
        """3xN site Jacobian (columns = solved joint dofs)."""
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, None, self.site_id)
        return jacp[:, self.dof]

    def write(self, q):
        if q is None:
            return
        for adr, v in zip(self.qadr, q):
            self.d.qpos[adr] = float(v)

    def clamp(self, q):
        q = np.asarray(q, dtype=float)
        for i, jid in enumerate(self.joint_ids):
            rng = self.m.jnt_range[jid]
            if rng[0] < rng[1]:
                q[i] = np.clip(q[i], rng[0], rng[1])
        return q

    def restq(self):
        return np.array([self.d.qpos[a] for a in self.qadr])

    def solve(self, target, q0=None, max_iter=80, tol=5e-5, lam=1e-3):
        """Damp the claw tip onto `target` (world xyz).

        Returns (ok, q, err, tip) where ok means the tight tolerance was hit
        EXACTLY; err is the best error found. Use `feasible()` for a practical
        "can this leg press that key" gate.
        """
        target = np.asarray(target, dtype=float)
        if q0 is None:
            q0 = self.restq()
        q = self.clamp(np.asarray(q0, float)).copy()
        best = (False, q, np.inf, self._tip_at(q).copy())
        for _ in range(max_iter):
            self.write(q)
            self._fwd()
            tip = self.d.site_xpos[self.site_id].copy()
            e = target - tip
            err = float(np.linalg.norm(e))
            if err < best[2]:
                best = (False, q.copy(), err, tip.copy())
            if err <= tol:
                return True, q.copy(), err, tip.copy()
            J = self.jac()
            # damped least squares: minimize |J dq - e|^2 + lam^2 |dq|^2
            A = J @ J.T + lam * lam * np.eye(3)
            dq = J.T @ np.linalg.solve(A, e)
            qnew = self.clamp(q + dq)
            if np.allclose(qnew, q, atol=1e-9):
                break
            q = qnew
        return best[0], best[1], best[2], best[3]

    def feasible(self, target, atol=0.003, **kw):
        """Practical reachability: can this leg place its claw within `atol`
        m (default 3 mm) of `target`? Returns (True, q, err) or (False, None,
        best_err)."""
        ok, q, err, _ = self.solve(target, **kw)
        return (ok or err <= atol), (q if (ok or err <= atol) else None), err

    def _tip_at(self, q):
        self.write(q)
        self._fwd()
        return self.d.site_xpos[self.site_id].copy()

    def action(self, q):
        """Full 59-action holding this limb at q (rest elsewhere)."""
        a = np.full(59, 0.5, dtype=np.float64)
        a[0:6] = 0.7  # all claws grip unless caller clears them
        for k, jname in enumerate(s4.LEG_JOINTS):
            if jname in SOLVE_JOINTS:
                a[self.base + k] = q[SOLVE_JOINTS.index(jname)]
            else:  # locked joints keep their rest value
                a[self.base + k] = self.lock_val[
                    "walker/%s_%s" % (jname, self.leg)]
        return a


SEGMENTS = {}


def get_ik(env, leg):
    if leg not in SEGMENTS:
        SEGMENTS[leg] = LegIK(env, leg)
    return SEGMENTS[leg]


_CLAW_SLOT = {
    "T1_left": s4.CLAW_T1L, "T1_right": s4.CLAW_T1R,
    "T2_left": s4.CLAW_T2L, "T2_right": s4.CLAW_T2R,
}


def compose_action(ik, q, a_base=None, release_claw=True):
    """Full 59-action holding THIS limb at q on top of a_base (default: a
    neutral-voltage stance), with the limb's claw released if requested."""
    import numpy as _np
    base = a_base if a_base is not None else _np.full(59, 0.5)
    a = _np.array(base, dtype=_np.float64, copy=True)
    for k, jname in enumerate(s4.LEG_JOINTS):
        if jname in SOLVE_JOINTS:
            a[ik.base + k] = q[SOLVE_JOINTS.index(jname)]
        else:
            a[ik.base + k] = ik.lock_val["walker/%s_%s" % (jname, ik.leg)]
    if release_claw:
        a[_CLAW_SLOT[ik.leg]] = 0.0
    return a


def kinematic_step(env, ik, q, action, zero_dofs=True):
    """Write `q` into the limb's joint qpos (kinematic pin) then env.step.

    The active leg is pinned in ALL its 8 joints (solve + locked) with their
    dof velocities zeroed, so contact forces cannot push the claw off its
    planned point -- the claw is exactly where the IK says. Returns the env
    after one step.
    """
    d = env.physics.data
    q8 = ik.full_q(q)
    for adr, v in zip(ik.all_qadr, q8):
        d.qpos[adr] = float(v)
    if zero_dofs:
        for dof in ik.all_dof:
            d.qvel[dof] = 0.0
    return env.step(action)