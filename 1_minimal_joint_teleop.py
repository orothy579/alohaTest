#!/usr/bin/env python3
"""
ALOHA bimanual tele-op  –  각 팔 그리퍼 분리 (2025-04-26)
  • Q/A … P/; \|  : 6-DoF × 2 팔 관절 증분
  • Z/X           : 왼쪽 그리퍼 열기 / 닫기
  • ,/.           : 오른쪽 그리퍼 열기 / 닫기
  • V             : ctrl · qpos 디버그 출력
"""
import mujoco
import glfw
import numpy as np
from pathlib import Path

# ──────────────── 모델 로드 ───────────────────────────
XML = Path(
    "/Users/lch/development/Robot/act/assets/bimanual_viperx_transfer_cube.xml")
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

# ──────────────── GLFW 초기화 ─────────────────────────
if not glfw.init():
    raise RuntimeError("GLFW init failed")
window = glfw.create_window(960, 720, "ALOHA tele-op", None, None)
glfw.make_context_current(window)

# ──────────────── 헬퍼 함수 ───────────────────────────


def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)


def aid_safe(j):
    """조인트-ID → 액추에이터-ID (없으면 None)"""
    idx = np.where(model.actuator_trnid[:, 0] == j)[0]
    return int(idx[0]) if idx.size else None


def safe_range(a):
    lo, hi = model.actuator_ctrlrange[a]
    return (-np.inf, np.inf) if (lo == hi == 0) else (lo, hi)

# ──────────────── 주요 인덱스 구하기 ──────────────────


def L(n): return f"vx300s_left/{n}"
def R(n): return f"vx300s_right/{n}"


joint_names = ["waist", "shoulder", "elbow",
               "roll", "wrist_angle", "wrist_rotate"]
LJ = [jid(L(n)) for n in joint_names]
RJ = [jid(R(n)) for n in joint_names]

# 각 손가락 슬라이드 (왼손·오른손 각각 2개)
A_L_LF = aid_safe(jid(L("left_finger")))    # + 범위
A_L_RF = aid_safe(jid(L("right_finger")))   # – 범위
A_R_LF = aid_safe(jid(R("left_finger")))    # + 범위
A_R_RF = aid_safe(jid(R("right_finger")))   # – 범위

# ──────────────── 팔 관절 키 매핑 ─────────────────────
DELTA = 0.10
KEYMAP = {}


def add_pair(kplus, kminus, j_id):
    a = aid_safe(j_id)
    if a is not None:
        KEYMAP[kplus] = (a, +DELTA)
        KEYMAP[kminus] = (a, -DELTA)


# 왼팔 (QWERTY 왼쪽)
keys_left = [(glfw.KEY_Q, glfw.KEY_A), (glfw.KEY_W, glfw.KEY_S),
             (glfw.KEY_E, glfw.KEY_D), (glfw.KEY_R, glfw.KEY_F),
             (glfw.KEY_T, glfw.KEY_G), (glfw.KEY_Y, glfw.KEY_H)]
for (kp, km), j in zip(keys_left, LJ):
    add_pair(kp, km, j)

# 오른팔 (QWERTY 오른쪽)
keys_right = [(glfw.KEY_U, glfw.KEY_J), (glfw.KEY_I, glfw.KEY_K),
              (glfw.KEY_O, glfw.KEY_L), (glfw.KEY_P, glfw.KEY_SEMICOLON),
              (glfw.KEY_LEFT_BRACKET, glfw.KEY_APOSTROPHE),
              (glfw.KEY_RIGHT_BRACKET, glfw.KEY_BACKSLASH)]
for (kp, km), j in zip(keys_right, RJ):
    add_pair(kp, km, j)

# ──────────────── 그리퍼 스캔코드 매핑 ────────────────
# Mac-US 물리: Z=6 X=7 ,=43 .=47
GRIP_SC = {
    6:  ((A_L_LF, +0.057), (A_L_RF, -0.057)),   # 왼손 열기
    7:  ((A_L_LF, +0.021), (A_L_RF, -0.021)),   # 왼손 닫기
    43: ((A_R_LF, +0.057), (A_R_RF, -0.057)),   # 오른손 열기
    47: ((A_R_LF, +0.021), (A_R_RF, -0.021)),   # 오른손 닫기
}

# ──────────────── ctrl 버퍼 & 파라미터 조정 ───────────
target = np.zeros(model.nu)
for a in range(model.nu):
    j = model.actuator_trnid[a, 0]
    if j >= 0:
        target[a] = data.qpos[j]        # 현재 자세로 초기화

for a, rng in (
    (A_L_LF, (0.021, 0.057)), (A_L_RF, (-0.057, -0.021)),
        (A_R_LF, (0.021, 0.057)), (A_R_RF, (-0.057, -0.021))):
    if a is None:
        continue
    model.actuator_ctrlrange[a] = rng
    model.actuator_gainprm[a, 0] *= 8        # 기본 kp ×8 (과도증폭 방지)
    model.actuator_forcelimited[a] = 0

mujoco.mj_forward(model, data)

# ──────────────── 키 콜백 ────────────────────────────


def on_key(win, key, sc, act, mods):
    if act == glfw.PRESS:
        print(f"key {key}  sc {sc}")
    if act not in (glfw.PRESS, glfw.REPEAT):
        return

    # 디버그 V
    if key == glfw.KEY_V:
        idxs = [jid(L("left_finger")), jid(R("left_finger")),
                jid(L("right_finger")), jid(R("right_finger"))]
        print("ctrl:", np.round(data.ctrl[[A_L_LF, A_L_RF, A_R_LF, A_R_RF]], 3),
              " qpos:", np.round(data.qpos[idxs], 3))
        return

    # 그리퍼
    if sc in GRIP_SC:
        for a, v in GRIP_SC[sc]:
            if a is None:
                continue
            lo, hi = safe_range(a)
            target[a] = np.clip(v, lo, hi)
        return

    # 팔 관절
    if key in KEYMAP:
        a, d = KEYMAP[key]
        lo, hi = safe_range(a)
        target[a] = np.clip(target[a]+d, lo, hi)


glfw.set_key_callback(window, on_key)

# ──────────────── 카메라 & 렌더 ───────────────────────
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10_000)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = model.stat.center
cam.distance, cam.azimuth, cam.elevation = 1.2*model.stat.extent, 90, -40

# ──────────────── 메인 루프 ──────────────────────────
while not glfw.window_should_close(window):
    data.ctrl[:] = target
    mujoco.mj_step(model, data)

    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 960, 720), scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
