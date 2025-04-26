#!/usr/bin/env python3
"""
Minimal ALOHA bimanual tele-op:
  • Q-A / … / P-; : 6 DoF × 2 팔 관절 증분 제어
  • Z/X , ,/.     : 그리퍼 완전 열기 / 완전 닫기
  • camera        : 모델 중심 고정 FREE 카메라
"""
import mujoco
import glfw
import numpy as np
from pathlib import Path

# ---------- 모델 로드 ----------------------------------------------------------
XML = Path(
    "/Users/lch/development/Robot/act/assets/bimanual_viperx_transfer_cube.xml")
model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

# ---------- GLFW 윈도우 --------------------------------------------------------
if not glfw.init():
    raise RuntimeError("GLFW init failed")
window = glfw.create_window(960, 720, "ALOHA tele-op", None, None)
glfw.make_context_current(window)

# ---------- 조인트 인덱스 ------------------------------------------------------
J = {
    # 왼팔 0-7
    'L_WAIST': 0, 'L_SHOUL': 1, 'L_ELB': 2, 'L_ROLL': 3,
    'L_WRANG': 4, 'L_WROT': 5, 'L_LF': 6, 'L_RF': 7,
    # 오른팔 8-15
    'R_WAIST': 8, 'R_SHOUL': 9, 'R_ELB': 10, 'R_ROLL': 11,
    'R_WRANG': 12, 'R_WROT': 13, 'R_LF': 14, 'R_RF': 15,
}

# ---------- 매핑: joint-id → actuator-id --------------------------------------
JOINT2ACT = {j: a for a, (j, _) in enumerate(model.actuator_trnid) if j >= 0}

# ---------- 제어 스텝 ----------------------------------------------------------
DELTA_JOINT = 0.10        # rad
# 그리퍼 절대 위치 (XML ctrlrange = 0.021~0.057 / -0.057~-0.021) :contentReference[oaicite:1]{index=1}
OPEN_L, CLOSE_L = 0.057, 0.021
OPEN_R, CLOSE_R = -0.021, -0.057

KEYMAP = {
    # 왼팔
    glfw.KEY_Q: (J['L_WAIST'], +DELTA_JOINT), glfw.KEY_A: (J['L_WAIST'], -DELTA_JOINT),
    glfw.KEY_W: (J['L_SHOUL'], +DELTA_JOINT), glfw.KEY_S: (J['L_SHOUL'], -DELTA_JOINT),
    glfw.KEY_E: (J['L_ELB'],   +DELTA_JOINT), glfw.KEY_D: (J['L_ELB'],   -DELTA_JOINT),
    glfw.KEY_R: (J['L_ROLL'],  +DELTA_JOINT), glfw.KEY_F: (J['L_ROLL'],  -DELTA_JOINT),
    # 오른팔
    glfw.KEY_U: (J['R_WAIST'], +DELTA_JOINT), glfw.KEY_J: (J['R_WAIST'], -DELTA_JOINT),
    glfw.KEY_I: (J['R_SHOUL'], +DELTA_JOINT), glfw.KEY_K: (J['R_SHOUL'], -DELTA_JOINT),
    glfw.KEY_O: (J['R_ELB'],   +DELTA_JOINT), glfw.KEY_L: (J['R_ELB'],   -DELTA_JOINT),
    glfw.KEY_P: (J['R_ROLL'],  +DELTA_JOINT), glfw.KEY_SEMICOLON: (J['R_ROLL'], -DELTA_JOINT),
}

GRIP_KEYS = {
    # 왼손 (Z: 열기, X: 닫기)
    glfw.KEY_Z:  (J['L_LF'], OPEN_L,  J['L_RF'], CLOSE_R),
    glfw.KEY_X:  (J['L_LF'], CLOSE_L, J['L_RF'], OPEN_R),
    # 오른손 (, : 열기, . : 닫기)
    glfw.KEY_COMMA:  (J['R_LF'], OPEN_L,  J['R_RF'], CLOSE_R),
    glfw.KEY_PERIOD: (J['R_LF'], CLOSE_L, J['R_RF'], OPEN_R),
}

# ---------- 목표(ctrl) 버퍼 초기화 --------------------------------------------
target = np.zeros(model.nu)
for a in range(model.nu):
    j = model.actuator_trnid[a, 0]
    if j >= 0:
        target[a] = data.qpos[j]          # 현재 자세로 동기화
mujoco.mj_forward(model, data)            # 한 프레임 일관성 갱신

# ---------- 키 콜백 -----------------------------------------------------------


def on_key(win, key, sc, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    # 그리퍼 절대 제어
    if key in GRIP_KEYS:
        jl, val_l, jr, val_r = GRIP_KEYS[key]
        target[JOINT2ACT[jl]] = val_l
        target[JOINT2ACT[jr]] = val_r
        return

    # 관절 증분 제어
    if key in KEYMAP:
        j, delta = KEYMAP[key]
        a = JOINT2ACT[j]
        # 안전 범위 :contentReference[oaicite:2]{index=2}
        lo, hi = model.actuator_ctrlrange[a]
        target[a] = np.clip(target[a] + delta, lo, hi)


glfw.set_key_callback(window, on_key)

# ---------- 카메라 ------------------------------------------------------------
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10_000)
mujoco.mjv_defaultCamera(cam)

cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# 모델 중심 :contentReference[oaicite:3]{index=3}
cam.lookat[:] = model.stat.center
cam.distance = 1.5 * model.stat.extent
cam.azimuth = 90
cam.elevation = -20

# ---------- 메인 루프 ---------------------------------------------------------
while not glfw.window_should_close(window):
    # servo 목표 주입 (자동 clamp) :contentReference[oaicite:4]{index=4}
    data.ctrl[:] = target
    mujoco.mj_step(model, data)

    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 960, 720), scene, ctx)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
