import mujoco
import sys
m = mujoco.MjModel.from_xml_path(
    "/Users/lch/development/Robot/act/assets/bimanual_viperx_ee_insertion.xml")

for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"{j:2d}  {name}")
