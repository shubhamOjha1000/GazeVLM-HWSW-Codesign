"""Write the IMU -> CPF rotation from one Aria VRS, for `--imu_rot calib`.

    python -m eyehead.imu_calib --vrs /content/raw/<seq>/<file>.vrs \
                                --out /content/drive/MyDrive/GazeVLM/imu_calib.json

The rotation comes from the factory calibration stored in the VRS:

    R_cpf<-imu = R_device<-cpf^T  R_device<-imu

`imu-right` is the default because `src.common.aria_io.load_imu_from_vrs` reads stream
1202-1, the right IMU, whenever it exists -- the label here must match the stream the CSV
was built from. Aria devices share one mechanical design, so a rotation from one recording
is a close approximation for the others (factory differences are typically a degree or
two); pass several VRS files to see the spread.

As with the gaze projector, this uses the factory device<->cpf and device<->imu extrinsics
rather than a CPF shortcut that can return nominal CAD values.
"""

import argparse
import json

import numpy as np


def rotation_cpf_imu(vrs_path, imu_label="imu-right"):
    from projectaria_tools.core import data_provider
    provider = data_provider.create_vrs_data_provider(vrs_path)
    calib = provider.get_device_calibration()
    T_device_cpf = calib.get_transform_device_cpf()
    imu = calib.get_imu_calib(imu_label)
    if imu is None:
        raise KeyError(f"{vrs_path}: no IMU calibration named {imu_label!r}; "
                       f"have {list(calib.get_imu_labels())}")
    T_device_imu = imu.get_transform_device_imu()
    T_cpf_imu = T_device_cpf.inverse() @ T_device_imu
    return np.asarray(T_cpf_imu.rotation().to_matrix(), dtype=np.float64).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser("IMU -> CPF rotation from Aria factory calibration")
    ap.add_argument("--vrs", nargs="+", required=True)
    ap.add_argument("--imu_label", default="imu-right")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Rs = []
    for v in a.vrs:
        R = rotation_cpf_imu(v, a.imu_label)
        Rs.append(R)
        print(f"{v}\n{np.round(R, 4)}\n")
    if len(Rs) > 1:
        ang = [np.degrees(np.arccos(np.clip((np.trace(Rs[0].T @ R) - 1) / 2, -1, 1)))
               for R in Rs[1:]]
        print(f"angle from the first recording's rotation: "
              f"{', '.join(f'{x:.2f} deg' for x in ang)}")

    R = Rs[0]
    print("which raw gyro axis drives each eye-convention head rate:")
    print(f"   head_yaw   =  {np.round(R[1], 3).tolist()} . w_imu   (+ = turning left)")
    print(f"   head_pitch = {np.round(-R[0], 3).tolist()} . w_imu   (+ = nodding up)")
    json.dump(dict(R_cpf_imu=R.tolist(), imu_label=a.imu_label, source_vrs=a.vrs[0],
                   n_vrs=len(Rs)), open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
