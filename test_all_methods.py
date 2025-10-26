# train_all.py
import os
import sys
import subprocess
from exp_method import *

def main():
    # user parameters (edit or get from sys.argv if needed)
    k_query = 5
    model_ver = 180
    testing = 0
    sparse_level = [0.1]

    # build commands
    cmd_asv = f"python DMD_ASV.py --k_query={k_query} --model_ver={model_ver} --is_training={testing}"
    cmd_asv2 = f"python DMD_ASV2.py --k_query={k_query} --model_ver={model_ver} --is_training={training}"
    cmd_psl = f"python DMD_psl.py --k_query={k_query} --model_ver={model_ver} --is_training={testing}"

    print(f"🚀 Running: {cmd_asv}")
    subprocess.run(cmd_asv, shell=True, check=True)

    print(f"🚀 Running: {cmd_asv2}")
    subprocess.run(cmd_asv2, shell=True, check=True)

    print(f"🚀 Running: {cmd_psl}")
    subprocess.run(cmd_psl, shell=True, check=True)

    print(f"🚀 Running SVD, SSA, LASSO test on psl")
    psl_data = np.load('base_ts_171_psl.npy', allow_pickle=True)
    exp_method.unsup_methods_test(psl_data, k_query, sparse_level, top_ks, tag='psl')

    print(f"🚀 Running SVD, SSA, LASSO test on ASV")
    asv_data = np.load('base_ts_150_ASV.npy', allow_pickle=True)
    exp_method.unsup_methods_test(asv_data, k_query, sparse_level, top_ks, tag='ASV')

    print(f"🚀 Running SVD, SSA, LASSO test on ASV2")
    asv2_data = np.load('base_ts_150_ASV2.npy', allow_pickle=True)
    exp_method.unsup_methods_test(asv2_data, k_query, sparse_level, top_ks, tag='ASV2')

if __name__ == "__main__":
    # Ensure current directory is importable
    sys.path.insert(0, os.getcwd())
    main()
