# train_all.py
import os
import sys
import subprocess

def main():
    # user parameters (edit or get from sys.argv if needed)
    k_query = 5
    model_ver = 180
    epoch = 400
    training = 1

    # build commands
    cmd_asv = f"python DMD_ASV.py --k_query={k_query} --model_ver={model_ver} --epoch={epoch} --is_training={training}"
    cmd_asv2 = f"python DMD_ASV2.py --k_query={k_query} --model_ver={model_ver} --epoch={epoch} --is_training={training}"
    cmd_psl = f"python DMD_psl.py --k_query={k_query} --model_ver={model_ver} --epoch={epoch} --is_training={training}"

    print(f"🚀 Running: {cmd_asv}")
    subprocess.run(cmd_asv, shell=True, check=True)

    print(f"🚀 Running: {cmd_asv2}")
    subprocess.run(cmd_asv2, shell=True, check=True)

    print(f"🚀 Running: {cmd_psl}")
    subprocess.run(cmd_psl, shell=True, check=True)

if __name__ == "__main__":
    # Ensure current directory is importable
    sys.path.insert(0, os.getcwd())
    main()
