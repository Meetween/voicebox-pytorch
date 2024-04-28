import subprocess
import os
import torch

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()  # get the number of GPUs

    processes = []
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)  # set the CUDA_VISIBLE_DEVICES environment variable
        process = subprocess.Popen(["python", "encode_audio.py", "--split_index", str(i)], env=env)
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()
