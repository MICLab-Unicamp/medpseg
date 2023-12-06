'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

This is where the monitoring of hardware happens, feeding the GUI bars with data
'''
import psutil
import GPUtil


def get_stats():
    '''
    Collects GPU, VRAM, CPU and RAM stats
    '''
    stats = {"gpu": 0, "gpu_ram": 0, "cpu": 0, "cpu_ram": 0}
    try:
        gpu = GPUtil.getGPUs()[0]
        stats["gpu"] = int(gpu.load*100)
        stats["gpu_ram"] = int(gpu.memoryUtil*100)
    except Exception as e:
        print(f"GPU Monitoring error: {e}")

    try:
        stats["cpu"] = int(psutil.cpu_percent())
        stats["cpu_ram"] = int(psutil.virtual_memory().percent)
    except Exception as e:
        print(f"CPU Monitoring error: {e}")
        
    return stats

if __name__ == "__main__":
    stats = get_stats()
    print(stats)
