"""
run experiment: sequentially call simulate.py with multiple sets of arguments
"""

import subprocess
import os


def run_experiment(arg_set):
    cmd = ["python", "./utils/simulate_fids.py"]

    for key, value in arg_set.items():
        cmd.append(f"--{key}")
        if isinstance(value, list):
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(str(value))

    subprocess.run(cmd)


prompts = {
    "sunset1": "The sun sets behind the mountains.",
    "sunset1-copy": "The sun sets behind the mountains.",
    "sunset2-copy2": "The mountains with sunset behind.",
    "stars": "The mountains with a night sky full of shining stars.",
}

save_folder = "./results/fid_by_prompts/"

devices = [0, 1, 2, 3]
monte_carlo_size = 200

if __name__ == "__main__":
    # Define multiple sets of arguments to try
    arg_sets = [
        {
            "prompt1": prompt,
            "prompt2": prompt,
            "save_folder": os.path.join(save_folder, keyword),
            "devices": devices,
            "monte_carlo_size": monte_carlo_size,
        }
        for keyword, prompt in prompts.items()
    ]

    for arg_set in arg_sets:
        run_experiment(arg_set)
