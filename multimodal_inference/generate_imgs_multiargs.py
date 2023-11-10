"""
Run generate_imgs.py with multiple sets of arguments.
"""


import subprocess
import os


def run_experiment(arg_set):
    cmd = ["python", "./utils/generate_imgs.py"]

    for key, value in arg_set.items():
        cmd.append(f"--{key}")
        if isinstance(value, list):
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(str(value))

    subprocess.run(cmd)


# Define multiple sets of arguments to try

prompts = {
    "sunset1": "The sun sets behind the mountains.",
    "sunset1-copy": "The sun sets behind the mountains.",
    "sunset2-copy2": "The mountains with sunset behind.",
    "stars": "The mountains with a night sky full of shining stars.",
}
guidance_scale = 2
device_ids = [0, 1, 2, 3]
total_num_imgs = 200
plot_folder = "./data/"


if __name__ == "__main__":
    # Define multiple sets of arguments to try
    arg_sets = [
        {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "total_num_imgs": total_num_imgs,
            "plot_folder": os.path.join(plot_folder, keyword),
            "device_ids": device_ids,
        }
        for keyword, prompt in prompts.items()
    ]

    for arg_set in arg_sets:
        run_experiment(arg_set)
