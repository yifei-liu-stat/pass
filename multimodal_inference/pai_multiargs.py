"""
PAI for multiple sets of arguments.
"""

import subprocess
import os


def run_experiment(arg_set):
    cmd = ["python", "./utils/pai.py"]

    for key, value in arg_set.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    subprocess.run(cmd)


if __name__ == "__main__":
    # Define multiple sets of arguments to try
    feature_dimension = 192
    sample_size = 200
    imgs_folder = "./data/"

    comparison_keywords = [
        ("sunset1", "sunset1-copy"),
        ("sunset1", "sunset2-copy2"),
        ("sunset1", "stars"),
        ("sunset2-copy2", "stars"),
    ]

    comparison_folders = [
        (imgs_folder + keyword1, imgs_folder + keyword2)
        for keyword1, keyword2 in comparison_keywords
    ]
    arg_sets = [
        {
            "feature_dimension": feature_dimension,
            "sample_size": sample_size,
            "image_folder_one": folder1,
            "image_folder_two": folder2,
            "fid_folder": "./results/fid_by_prompts/",
        }
        for folder1, folder2 in comparison_folders
    ]
    for arg_set in arg_sets:
        run_experiment(arg_set)
