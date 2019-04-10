from pathlib import Path


def process_file(file_path):
    print(f"Processing {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()

    processed_lines = lines
    with open(file_path, "w") as f:
        for processed_line in processed_lines:
            f.write(processed_line)


if __name__ == "__main__":
    root_dir = Path("/home/ylao/repo/Open3D/src")
    file_paths = list(root_dir.glob("**/*.cpp"))
    file_paths = file_paths + list(root_dir.glob("**/*.h"))
    for file_path in file_paths:
        process_file(file_path)
