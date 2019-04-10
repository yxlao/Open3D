from pathlib import Path

root_dir = Path("/home/ylao/repo/Open3D/src")


def fix_angle_brackets(lines):
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        if "#include <" in line:
            print(f"Convert: {line}")
            line = line.replace('<', '"')
            line = line.replace('>', '"')
            print(f"To: {line}")
            lines[line_idx] = line
    return lines


def fix_self_include_header(lines, file_path):
    header_file_name = file_path.stem + ".h"
    header_file_relative_path = file_path.relative_to(
        root_dir).parent / header_file_name

    src_line = f"#include \"{header_file_name}\"\n"
    dst_line = f"#include \"{header_file_relative_path}\"\n"

    for line_idx in range(len(lines)):
        if lines[line_idx] == src_line:
            print(f"Convert: {src_line}")
            print(f"To: {dst_line}")
            lines[line_idx] = dst_line
    return lines


def process_file(file_path):
    # print(f"Processing {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = fix_self_include_header(lines, file_path)
    lines = fix_angle_brackets(lines)
    # lines = fix_relative_paths(lines, file_path)

    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    file_paths = list(root_dir.glob("**/*.cpp"))
    file_paths = file_paths + list(root_dir.glob("**/*.h"))
    for file_path in file_paths:
        process_file(file_path)
