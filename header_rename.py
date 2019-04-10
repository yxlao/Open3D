from pathlib import Path
import re

root_dir = Path("/home/ylao/repo/Open3D/src")


def fix_angle_brackets(lines):
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        if "#include <Open3D" in line:
            print(f"Convert: {line}")
            line = line.replace('<', '"')
            line = line.replace('>', '"')
            print(f"To: {line}")
            lines[line_idx] = line
    return lines


def fix_self_include_header(lines, file_path):
    header_file_name = file_path.stem + ".h"
    header_relative_path = file_path.relative_to(
        root_dir).parent / header_file_name

    src_line = f"#include \"{header_file_name}\"\n"
    dst_line = f"#include \"{header_relative_path}\"\n"

    for line_idx in range(len(lines)):
        if lines[line_idx] == src_line:
            print(f"Convert: {src_line}")
            print(f"To: {dst_line}")
            lines[line_idx] = dst_line
    return lines


def fix_header_relative_paths(lines,
                              map_header_file_name_to_relative_path):
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        match = re.search('#include "(.*)"', line)
        if match:
            header_file_name = match.group(1)
            if header_file_name in map_header_file_name_to_relative_path:
                header_relative_path = map_header_file_name_to_relative_path[
                    header_file_name]
                print(f"Convert: {line}")
                line = f"#include \"{header_relative_path}\"\n"
                print(f"To: {line}")
                lines[line_idx] = line
    return lines


def process_file(file_path, map_header_file_name_to_relative_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = fix_self_include_header(lines, file_path)
    lines = fix_angle_brackets(lines)
    lines = fix_header_relative_paths(lines,
                                      map_header_file_name_to_relative_path)

    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    cpp_file_paths = list(root_dir.glob("**/*.cpp"))
    header_file_paths = list(root_dir.glob("**/*.h"))
    file_paths = cpp_file_paths + header_file_paths

    map_header_file_name_to_relative_path = dict()
    for header_file_path in header_file_paths:
        header_file_name = header_file_path.name
        header_relative_path = header_file_path.relative_to(root_dir)
        if header_file_name in map_header_file_name_to_relative_path:
            raise ValueError(f"Repeated header file name {header_file_name}")
        else:
            map_header_file_name_to_relative_path[
                header_file_name] = str(header_relative_path)
    print(map_header_file_name_to_relative_path)

    for file_path in file_paths:
        process_file(file_path, map_header_file_name_to_relative_path)
