from pathlib import Path
import re
import subprocess
from pprint import pprint

root_dir = Path("/home/ylao/repo/Open3D/src")


def format_file(file_path):
    cmd = ["/usr/bin/clang-format-5.0", "-i", str(file_path)]
    subprocess.run(cmd)


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


def sort_includes(lines, file_path):
    header_file_name = file_path.stem + ".h"
    header_relative_path = file_path.relative_to(
        root_dir).parent / header_file_name

    start_line_idx = -1
    end_line_idx = -1
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        include_str = "#include "
        if line[:len(include_str)] == include_str:
            if start_line_idx == -1:
                start_line_idx = line_idx
            end_line_idx = line_idx

    if start_line_idx != -1:
        before_header_lines = lines[:start_line_idx]
        header_lines = lines[start_line_idx:end_line_idx + 1]
        after_header_lines = lines[end_line_idx + 1:]

        self_header_line = None
        self_header_line_idx = None
        for header_line_idx in range(len(header_lines)):
            header_line = header_lines[header_line_idx]
            match = re.search('#include "(.*)"\n', header_line)
            if match and str(header_relative_path) == match.group(1):
                self_header_line = header_line
                self_header_line_idx = header_line_idx
                break
        if self_header_line_idx is not None:
            del header_lines[self_header_line_idx]

        # print("Before ==========")
        # pprint(header_lines)
        # print("=================")

        abort_change = False
        external_header_lines = []
        open3d_header_lines = []
        for header_line in header_lines:
            if "#include <" in header_line:
                external_header_lines.append(header_line)
            elif '#include "' in header_line:
                open3d_header_lines.append(header_line)
            elif 'def' in header_line:
                abort_change = True

        if ("UnitTest" in str(file_path)):
            external_header_lines = list(sorted(external_header_lines))
            open3d_header_lines = list(sorted(open3d_header_lines))

        header_lines = []
        if self_header_line is not None:
            header_lines.append(self_header_line)
        if len(external_header_lines) > 0:
            header_lines.append("\n")
            header_lines += external_header_lines
        if len(open3d_header_lines) > 0:
            header_lines.append("\n")
            header_lines += open3d_header_lines

        # print("After ==========")
        # pprint(header_lines)
        # print("================")

        if not abort_change:
            lines = before_header_lines + header_lines + after_header_lines

    return lines


def process_file(file_path, map_header_file_name_to_relative_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    before_text = "".join(lines)

    lines = fix_self_include_header(lines, file_path)
    lines = fix_angle_brackets(lines)
    lines = fix_header_relative_paths(lines,
                                      map_header_file_name_to_relative_path)
    lines = sort_includes(lines, file_path)

    with open(file_path, "w") as f:
        for line in lines:
            f.write(line)

    after_text = "".join(lines)
    if before_text != after_text:
        format_file(file_path)


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

    for file_path in file_paths:
        process_file(file_path, map_header_file_name_to_relative_path)
