# This script is intended to generate "device_aspect_macros.h" from "aspects.def" and "aspects_deprecated.def".

import os
import sys


def process_aspects(file_path, macro_name):
    with open(file_path, "r") as file:
        content = file.read()
    lines = content.strip().splitlines()

    output = ""
    for line in lines:
        if not line.startswith("__SYCL_ASPECT"):
            continue
        macro_args = line[line.index("(") + 1:line.rindex(")")]
        aspect_name, aspect_number, *_ = macro_args.split(", ", 2)

        output += f"// {macro_name}({aspect_name}, {aspect_number})\n"
        output += f"#ifndef __SYCL_ALL_DEVICES_HAVE_{aspect_name}__\n"
        output += f"#define __SYCL_ALL_DEVICES_HAVE_{aspect_name}__ 0\n"
        output += "#endif\n"
        output += f"#ifndef __SYCL_ANY_DEVICE_HAS_{aspect_name}__\n"
        output += f"#define __SYCL_ANY_DEVICE_HAS_{aspect_name}__ 0\n"
        output += "#endif\n\n"

    return output

header_output = "" # to simplify internal customizations
header_output += """//==------------------- device_aspect_macros.hpp - SYCL device -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IMPORTANT: device_aspect_macros.hpp is a generated file - DO NOT EDIT
//            original definitions are in aspects.def & aspects_deprecated.def
//            See: sycl/source/device_aspect_macros_generator.py
//

#pragma once\n
"""

include_sycl_dir = sys.argv[1]
build_include_sycl_dir = sys.argv[2]

header_output += process_aspects(
    os.path.join(include_sycl_dir, "info/aspects_deprecated.def"), "__SYCL_ASPECT_DEPRECATED"
)
header_output += process_aspects(os.path.join(include_sycl_dir, "info/aspects.def"), "__SYCL_ASPECT")

output_path = os.path.join(build_include_sycl_dir, "device_aspect_macros.hpp")
with open(output_path, "w") as header_file:
    header_file.write(header_output)
