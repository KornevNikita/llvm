//===----- sycl-aspect-filter.cpp - Utility to filter the file table ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool transforms an input file table by removing rows with device code
// files that use features unsupported for the target architecture given as
// tool's argument.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SimpleTable.h"

#include <iostream>

using namespace llvm;

cl::opt<std::string> InputFilename(cl::Positional, cl::value_desc("filename"),
                                   cl::desc("Input filename"));

cl::opt<std::string> Output("o", cl::value_desc("filename"),
                            cl::desc("Output filename"));

cl::opt<std::string>
    Target("target", cl::value_desc("target"),
           cl::desc("Target device architecture to filter for"));

cl::opt<std::string>
    DeviceConfigFile("device-config-file", cl::value_desc("path"),
                     cl::desc("Path to the device configuration file"));

static void error(const Twine &Message) {
  errs() << "sycl-aspect-filter: " << Message << '\n';
  exit(1);
}

void filterTable(util::SimpleTable &Table) {
  std::string OutputFileName =
      Output.empty() ? sys::path::stem(InputFilename).str() + "_filtered" +
                           sys::path::extension(InputFilename).str()
                     : Output;

  // Copy the input table to the output if it doesn't contain Properties
  if (Table.getColumnId("Properties") == -1) {
    std::error_code EC;
    raw_fd_ostream OS{OutputFileName, EC, sys::fs::OpenFlags::OF_None};
    if (EC)
      error("Can't open the output file " + OutputFileName);
    Table.write(OS, /*WriteTitles=*/true);
    return;
  }

  // Check if the property file contains [SYCL/device requirements] property set
  // and if the required properties are supported by the target
  for (int i = 0; i < Table.getNumRows(); i++) {
    StringRef PropFile = Table[i].getCell("Properties");
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
        MemoryBuffer::getFileAsStream(PropFile);
    if (std::error_code EC = FileOrErr.getError())
      error("Can't read the property file " + PropFile);

    StringRef Properties = FileOrErr->get()->getBuffer();
    if (Properties.find("[SYCL/device requirements]") == StringRef::npos) {
      // copy the line to the output
      continue;
    }

    line_iterator LI(*FileOrErr->get());
    while (*LI != "[SYCL/device requirements]")
      LI++;

    LI++;
    while ((*LI).starts_with("aspects") ||
           (*LI).starts_with("reqd_sub_group_size") ||
           (*LI).starts_with("fixed_target")) {
      if (*LI == "aspects") {
        int size = StringRef("aspects|").size();
        const uint32_t *Array =
            reinterpret_cast<const uint32_t *>((*LI).substr(size).data());
        std::cout << Array[0] << std::endl;
        // std::cout << Array[1] << std::endl;
        // std::cout << Array[2] << std::endl;
        // std::cout << Array[3] << std::endl;
        if (false /*!AspectsSupported()*/)
          continue;
      } else if (*LI == "reqd_sub_group_size") {
        if (false /*!ReqdSubGroupSizeSupported()*/)
          continue;
      } else if (*LI == "fixed_target") {
        if (false /*!FixedTarget()*/)
          continue;
        // Read the byte array and check if the aspects are supported by the
        // target
      }
    }
  }

  // Get columns names
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileAsStream(InputFilename);
  StringRef FirstLine = *line_iterator(*FileOrErr->get());
  SmallVector<StringRef> ColumnNames;
  // The first line of the table looks like: [...|...|...]
  // So we need to get a substring without braces
  FirstLine.substr(1, FirstLine.size() - 2).split(ColumnNames, "|");

  Expected<util::SimpleTable::UPtrTy> NewTableOrError =
      util::SimpleTable::create(ColumnNames);
  if (!NewTableOrError)
    error("Failed to create a new table");
  util::SimpleTable NewTable = *NewTableOrError->get();

  // Write the result table to the output
  std::error_code EC;
  raw_fd_ostream OS{OutputFileName, EC, sys::fs::OpenFlags::OF_None};
  if (EC)
    error("Can't open the output file " + OutputFileName);
  NewTable.write(OS, /*WriteTitles=*/true);
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "sycl-aspect-filter\n");

  if (InputFilename.empty())
    error("Input file not provided.");
  if (!sys::fs::exists(InputFilename))
    error("Input file \'" + InputFilename + "\' not found.");

  if (Target.empty())
    error("Target not provided.");
  // Need to check if the target is valid (when the device config file is ready)

  if (DeviceConfigFile.empty())
    error("Path to the device configuration file not provided.");
  if (!sys::fs::exists(DeviceConfigFile))
    error("Device configuration file \'" + DeviceConfigFile + "\' not found.");

  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename, /*IsText=*/true);
  if (std::error_code EC = FileOrErr.getError())
    error("Could not open input file: " + EC.message());

  Expected<util::SimpleTable::UPtrTy> Table =
      util::SimpleTable::read(InputFilename);
  if (!Table)
    error("Can't read the input table");
  filterTable(*Table->get());

  return 0;
}