// Checks that a kernel defined in a module built in --offload-arch mode is
// taken from that module without ranking, and that other kernels keep using
// the existing selection.

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace sycl;

namespace {

// An --offload-arch module. Its compile_target does not match the mock device,
// as it describes only one of the sub-modules. Its binary is the kernel name so
// that the test can tell which image reached the backend.
unittest::MockDeviceImage generateIgcaModule(const std::string &KernelName) {
  uint32_t One = 1;
  std::vector<char> IsIgcaModule(sizeof(One));
  std::memcpy(IsIgcaModule.data(), &One, sizeof(One));

  // A byte array property value is preceded by an 8 byte size header.
  std::string CompileTarget = "intel_gpu_pvc";
  std::vector<char> CompileTargetValue(8 + CompileTarget.size());
  std::copy(CompileTarget.begin(), CompileTarget.end(),
            CompileTargetValue.data() + 8);

  unittest::MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IGCA_METADATA,
                 unittest::MockProperty{"isIgcaModule", IsIgcaModule,
                                        SYCL_PROPERTY_TYPE_UINT32});
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 unittest::MockProperty{"compile_target", CompileTargetValue,
                                        SYCL_PROPERTY_TYPE_BYTE_ARRAY});

  std::vector<unsigned char> Bin(KernelName.begin(), KernelName.end());
  Bin.push_back(0);

  return unittest::MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                                   __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                                   "",
                                   "",
                                   std::move(Bin),
                                   unittest::makeEmptyKernels({KernelName}),
                                   std::move(PropSet)};
}

} // namespace

class IgcaKernel;
class JitOnlyKernel;

MOCK_INTEGRATION_HEADER(IgcaKernel)
MOCK_INTEGRATION_HEADER(JitOnlyKernel)

static unittest::MockDeviceImage Imgs[] = {
    generateIgcaModule("IgcaKernel"),
    unittest::generateDefaultImage({"IgcaKernel"}),
    unittest::generateDefaultImage({"JitOnlyKernel"}),
    unittest::generateDefaultImage({"JitOnlyKernel"})};

static unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

namespace {

std::vector<std::string> BinariesGivenToBackend;
// Largest number of images the runtime asked the backend to choose between.
uint32_t MaxImagesOfferedToRanking = 0;

ur_result_t afterProgramCreateWithBinary(void *pParams) {
  auto Params = *static_cast<ur_program_create_with_binary_params_t *>(pParams);
  for (uint32_t I = 0; I < *Params.pnumDevices; ++I)
    BinariesGivenToBackend.push_back(
        reinterpret_cast<const char *>(*Params.pppBinaries[I]));
  return UR_RESULT_SUCCESS;
}

ur_result_t afterDeviceSelectBinary(void *pParams) {
  auto Params = *static_cast<ur_device_select_binary_params_t *>(pParams);
  MaxImagesOfferedToRanking =
      std::max(MaxImagesOfferedToRanking, *Params.pNumBinaries);
  **Params.ppSelectedBinary = 0;
  return UR_RESULT_SUCCESS;
}

class IgcaModuleSelectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    BinariesGivenToBackend.clear();
    MaxImagesOfferedToRanking = 0;
    mock::getCallbacks().set_after_callback("urProgramCreateWithBinary",
                                            &afterProgramCreateWithBinary);
    mock::getCallbacks().set_after_callback("urDeviceSelectBinary",
                                            &afterDeviceSelectBinary);
  }

  unittest::UrMock<> Mock;
};

} // namespace

TEST_F(IgcaModuleSelectionTest, IgcaModuleSelectedWithoutRanking) {
  queue{}.single_task<IgcaKernel>([]() {});

  ASSERT_EQ(BinariesGivenToBackend.size(), 1U);
  EXPECT_EQ(BinariesGivenToBackend.back(), "IgcaKernel");
  EXPECT_LE(MaxImagesOfferedToRanking, 1U);
}

TEST_F(IgcaModuleSelectionTest, KernelBundleFindsIgcaModule) {
  device Dev = platform{}.get_devices()[0];
  EXPECT_TRUE(sycl::is_compatible<IgcaKernel>(Dev));

  kernel_id Id = sycl::get_kernel_id<IgcaKernel>();
  kernel_bundle KB = sycl::get_kernel_bundle<bundle_state::executable>(
      context{Dev}, {Dev}, {Id});
  EXPECT_TRUE(KB.has_kernel(Id));
  ASSERT_EQ(BinariesGivenToBackend.size(), 1U);
  EXPECT_EQ(BinariesGivenToBackend.back(), "IgcaKernel");
}

TEST_F(IgcaModuleSelectionTest, NonIgcaKernelStillRanked) {
  queue{}.single_task<JitOnlyKernel>([]() {});

  EXPECT_GT(MaxImagesOfferedToRanking, 1U);
}
