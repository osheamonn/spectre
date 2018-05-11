// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_ricci(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &gr::ricci_tensor<Dim, Frame::Inertial, TypeOfIndex, DataType>, "GrTests",
      "ricci_tensor", {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Ricci.",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  const double d(std::numeric_limits<double>::signaling_NaN());
  const DataVector dv(5);
  test_ricci<1, IndexType::Spatial>(dv);
  test_ricci<2, IndexType::Spatial>(dv);
  test_ricci<3, IndexType::Spatial>(dv);
  test_ricci<1, IndexType::Spacetime>(d);
  test_ricci<2, IndexType::Spacetime>(d);
  test_ricci<3, IndexType::Spacetime>(d);
}
