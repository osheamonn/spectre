// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower_first_index(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &raise_or_lower_first_index<
          DataType,
          Tensor_detail::TensorIndexType<Dim, UpOrLo, Frame::Inertial, Index>,
          Tensor_detail::TensorIndexType<Dim, UpLo::Lo, Frame::Inertial,
                                         Index>>,
      "GrTests", "raise_or_lower_first_index", {{{-10., 10.}}}, used_for_size);
}

template <size_t Dim, UpLo UpOrLo, IndexType Index, typename DataType>
void test_raise_or_lower(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &raise_or_lower_index<DataType, Tensor_detail::TensorIndexType<
                                          Dim, UpOrLo, Frame::Inertial, Index>>,
      "numpy", "matmul", {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_trace_last_indices(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &trace_last_indices<Dim, Frame::Inertial, TypeOfIndex, DataType>,
      "GrTests", "trace_last_indices", {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_trace(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &trace<Dim, Frame::Inertial, TypeOfIndex, DataType>, "numpy", "tensordot",
      {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IndexManipulation",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector dv(5);
  const double d = std::numeric_limits<double>::signaling_NaN();

  test_raise_or_lower_first_index<1, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower_first_index<2, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower_first_index<3, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower_first_index<1, UpLo::Up, IndexType::Spatial>(dv);
  test_raise_or_lower_first_index<2, UpLo::Up, IndexType::Spatial>(dv);
  test_raise_or_lower_first_index<3, UpLo::Up, IndexType::Spatial>(dv);

  test_raise_or_lower<1, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower<2, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower<3, UpLo::Lo, IndexType::Spacetime>(d);
  test_raise_or_lower<1, UpLo::Up, IndexType::Spatial>(dv);
  test_raise_or_lower<2, UpLo::Up, IndexType::Spatial>(dv);
  test_raise_or_lower<3, UpLo::Up, IndexType::Spatial>(dv);

  test_trace_last_indices<1, IndexType::Spacetime>(d);
  test_trace_last_indices<2, IndexType::Spacetime>(d);
  test_trace_last_indices<3, IndexType::Spacetime>(d);
  test_trace_last_indices<1, IndexType::Spatial>(dv);
  test_trace_last_indices<2, IndexType::Spatial>(dv);
  test_trace_last_indices<3, IndexType::Spatial>(dv);

  test_trace<1, IndexType::Spacetime>(d);
  test_trace<2, IndexType::Spacetime>(d);
  test_trace<3, IndexType::Spacetime>(d);
  test_trace<1, IndexType::Spatial>(dv);
  test_trace<2, IndexType::Spatial>(dv);
  test_trace<3, IndexType::Spatial>(dv);
}
