// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Math.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Math", "[Unit][Utilities]") {
  SECTION("Test number_of_digits") {
    CHECK(2 == number_of_digits(10));
    CHECK(1 == number_of_digits(0));
    CHECK(1 == number_of_digits(-1));
    CHECK(1 == number_of_digits(9));
    CHECK(2 == number_of_digits(-99));
  }

  SECTION("Test evaluate_polynomial") {
    const std::vector<double> poly_coeffs{1., 2.5, 0.3, 1.5};
    CHECK_ITERABLE_APPROX(evaluate_polynomial(poly_coeffs, 0.5), 2.5125);
    CHECK_ITERABLE_APPROX(
        evaluate_polynomial(poly_coeffs,
                            DataVector({-0.5, -0.1, 0., 0.8, 1., 12.})),
        DataVector({-0.3625, 0.7515, 1., 3.96, 5.3, 2666.2}));
    const std::vector<DataVector> poly_variable_coeffs{DataVector{1., 0., 2.},
                                                       DataVector{0., 2., 1.}};
    CHECK_ITERABLE_APPROX(
        evaluate_polynomial(poly_variable_coeffs, DataVector({0., 0.5, 1.})),
        DataVector({1., 1., 3.}));
  }
}
