// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"            // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd {
namespace Solutions {

/*!
 * \brief Periodic GrMhd solution in Minkowski spacetime.
 *
 * An analytic solution to the 3-D GrMhd system. The user specifies the mean
 * flow velocity of the fluid, the wavevector of the density profile, and the
 * amplitude \f$A\f$ of the density profile. The magnetic field is taken to be
 * zero everywhere. In Cartesian coordinates \f$(x, y, z)\f$, and using
 * dimensionless units, the primitive quantities at a given time \f$t\f$ are
 * then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= 1 + A \sin(\vec{k}\cdot(\vec{x} - \vec{v}t)) \\
 * \vec{v}(\vec{x},t) &= [v_x, v_y, v_z]^{T},\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho}\\
 * \vec{B}(\vec{x},t) &= [0, 0, 0]^{T}
 * \f}
 */
class SmoothFlow {
 public:
  /// The mean flow velocity.
  struct MeanVelocity {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The mean flow velocity."};
  };

  /// The wave vector of the profile.
  struct WaveVector {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The wave vector of the profile."};
  };

  /// The constant pressure throughout the fluid.
  struct Pressure {
    using type = double;
    static constexpr OptionString help = {
        "The constant pressure throughout the fluid."};
    static type lower_bound() { return 0.0; }
  };

  /// The adiabatic exponent for the polytropic fluid.
  struct AdiabaticExponent {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic exponent for the polytropic fluid."};
    static type lower_bound() { return 1.0; }
  };

  /// The perturbation amplitude of the rest mass density of the fluid.
  struct PerturbationSize {
    using type = double;
    static constexpr OptionString help = {
        "The perturbation size of the rest mass density."};
    static type lower_bound() { return -1.0; }
    static type upper_bound() { return 1.0; }
  };

  using options = tmpl::list<MeanVelocity, WaveVector, Pressure,
                             AdiabaticExponent, PerturbationSize>;
  static constexpr OptionString help = {
      "Periodic smooth flow in Minkowski spacetime with zero magnetic field."};

  SmoothFlow() = default;
  SmoothFlow(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow& operator=(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow(SmoothFlow&& /*rhs*/) noexcept = default;
  SmoothFlow& operator=(SmoothFlow&& /*rhs*/) noexcept = default;
  ~SmoothFlow() = default;

  SmoothFlow(MeanVelocity::type mean_velocity, WaveVector::type wavevector,
             Pressure::type pressure,
             AdiabaticExponent::type adiabatic_exponent,
             PerturbationSize::type perturbation_size) noexcept;

  explicit SmoothFlow(CkMigrateMessage* /*unused*/) noexcept {}

  template <typename DataType>
  using variables_t =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  using dt_variables_t = db::wrap_tags_in<Tags::dt, variables_t<DataType>>;

  /// Retrieve the primitive variables at time `t` and spatial coordinates `x`
  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_t<DataType>> variables(
      const tnsr::I<DataType, 3>& x, double t,
      variables_t<DataType> /*meta*/) const noexcept;

  /// Retrieve the time derivative of the primitive variables at time `t` and
  /// spatial coordinates `x`
  template <typename DataType>
  tuples::tagged_tuple_from_typelist<dt_variables_t<DataType>> variables(
      const tnsr::I<DataType, 3>& x, double t,
      dt_variables_t<DataType> /*meta*/) const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT
  MeanVelocity::type mean_velocity() const noexcept { return mean_velocity_; }
  WaveVector::type wavevector() const noexcept { return wavevector_; }
  Pressure::type pressure() const noexcept { return pressure_; }
  AdiabaticExponent::type adiabatic_exponent() const noexcept {
    return adiabatic_exponent_;
  }
  PerturbationSize::type perturbation_size() const noexcept {
    return perturbation_size_;
  }
  double k_dot_v() const noexcept { return k_dot_v_; }

 private:
  // Computes the phase.
  template <typename DataType>
  DataType k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x, double t) const
      noexcept;
  MeanVelocity::type mean_velocity_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  WaveVector::type wavevector_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  Pressure::type pressure_ = std::numeric_limits<double>::signaling_NaN();
  AdiabaticExponent::type adiabatic_exponent_ =
      std::numeric_limits<double>::signaling_NaN();
  PerturbationSize::type perturbation_size_ =
      std::numeric_limits<double>::signaling_NaN();
  // The angular frequency.
  double k_dot_v_ = std::numeric_limits<double>::signaling_NaN();
};

inline bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept {
  return lhs.mean_velocity() == rhs.mean_velocity() and
         lhs.wavevector() == rhs.wavevector() and
         lhs.pressure() == rhs.pressure() and
         lhs.adiabatic_exponent() == rhs.adiabatic_exponent() and
         lhs.perturbation_size() == rhs.perturbation_size() and
         lhs.k_dot_v() == rhs.k_dot_v();
}

inline bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace grmhd
