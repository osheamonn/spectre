# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Burgers
  BurgersSolutions
  DiscontinuousGalerkin
  DomainCreators
  Informer
  LinearOperators
  Time
  Utilities
  )

add_spectre_executable(
  EvolveBurgers
  EvolveBurgers
  Evolution/Executables/Burgers
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
