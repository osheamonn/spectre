// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp" // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp" // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
class DataVector;
namespace intrp {
namespace Tags {
struct NumberOfElements;
} // namespace Tags
} // namespace intrp
/// \endcond

namespace {

template <typename Metavariables, size_t VolumeDim>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator<
          VolumeDim>::template return_tag_list<Metavariables>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = Time;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables, 3>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.RegisterElement",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars, 3>>;
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars, 3>>{});
  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.simple_action<mock_interpolator<metavars, 3>,
                       ::intrp::Actions::InitializeInterpolator<3>>(0);

  const auto& box =
      runner.template algorithms<mock_interpolator<metavars, 3>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars, 3>::initial_databox>();

  CHECK(db::get<::intrp::Tags::NumberOfElements>(box) == 0);

  runner.simple_action<mock_interpolator<metavars, 3>,
                       ::intrp::Actions::RegisterElement>(0);

  CHECK(db::get<::intrp::Tags::NumberOfElements>(box) == 1);

  runner.simple_action<mock_interpolator<metavars, 3>,
                       ::intrp::Actions::RegisterElement>(0);

  CHECK(db::get<::intrp::Tags::NumberOfElements>(box) == 2);
}

}  // namespace
