// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
namespace Actions {
/*!
 * \brief Initializes the DataBox on the observer parallel component
 */
struct Initialize {
  using simple_tags =
      db::AddSimpleTags<Tags::NumberOfEvents, Tags::ReductionArrayComponentIds,
                        Tags::VolumeArrayComponentIds, Tags::TensorData>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(db::create<simple_tags>(
        db::item_type<Tags::NumberOfEvents>{},
        db::item_type<Tags::ReductionArrayComponentIds>{},
        db::item_type<Tags::VolumeArrayComponentIds>{},
        db::item_type<Tags::TensorData>{}));
  }
};

/*!
 * \brief Initializes the DataBox of the observer parallel component that writes
 * to disk.
 */
struct InitializeWriter {
  using simple_tags =
      db::AddSimpleTags<Tags::TensorData, Tags::VolumeObserversContributed,
                        Tags::ReductionFileLock, Tags::VolumeFileLock>;
  using compute_tags = db::AddComputeTags<>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(db::create<simple_tags>(
        db::item_type<Tags::TensorData>{},
        db::item_type<Tags::VolumeObserversContributed>{},
        Parallel::create_lock(), Parallel::create_lock()));
  }
};
}  // namespace Actions
}  // namespace observers
