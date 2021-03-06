// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <sstream>
#include <string>

/*!
 * \ingroup UtilitiesGroup
 * \brief Make a string by streaming into object.
 *
 * \snippet Test_MakeString.cpp make_string
 */
class MakeString {
 public:
  MakeString() = default;
  MakeString(const MakeString&) = delete;
  MakeString& operator=(const MakeString&) = delete;
  MakeString(MakeString&&) = default;
  MakeString& operator=(MakeString&&) = delete;
  ~MakeString() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::string() const noexcept { return stream_.str(); }

  template <class T>
  friend MakeString operator<<(MakeString&& ms, const T& t) noexcept {
    // clang-tidy: can get unintentional pointer casts
    ms.stream_ << t;  // NOLINT
    return std::move(ms);
  }

  template <class T>
  friend MakeString& operator<<(MakeString& ms, const T& t) noexcept {
    // clang-tidy: can get unintentional pointer casts
    ms.stream_ << t;  // NOLINT
    return ms;
  }

 private:
  std::stringstream stream_{};
};
