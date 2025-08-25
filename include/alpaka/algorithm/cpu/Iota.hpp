/* Copyright 2025 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/algorithm/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"

namespace alpaka
{

    struct DevCpu;

    namespace detail
    {

        template<typename TDim, typename TView, typename TExtent>
        struct TaskIotaCpu
        {
        public:
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            static_assert(std::is_trivially_copyable_v<Elem>, "Iota only supports trivially copyable view types");

            template<typename TViewFwd>
            TaskIotaCpu(TViewFwd&& view, Elem const& init, TExtent const& extent)
                : m_init{init}
                , m_extent{getExtents(extent)}
                , m_dstExtent{getExtents(view)}
                , m_dstPitchBytes{getPitchesInBytes(view)}
                , m_dstMemNative{getPtrNative(view)}
            {
                // TODO: add asserts on the dimentions
            }

            ALPAKA_FN_HOST auto operator()() -> void
            {
                if(static_cast<std::size_t>(m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        m_extent,
                        [&, init = m_init](Vec<TDim, ExtentSize> const& idx) mutable -> void
                        {
                            auto offsetBytes = static_cast<std::uintptr_t>((idx * m_dstPitchBytes).sum());
                            auto* elem = reinterpret_cast<Elem*>(__builtin_assume_aligned(
                                reinterpret_cast<std::uint8_t*>(m_dstMemNative) + offsetBytes,
                                alignof(Elem)));

                            *elem = init;
                            init += static_cast<Elem>(1);
                        });
                }
            }

        private:
            Elem m_init;
            Vec<TDim, ExtentSize> const m_extent;
            Vec<TDim, DstSize> const m_dstExtent;
            Vec<TDim, DstSize> const m_dstPitchBytes;
            Elem* const m_dstMemNative;
        };

        template<typename TView, typename TExtent>
        struct TaskIotaCpu<DimInt<0u>, TView, TExtent>
        {
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskIotaCpu(TViewFwd&& view, Elem const& init, [[maybe_unused]] TExtent const& extent)
                : m_init{init}
                , m_dstMemNative{getPtrNative(view)}
            {
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

        private:
            Elem const m_init;
            Elem* const m_dstMemNative;
        };

    } // namespace detail

    namespace trait
    {
        template<typename TDim>
        struct CreateTaskIota<TDim, DevCpu>
        {
            template<typename TExtent, typename TViewFwd>
            ALPAKA_FN_HOST static auto createTaskIota(
                TViewFwd&& view,
                alpaka::Elem<std::remove_reference_t<TViewFwd>> const& init,
                TExtent const& extent)
            {
                using TView = std::remove_reference_t<TViewFwd>;
                using Elem = alpaka::Elem<TView>;
                static_assert(std::is_trivially_copyable_v<Elem>, "Iota only supports trivially copyable view types");

                return alpaka::detail::TaskIotaCpu<TDim, TView, TExtent>{std::forward<TViewFwd>(view), init, extent};
            }
        };
    } // namespace trait

} // namespace alpaka
