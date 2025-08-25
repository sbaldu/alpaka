/* Copyright 2025 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/algorithm/Traits.hpp"
#include "alpaka/exec/UniformElements.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <cstdint>

namespace alpaka
{

    template<typename TApi>
    class DevUniformCudaHipRt;

    template<typename TApi, typename TDim, typename Idx>
    class AccGpuUniformCudaHipRt;

    namespace detail
    {

        template<typename TAcc, typename TElem, typename TIdx, typename TPitchBytes>
        ALPAKA_FN_ACC void IotaDeviceFN(TAcc const& acc, TElem* ptr, TIdx idx, TElem init, TPitchBytes pitchBytes)
        {
            std::uintptr_t offsetBytes = static_cast<std::uintptr_t>((pitchBytes * idx).sum());
            TElem* elem = reinterpret_cast<TElem*>(
                __builtin_assume_aligned(reinterpret_cast<std::uint8_t*>(ptr) + offsetBytes, alignof(TElem)));
            auto offset = static_cast<std::uintptr_t>(elem - ptr) / sizeof(TElem);
            *elem = init + static_cast<TElem>(offset);
        }

        template<typename TElem, typename TExtent, typename TPitchBytes>
        struct IotaKernelND
        {
        public:
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                TElem* ptr,
                TElem init,
                TExtent extent,
                TPitchBytes pitchBytes) const
            {
                for(auto const& idx : alpaka::uniformElementsND(acc, extent))
                {
                    IotaDeviceFN(acc, ptr, idx, init, pitchBytes);
                }
            }
        };

        template<typename TElem>
        struct IotaKernel0D
        {
            ALPAKA_FN_ACC void operator()(TElem* ptr, TElem init) const
            {
                *ptr = init;
            }
        };
    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TApi>
        struct CreateTaskIota<TDim, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewFwd>
            ALPAKA_FN_HOST static auto createTaskIota(
                TViewFwd&& view,
                alpaka::Elem<std::remove_reference_t<TViewFwd>> const& init,
                TExtent const& extent)
            {
                using View = std::remove_reference_t<TViewFwd>;
                using Idx = alpaka::Idx<View>;
                using Acc = AccGpuUniformCudaHipRt<TApi, TDim, Idx>;
                using Elem = alpaka::Elem<View>;
                using Vec = alpaka::Vec<TDim, Idx>;
                using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;

                if constexpr(TDim::value == 0)
                {
                    WorkDiv grid{Vec{}, Vec{}, Vec{}};
                    return alpaka::createTaskKernel<Acc>(grid, alpaka::detail::IotaKernel0D<Elem>());
                }
                else
                {
                    Vec const elements = Vec::ones();
                    Vec threads = Vec::ones();
                    threads.x() = 1024;
                    Vec const blocks = Vec::ones();
                    WorkDiv grid = WorkDiv(blocks, threads, elements);
                    return alpaka::createTaskKernel<Acc>(
                        grid,
                        alpaka::detail::IotaKernelND<Elem, TExtent, decltype(getPitchesInBytes(view))>{},
                        std::data(view),
                        init,
                        extent,
                        getPitchesInBytes(view));
                }
            }
        };
    } // namespace trait

} // namespace alpaka
