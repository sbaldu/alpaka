
#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/elem/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/vec/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

namespace alpaka
{

    namespace trait
    {
        template<typename TDim, typename TDev, typename TSfinae = void>
        struct CreateTaskIota;
    } // namespace trait

    template<typename TViewFwd, typename TElem, typename TExtent>
    ALPAKA_FN_HOST auto createTaskIota(TViewFwd&& view, TElem const& init, TExtent const& extent)
    {
        using TView = std::remove_reference_t<TViewFwd>;
        static_assert(!std::is_const_v<TView>, "The view must not be const!");
        static_assert(
            Dim<TView>::value == Dim<TExtent>::value,
            "The view and the extent are required to have the same dimensionality!");
        static_assert(
            meta::IsIntegralSuperset<Idx<TView>, Idx<TExtent>>::value,
            "The view and the extent must have compatible index types!");

        return trait::CreateTaskIota<Dim<TView>, Dev<TView>>::createTaskIota(
            std::forward<TViewFwd>(view),
            init,
            extent);
    }

    template<typename TQueue, typename TViewFwd, typename TElem, typename TExtent>
    ALPAKA_FN_HOST auto iota(TQueue& queue, TViewFwd&& view, TElem const& init, TExtent const& extent) -> void
    {
        enqueue(queue, createTaskIota(std::forward<TViewFwd>(view), init, extent));
    }

    template<typename TQueue, typename TViewFwd, typename TElem>
    ALPAKA_FN_HOST auto iota(TQueue& queue, TViewFwd&& view, TElem const& init) -> void
    {
        enqueue(queue, createTaskIota(std::forward<TViewFwd>(view), init, getExtents(view)));
    }

    template<typename TQueue, typename TViewFwd, typename TElem>
    ALPAKA_FN_HOST auto iota(TQueue& queue, TViewFwd&& begin, TViewFwd&& end, TElem const& init) -> void
    {
        auto extent = std::distance(begin, end);
        enqueue(queue, createTaskIota(std::forward<TViewFwd>(begin), init, extent));
    }

} // namespace alpaka
