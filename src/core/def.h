#ifndef PHARE_CORE_DEF_H
#define PHARE_CORE_DEF_H

#include <stdexcept>

#if defined(PHARE_WITH_GPU)

#define _PHARE_GPU_FN_DEV_ __device__
#define _PHARE_GPU_FN_HST_ __host__
#define _PHARE_FN_SIG_ _PHARE_GPU_FN_HST_ _PHARE_GPU_FN_DEV_


namespace PHARE
{
inline void throw_runtime_error(char* err) _PHARE_GPU_FN_DEV_
{
    // gpu cannot throw
}

inline void throw_runtime_error(char* err) _PHARE_GPU_FN_HST_
{
    throw std::runtime_error(err);
}
} // namespace PHARE

#else

#define _PHARE_GPU_FN_DEV_
#define _PHARE_GPU_FN_HST_
#define _PHARE_FN_SIG_

namespace PHARE
{
inline void throw_runtime_error(char* err)
{
    throw std::runtime_error(err);
}
} // namespace PHARE


#endif


#endif /*PHARE_CORE_DEF_H*/
