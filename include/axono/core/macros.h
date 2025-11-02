#pragma once

// 强制内联宏
#if defined(__GNUC__) || defined(__clang__)
    #define AXONO_FORCE_INLINE inline __attribute__((always_inline))
    #define AXONO_LIKELY(x) __builtin_expect(!!(x), 1)
    #define AXONO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    #define AXONO_FORCE_INLINE __forceinline
    #define AXONO_LIKELY(x) (x)
    #define AXONO_UNLIKELY(x) (x)
#else
    #define AXONO_FORCE_INLINE inline
    #define AXONO_LIKELY(x) (x)
    #define AXONO_UNLIKELY(x) (x)
#endif

// DLL 导出/导入宏
#ifdef AXONO_BUILD_SHARED_LIB
    #ifdef _WIN32
        #define AXONO_EXPORT __declspec(dllexport)
    #else
        #define AXONO_EXPORT __attribute__((visibility("default")))
    #endif
#else
    #define AXONO_EXPORT
#endif

// 禁用拷贝和移动
#define AXONO_DISALLOW_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete

#define AXONO_DISALLOW_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete

#define AXONO_DISALLOW_COPY_AND_MOVE(ClassName) \
    AXONO_DISALLOW_COPY(ClassName); \
    AXONO_DISALLOW_MOVE(ClassName)
