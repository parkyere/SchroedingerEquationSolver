// Single VMA configuration point for every ses_vk TU (GMFs + vma_impl.cpp).
// Plain textual header, NOT a module. Include <volk.h> BEFORE this header;
// vma_impl.cpp additionally defines VMA_IMPLEMENTATION first.
#pragma once
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
