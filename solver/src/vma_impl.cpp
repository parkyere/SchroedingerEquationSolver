// The single VMA implementation TU for the whole ses_vk world. It used to be
// stamped per-binary (main.cpp / vkcheck_main.cpp defining VMA_IMPLEMENTATION
// before the textual vk_device.hpp chain); modules severed that textual path,
// so the implementation lives here, compiled once into ses_solver.
// Configuration lives in ses_vma.h (shared with every ses_vk GMF): VMA rides
// volk's dynamically fetched pointers.
#include <volk.h>

#define VMA_IMPLEMENTATION
#include "ses_vma.h"
