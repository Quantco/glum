#pragma once

#ifndef _WIN32
  #define JEMALLOC_NO_DEMANGLE
  #if __APPLE__
    #define JEMALLOC_NO_RENAME
  #endif
  #include <jemalloc/jemalloc.h>
#endif

std::size_t round_to_align(std::size_t size, std::size_t alignment) {
  std::size_t remainder = size % alignment;

  if (remainder == 0) {
    return size;
  } else {
    return size + alignment - remainder;
  }
}

