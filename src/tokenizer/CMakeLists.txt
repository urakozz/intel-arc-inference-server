if(NOT B70_TOKENIZER_ENABLED)
  return()
endif()

# b70_tokenizer (the C++ wrapper) is added in Task 4, with tokenizer.cc - its
# only source. Declaring it here would fail configure: CMake resolves a
# target's sources at generate time, not at build time.
