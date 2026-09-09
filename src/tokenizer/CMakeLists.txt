if(NOT B70_TOKENIZER_ENABLED)
  return()
endif()

# b70_tokenizer (the C++ wrapper) is added in Task 4, with tokenizer.cc - its
# only source. Declaring it here would fail configure: CMake resolves a
# target's sources at generate time, not at build time.


add_library(b70_tokenizer STATIC tokenizer.cc chat_template.cc streamer.cc)
target_include_directories(b70_tokenizer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_tokenizer PUBLIC b70_tok_rs b70_third_party_minja)
target_compile_definitions(b70_tokenizer PRIVATE
  B70_CHAT_TEMPLATE_FALLBACK_PATH="${CMAKE_CURRENT_SOURCE_DIR}/chat_template_fallback.jinja")
