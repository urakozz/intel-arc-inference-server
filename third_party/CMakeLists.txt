# Vendored single headers (third_party/VERSIONS is the pin record). SYSTEM so
# -Werror applies to our code, not theirs.
add_library(b70_third_party_json INTERFACE)
target_include_directories(b70_third_party_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/json)

add_library(b70_third_party_minja INTERFACE)
target_include_directories(b70_third_party_minja SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/minja)
target_link_libraries(b70_third_party_minja INTERFACE b70_third_party_json)
# SYSTEM does not cover middle-end warnings on code inlined into our functions: g++ 15 at
# -O2 reports a false -Warray-bounds inside nlohmann::ordered_map's tuple compare, inlined
# from minja's builtins (chat_template.cc, minja_ext_test.cc; clang does not warn).
target_compile_options(b70_third_party_minja INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>)

add_library(b70_third_party_httplib INTERFACE)
target_include_directories(b70_third_party_httplib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/httplib)
target_link_libraries(b70_third_party_httplib INTERFACE pthread)
