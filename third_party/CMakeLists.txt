# Vendored single headers (third_party/VERSIONS is the pin record). SYSTEM so
# -Werror applies to our code, not theirs.
add_library(b70_third_party_json INTERFACE)
target_include_directories(b70_third_party_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/json)

add_library(b70_third_party_minja INTERFACE)
target_include_directories(b70_third_party_minja SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/minja)
target_link_libraries(b70_third_party_minja INTERFACE b70_third_party_json)
