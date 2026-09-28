add_library(b70_server STATIC openai.cc prefix_cache.cc server.cc spec_accept.cc toolcall.cc)
target_include_directories(b70_server PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_server PUBLIC b70_third_party_json b70_third_party_httplib)
