# The tokenizer component (spec 3 §3.1): one Rust crate built by cargo into a
# static archive. Same optionality shape as cmake/prefill.cmake - AUTO turns
# it on when cargo is usable, OFF leaves the decode build byte-identical.
set(B70_TOKENIZER "AUTO" CACHE STRING "Rust tokenizer component: AUTO, ON or OFF")
set_property(CACHE B70_TOKENIZER PROPERTY STRINGS AUTO ON OFF)
set(B70_TOKENIZER_ENABLED OFF)

if(NOT B70_TOKENIZER STREQUAL "OFF")
  find_program(B70_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" "$ENV{CARGO_HOME}/bin")
  if(B70_CARGO)
    execute_process(COMMAND "${B70_CARGO}" --version
                    RESULT_VARIABLE _cargo_rc
                    OUTPUT_VARIABLE _cargo_v
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(_cargo_rc 1)
    set(_cargo_v "cargo not found (looked in ~/.cargo/bin and PATH)")
  endif()
  if(_cargo_rc EQUAL 0)
    set(B70_TOKENIZER_ENABLED ON)
    message(STATUS "b70: tokenizer component ON  -- ${_cargo_v}")
  elseif(B70_TOKENIZER STREQUAL "ON")
    message(FATAL_ERROR "B70_TOKENIZER=ON but cargo is unusable: ${_cargo_v}")
  else()
    message(STATUS "b70: tokenizer component OFF -- ${_cargo_v}")
  endif()
endif()

if(B70_TOKENIZER_ENABLED)
  set(_rs_dir "${CMAKE_SOURCE_DIR}/src/tokenizer/rs")
  set(_rs_target_dir "${CMAKE_BINARY_DIR}/tokenizer-rs")
  set(B70_TOK_RS_LIB "${_rs_target_dir}/release/libb70_tok.a")
  file(GLOB_RECURSE _rs_sources "${_rs_dir}/src/*.rs")
  add_custom_command(
    OUTPUT "${B70_TOK_RS_LIB}"
    COMMAND "${B70_CARGO}" build --release --quiet
            --manifest-path "${_rs_dir}/Cargo.toml" --target-dir "${_rs_target_dir}"
    DEPENDS ${_rs_sources} "${_rs_dir}/Cargo.toml" "${_rs_dir}/Cargo.lock"
    COMMENT "cargo build --release (tokenizers crate)"
    VERBATIM)
  add_custom_target(b70_tok_rs_build DEPENDS "${B70_TOK_RS_LIB}")
  add_library(b70_tok_rs STATIC IMPORTED GLOBAL)
  set_target_properties(b70_tok_rs PROPERTIES IMPORTED_LOCATION "${B70_TOK_RS_LIB}")
  add_dependencies(b70_tok_rs b70_tok_rs_build)
  # What a Rust staticlib needs from the C runtime on Linux (rustc --print
  # native-static-libs says the same; verified in Task 3 Step 3).
  target_link_libraries(b70_tok_rs INTERFACE pthread dl m)
endif()
