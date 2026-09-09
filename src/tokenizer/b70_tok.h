// The C ABI of the Rust crate. Allocation rule: every out-pointer is
// Rust-allocated and released ONLY by b70_tok_free_buf. Every function returns
// 0 on success, <0 on error; b70_tok_last_error() describes the most recent
// error on this thread.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct b70_tok b70_tok;

int b70_tok_new(const char* tokenizer_json_path, b70_tok** out);
void b70_tok_free(b70_tok* t);

// Vocab INCLUDING added tokens (248,077 for the checkpoint - docs/11).
uint32_t b70_tok_vocab_size(const b70_tok* t);

// text: UTF-8, `len` bytes (no NUL needed). add_special=0 is what the golden
// prompts and the chat path use (the template already wrote <|im_start|>...).
int b70_tok_encode(const b70_tok* t, const char* text, size_t len,
                   int add_special, uint32_t** out_ids, size_t* out_n);
int b70_tok_decode(const b70_tok* t, const uint32_t* ids, size_t n,
                   int skip_special, char** out_utf8, size_t* out_len);

// The piece for one id (e.g. "Ġthe" in byte-level form - for debugging only).
int b70_tok_id_to_token(const b70_tok* t, uint32_t id, char** out,
                        size_t* out_len);
// -1 if unknown.
int64_t b70_tok_token_to_id(const b70_tok* t, const char* token, size_t len);

void b70_tok_free_buf(void* p, size_t len_or_n, int is_ids);
const char* b70_tok_last_error(void);

#ifdef __cplusplus
}
#endif
