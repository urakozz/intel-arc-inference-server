use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::slice;

use tokenizers::Tokenizer;

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

fn set_err(e: impl std::fmt::Display) -> i32 {
    LAST.with(|last| *last.borrow_mut() = CString::new(e.to_string()).unwrap_or_default());
    -1
}

#[allow(non_camel_case_types)]
pub struct b70_tok(Tokenizer);

#[no_mangle]
pub extern "C" fn b70_tok_new(path: *const c_char, out: *mut *mut b70_tok) -> i32 {
    if path.is_null() || out.is_null() {
        return set_err("null argument");
    }
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
    match Tokenizer::from_file(&path) {
        Ok(tokenizer) => {
            unsafe {
                *out = Box::into_raw(Box::new(b70_tok(tokenizer)));
            }
            0
        }
        Err(error) => set_err(format!("{path}: {error}")),
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_free(tokenizer: *mut b70_tok) {
    if !tokenizer.is_null() {
        unsafe {
            drop(Box::from_raw(tokenizer));
        }
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_vocab_size(tokenizer: *const b70_tok) -> u32 {
    unsafe { &*tokenizer }.0.get_vocab_size(true) as u32
}

#[no_mangle]
pub extern "C" fn b70_tok_encode(
    tokenizer: *const b70_tok,
    text: *const c_char,
    len: usize,
    add_special: i32,
    out_ids: *mut *mut u32,
    out_n: *mut usize,
) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(text as *const u8, len) };
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => return set_err(error),
    };
    match unsafe { &*tokenizer }.0.encode(text, add_special != 0) {
        Ok(encoding) => {
            let mut ids = encoding.get_ids().to_vec();
            ids.shrink_to_fit();
            unsafe {
                *out_n = ids.len();
                *out_ids = ids.as_mut_ptr();
            }
            std::mem::forget(ids);
            0
        }
        Err(error) => set_err(error),
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_decode(
    tokenizer: *const b70_tok,
    ids: *const u32,
    n: usize,
    skip_special: i32,
    out: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    let ids = unsafe { slice::from_raw_parts(ids, n) };
    match unsafe { &*tokenizer }.0.decode(ids, skip_special != 0) {
        Ok(text) => {
            let mut bytes = text.into_bytes();
            bytes.shrink_to_fit();
            unsafe {
                *out_len = bytes.len();
                *out = bytes.as_mut_ptr() as *mut c_char;
            }
            std::mem::forget(bytes);
            0
        }
        Err(error) => set_err(error),
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_id_to_token(
    tokenizer: *const b70_tok,
    id: u32,
    out: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    match unsafe { &*tokenizer }.0.id_to_token(id) {
        Some(token) => {
            let mut bytes = token.into_bytes();
            bytes.shrink_to_fit();
            unsafe {
                *out_len = bytes.len();
                *out = bytes.as_mut_ptr() as *mut c_char;
            }
            std::mem::forget(bytes);
            0
        }
        None => set_err(format!("id {id} has no token")),
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_token_to_id(
    tokenizer: *const b70_tok,
    token: *const c_char,
    len: usize,
) -> i64 {
    let bytes = unsafe { slice::from_raw_parts(token as *const u8, len) };
    match std::str::from_utf8(bytes)
        .ok()
        .and_then(|token| unsafe { &*tokenizer }.0.token_to_id(token))
    {
        Some(id) => id as i64,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_free_buf(p: *mut std::ffi::c_void, len: usize, is_ids: i32) {
    if p.is_null() {
        return;
    }
    unsafe {
        if is_ids != 0 {
            drop(Vec::from_raw_parts(p as *mut u32, len, len));
        } else {
            drop(Vec::from_raw_parts(p as *mut u8, len, len));
        }
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_last_error() -> *const c_char {
    LAST.with(|last| last.borrow().as_ptr())
}
