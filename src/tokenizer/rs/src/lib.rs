use std::cell::RefCell;
use std::collections::HashSet;
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

// The tokenizer, the added tokens matched on the raw text with no strip or word rule (the text splits at each before
// any other rule runs: b70_tok_encode_ends) and the longest added token's bytes.
#[allow(non_camel_case_types)]
pub struct b70_tok(Tokenizer, HashSet<u32>, u32);

#[no_mangle]
pub extern "C" fn b70_tok_new(path: *const c_char, out: *mut *mut b70_tok) -> i32 {
    if path.is_null() || out.is_null() {
        return set_err("null argument");
    }
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
    match Tokenizer::from_file(&path) {
        Ok(mut tokenizer) => {
            // The engine tokenises whole prompts: a file's `truncation` / `padding` blocks
            // never apply. A quantised repo has shipped truncation at 2048 ids (a calibration
            // setting), which silently cut every longer prompt's tail (no_truncation_test).
            if let Err(error) = tokenizer.with_truncation(None) {
                return set_err(format!("{path}: {error}"));
            }
            tokenizer.with_padding(None);
            let added = tokenizer.get_added_tokens_decoder();
            let mut splits: HashSet<u32> = added
                .iter()
                .filter(|(_, t)| !t.normalized && !t.lstrip && !t.rstrip && !t.single_word)
                .map(|(id, _)| *id)
                .collect();
            // A pre-tokenizer that treats the text's start apart from its other pieces (a prefix space on the first
            // piece only) makes a tail's encode differ from the same text inside a longer one: such a tokenizer gets
            // no split points.
            if let Some(token) = splits.iter().next().and_then(|id| added.get(id)) {
                let head = format!("a{}", token.content);
                if ["b c", " b", "\n b", "B"].iter().any(|tail| {
                    let whole = tokenizer.encode(format!("{head}{tail}"), false);
                    let (h, t) = (tokenizer.encode(head.as_str(), false), tokenizer.encode(*tail, false));
                    match (whole, h, t) {
                        (Ok(w), Ok(h), Ok(t)) => w.get_ids() != [h.get_ids(), t.get_ids()].concat().as_slice(),
                        _ => true,
                    }
                }) {
                    splits.clear();
                }
            }
            let longest = added.values().map(|t| t.content.len() as u32).max().unwrap_or(0);
            unsafe {
                *out = Box::into_raw(Box::new(b70_tok(tokenizer, splits, longest)));
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
pub extern "C" fn b70_tok_encode_ends(
    tokenizer: *const b70_tok,
    text: *const c_char,
    len: usize,
    out_ids: *mut *mut u32,
    out_ends: *mut *mut u32,
    out_n: *mut usize,
) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(text as *const u8, len) };
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => return set_err(error),
    };
    if len > u32::MAX as usize {
        return set_err("text over 4 GiB");
    }
    let t = unsafe { &*tokenizer };
    match t.0.encode(text, false) {
        Ok(encoding) => {
            let mut ids = encoding.get_ids().to_vec();
            let mut ends: Vec<u32> = ids
                .iter()
                .zip(encoding.get_offsets())
                .map(|(id, &(_, end))| if t.1.contains(id) { end as u32 } else { 0 })
                .collect();
            ids.shrink_to_fit();
            ends.shrink_to_fit();
            unsafe {
                *out_n = ids.len();
                *out_ids = ids.as_mut_ptr();
                *out_ends = ends.as_mut_ptr();
            }
            std::mem::forget(ids);
            std::mem::forget(ends);
            0
        }
        Err(error) => set_err(error),
    }
}

#[no_mangle]
pub extern "C" fn b70_tok_longest_added(tokenizer: *const b70_tok) -> u32 {
    unsafe { &*tokenizer }.2
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
