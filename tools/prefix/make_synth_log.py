#!/usr/bin/env python3
"""An opencode-like request log, synthetic (spec 7 C3's machinery, plan 7c Task 4).

    make_synth_log.py <out dir> [--turns 12] [--repo .]

Writes <out dir>/NNNNNN.json in the shape `b70-serve --log-requests` writes (endpoint,
request body; prompt_ids / out_ids are left empty -- replay_log.py sends the request bodies
and caps each response at `max_tokens_logged`). The session: a long system prompt with
opencode's tool set (bash, read, edit, write, grep, glob, list, todowrite), a user task, then
`--turns` requests, each resending the whole conversation so far: assistant turns with tool
calls (arguments as JSON strings, as opencode sends them), tool results holding real source
files of this repository, assistant text. Two short side requests (title generation, a
summary subagent) are interleaved after turns 2 and 7, diverging from the main session at
position 0, as opencode's are.

The histories are fixed text, not what any model generated: the replay's prompts are the
same with the cache on and off, so the two runs' outputs are comparable.
"""
import argparse
import json
import os

SYSTEM = """You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Refuse to write code or explain code that may be used maliciously. Before you begin work, think about what the code you're editing is supposed to do based on the filenames and directory structure.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it. Your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting. Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks.

# Proactiveness
You are allowed to be proactive, but only when the user asks you to do something. Strike a balance between doing the right thing when asked, including taking actions and follow-up actions, and not surprising the user with actions you take without asking.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns. NEVER assume that a given library is available. When you create a new component, first look at existing components to see how they're written. When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries.

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.
- Implement the solution using all tools available to you
- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.
- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands if they were provided to you to ensure your code is correct.

# Tool usage policy
- When doing file search, prefer to use the Task tool in order to reduce context usage.
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance.

Here is useful information about the environment you are running in:
<env>
  Working directory: /home/user/b70-inference-server
  Is directory a git repo: yes
  Platform: linux
</env>
"""


def tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = [
    tool("bash", "Executes a given bash command in a persistent shell session with optional "
         "timeout, ensuring proper handling and security measures. Before executing the "
         "command, verify the directory exists; always quote file paths that contain spaces.",
         {"command": {"type": "string", "description": "The command to execute"},
          "timeout": {"type": "number", "description": "Optional timeout in milliseconds"},
          "description": {"type": "string", "description": "Clear, concise description of what "
                          "this command does in 5-10 words"}}, ["command", "description"]),
    tool("read", "Reads a file from the local filesystem. The filePath parameter must be an "
         "absolute path. By default, it reads up to 2000 lines starting from the beginning.",
         {"filePath": {"type": "string", "description": "The path to the file to read"},
          "offset": {"type": "number", "description": "The line number to start reading from"},
          "limit": {"type": "number", "description": "The number of lines to read"}},
         ["filePath"]),
    tool("edit", "Performs exact string replacements in files. You must use your read tool at "
         "least once before editing. The edit will FAIL if oldString is not unique.",
         {"filePath": S, "oldString": S, "newString": S,
          "replaceAll": {"type": "boolean", "description": "Replace all occurrences"}},
         ["filePath", "oldString", "newString"]),
    tool("write", "Writes a file to the local filesystem, overwriting it if it exists.",
         {"filePath": S, "content": S}, ["filePath", "content"]),
    tool("grep", "Fast content search tool that works with any codebase size. Searches file "
         "contents using regular expressions and returns matching file paths.",
         {"pattern": S, "path": S, "include": S}, ["pattern"]),
    tool("glob", "Fast file pattern matching tool; returns matching file paths sorted by "
         "modification time.", {"pattern": S, "path": S}, ["pattern"]),
    tool("list", "Lists files and directories in a given path.",
         {"path": S, "ignore": {"type": "array", "items": S}}, []),
    tool("todowrite", "Use this tool to create and manage a structured task list for your "
         "current coding session.",
         {"todos": {"type": "array", "items": {"type": "object", "properties": {
             "content": S, "status": S, "priority": S, "id": S}}}}, ["todos"]),
]


def call(i, name, args):
    return {"id": f"call_{i:04d}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def src(path, lines=160):
        with open(os.path.join(a.repo, path), encoding="utf-8") as f:
            return "".join(f.readlines()[:lines])

    files = ["src/server/server.cc", "src/server/openai.cc", "src/server/toolcall.cc",
             "src/runtime/engine.h", "src/server/deps.h", "src/cli/b70_serve.cc"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "The server drops the connection when a client sends "
             "an empty stop list. Find where stop strings are parsed and handled, fix it, and "
             "add a test."}]
    # (assistant reasoning, tool call) per step; tool results follow each call.
    steps = [
        ("I'll start by listing the server sources.", "list", {"path": "src/server"}),
        ("Let me read the request parser.", "read", {"filePath": "src/server/openai.cc"}),
        ("Now the server loop that applies stop strings.", "read", {"filePath": "src/server/server.cc"}),
        ("Searching for other stop handling.", "grep", {"pattern": "stop", "path": "src/server"}),
        ("Let me look at the tool-call parser too, it holds text back.", "read",
         {"filePath": "src/server/toolcall.cc"}),
        ("The engine interface, to see what step() returns.", "read", {"filePath": "src/server/deps.h"}),
        ("Tracking the work.", "todowrite", {"todos": [
            {"content": "guard empty stop strings", "status": "in_progress", "priority": "high", "id": "1"},
            {"content": "add a protocol test", "status": "pending", "priority": "medium", "id": "2"}]}),
        ("Applying the fix.", "edit", {"filePath": "src/server/server.cc",
         "oldString": "hold = std::max(hold, stop.empty() ? 0U : stop.size() - 1);",
         "newString": "if (stop.empty()) continue;\n    hold = std::max(hold, stop.size() - 1);"}),
        ("Building the tests.", "bash", {"command": "cmake --build build -j44 --target protocol_test",
                                         "description": "Build the protocol test"}),
        ("Running them.", "bash", {"command": "ctest --test-dir build -R protocol_test",
                                   "description": "Run the protocol test"}),
        ("Checking the CLI for a related flag.", "read", {"filePath": "src/cli/b70_serve.cc"}),
        ("The engine header documents the pending-token protocol.", "read",
         {"filePath": "src/runtime/engine.h"}),
    ]
    results = [
        "deps.h\nopenai.cc\nopenai.h\nprefix_cache.cc\nprefix_cache.h\nserver.cc\nserver.h\n"
        "toolcall.cc\ntoolcall.h\nCMakeLists.txt",
        src("src/server/openai.cc"), src("src/server/server.cc"),
        "src/server/server.cc:182:  for (const std::string& stop : r.stop) {\n"
        "src/server/openai.cc:188:    out.stop = string_list(request.at(\"stop\"));\n"
        "src/server/server.h:58:  std::vector<std::string> stop;",
        src("src/server/toolcall.cc"), src("src/server/deps.h", 80), "Todos updated.",
        "Edit applied successfully.",
        "[100%] Built target protocol_test",
        "100% tests passed, 0 tests failed out of 1",
        src("src/cli/b70_serve.cc"), src("src/runtime/engine.h", 120),
    ]
    records = []

    def emit(body, logged_out):
        records.append({"endpoint": "/v1/chat/completions", "request": body,
                        "prompt_ids": [], "out_ids": [], "max_tokens_logged": logged_out})

    def main_req(max_out):
        emit({"model": "b70", "messages": json.loads(json.dumps(msgs)), "tools": TOOLS,
              "stream": True, "max_tokens": 4096}, max_out)

    turns = min(a.turns, len(steps))
    for t in range(turns):
        main_req(96)
        if t == 2:
            emit({"model": "b70", "stream": True, "max_tokens": 64, "messages": [
                {"role": "system", "content": "You are a title generator. You output ONLY a "
                 "thread title. The title is at most 50 characters, a single line."},
                {"role": "user", "content": msgs[1]["content"]}]}, 24)
        if t == 7:
            emit({"model": "b70", "stream": True, "max_tokens": 256, "messages": [
                {"role": "system", "content": "Summarize what was done in this coding session "
                 "so far, in five bullet points."},
                {"role": "user", "content": "Files read: openai.cc, server.cc, toolcall.cc, "
                 "deps.h. A fix to the stop-string loop is planned."}]}, 48)
        say, name, args = steps[t]
        msgs.append({"role": "assistant", "content": say, "tool_calls": [call(t, name, args)]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{t:04d}", "content": results[t]})
    msgs.append({"role": "assistant", "content": "The fix is in and the protocol test passes."})
    msgs.append({"role": "user", "content": "Thanks. Now explain the pending-token protocol in "
                 "two sentences."})
    main_req(96)
    for i, r in enumerate(records):
        with open(os.path.join(a.out, f"{i + 1:06d}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f)
    print(f"{len(records)} requests ({turns + 1} main, 2 side) in {a.out}")


if __name__ == "__main__":
    main()
