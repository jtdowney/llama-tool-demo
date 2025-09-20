# llama-tool-demo

A Rust CLI application demonstrating tool-calling capabilities with Large Language Models via llama.cpp. This project showcases how to build an LLM-powered assistant that can execute functions (tools) based on natural language requests.

## Installation

### Prerequisites

- Rust stable (2024 edition)
- C++ compiler (for llama.cpp bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/jtdowney/llama-tool-demo
cd llama-tool-demo

# Build the project
cargo build --release

# The binary will be at ./target/release/llama-tool-demo
```

## Usage

### Using a Local Model

If you have a GGUF model file locally:

```bash
cargo run -- --model path/to/model.gguf
```

### Downloading from HuggingFace

The tool can automatically download models from HuggingFace:

```bash
# Download and use Mistral-7B-Instruct
cargo run -- \
  --hf-repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --hf-file mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Use a specific revision/branch
cargo run -- \
  --hf-repo NousResearch/Hermes-3-Llama-3.1-8B-GGUF \
  --hf-file Hermes-3-Llama-3.1-8B.Q4_K_M.gguf \
  --hf-revision main
```

### Command-Line Options

```bash
LLM tool-calling demonstration with llama.cpp

Usage: llama-tool-demo [OPTIONS] <--model <MODEL>|--hf-file <HF_FILE>>

Options:
  -m, --model <MODEL>                Local path to model.gguf (if provided, this takes precedence)
      --hf-repo <HF_REPO>            HF repo id, e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF
      --hf-file <HF_FILE>            GGUF filename inside the repo, e.g. mistral-7b-instruct-v0.2.Q4_K_M.gguf
      --hf-revision <HF_REVISION>    Optional git revision (branch/tag/commit) [default: main]
  -p, --prompt <PROMPT>              Initial user prompt [default: "What is the current time?"]
      --max-tokens <MAX_TOKENS>      Max generation tokens per assistant turn [default: 256]
  -t, --temperature <TEMPERATURE>    Temperature for sampling (0.0 = deterministic, 1.0 = creative) [default: 0.2]
      --top-p <TOP_P>                Top-p sampling parameter [default: 0.95]
      --max-rounds <MAX_ROUNDS>      Maximum rounds (assistant/tool exchanges) before stopping [default: 4]
      --context-size <CONTEXT_SIZE>  Context size for the model (number of tokens) [default: 8192]
      --template <TEMPLATE>          Optional custom chat template name in the GGUF (use default if empty) [default: ]
  -h, --help                         Print help
  -V, --version                      Print version
```

### Examples

#### Basic Time Query

```bash
# Uses the default prompt "What is the current time?"
cargo run -- --model llama-3.2-3b-instruct.Q4_K_M.gguf
```

#### Custom Conversation

```bash
cargo run -- \
  --model mistral-7b.gguf \
  --prompt "Hello! Can you tell me what time it is?" \
  --temperature 0.7
```

#### Extended Conversation

```bash
cargo run -- \
  --model llama-3.2-3b-instruct.Q4_K_M.gguf \
  --prompt "What's the time? Also, let me know when you're done." \
  --max-rounds 10 \
  --max-tokens 512
```

## How It Works

1. **Model Initialization**: The app loads a GGUF model either from disk or downloads it from HuggingFace
2. **System Prompt**: A specialized prompt template instructs the model to output JSON when calling tools
3. **Chat Loop**:
   - User provides input
   - Model generates a response
   - If response contains a tool call (JSON), the tool is executed
   - Tool results are fed back to the model
   - Process continues until `end_conversation` is called or max rounds reached
4. **Tool Execution**: Tools are simple Rust functions that take JSON arguments and return string results

## Extending with New Tools

To add a new tool, modify the `builtin_tools()` function in `src/main.rs`:

```rust
fn builtin_tools() -> Vec<ToolSpec> {
    vec![
        // ... existing tools ...
        ToolSpec {
            name: "weather".into(),
            description: "Get weather for a location".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        },
    ]
}
```

Then implement the tool in `run_tool()`:

```rust
fn run_tool(call: &ToolCall) -> anyhow::Result<String> {
    match call.tool_name.as_str() {
        // ... existing tools ...
        "weather" => {
            let location = call.arguments.get("location")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            // Your implementation here
            Ok(format!("Weather in {}: Sunny, 72°F", location))
        }
        other => bail!("Unknown tool: {other}"),
    }
}
```

## Supported Models

This tool works with instruction-tuned models in GGUF format. Recommended models:

- **Mistral-7B-Instruct**: Good balance of size and capability
- **Llama-3.2-3B-Instruct**: Smaller, faster option

Models should be quantized (Q4_K_M, Q5_K_M, etc.) for optimal performance.

## Logging

Set the `RUST_LOG` environment variable to control logging verbosity:

```bash
# Show debug information
RUST_LOG=debug cargo run -- --model model.gguf

# Show only info and above
RUST_LOG=info cargo run -- --model model.gguf

# Show trace-level details
RUST_LOG=trace cargo run -- --model model.gguf
```
