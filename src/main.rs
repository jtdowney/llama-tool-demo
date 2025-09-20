use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use anyhow::{Context, anyhow, bail};
use clap::{ArgGroup, Parser};
use hf_hub::{Repo, api::sync::ApiBuilder};
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special, params::LlamaModelParams,
    },
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use serde_json::{Deserializer, json};
use tracing::{debug, error, info, info_span, trace, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const SYSTEM_PROMPT_TEMPLATE: &str = include_str!("system_prompt.md");
const DEFAULT_CONTEXT_SIZE: u32 = 8192;
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_TEMPERATURE: f32 = 0.2;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_MAX_ROUNDS: usize = 4;

enum RoundResult {
    Continue,
    Stop,
}

/// Command-line arguments for the tool-demo application
#[derive(Parser, Debug)]
#[command(
    name = "tool-demo",
    about = "LLM tool-calling demonstration with llama.cpp",
    version,
    author
)]
#[command(group(ArgGroup::new("model_source").required(true).args(["model", "hf_file"])))]
struct Args {
    /// Local path to model.gguf (if provided, this takes precedence)
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// HF repo id, e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF
    #[arg(long, requires = "hf_file")]
    hf_repo: Option<String>,

    /// GGUF filename inside the repo, e.g. mistral-7b-instruct-v0.2.Q4_K_M.gguf
    #[arg(long, requires = "hf_repo")]
    hf_file: Option<String>,

    /// Optional git revision (branch/tag/commit)
    #[arg(long, default_value = "main")]
    hf_revision: String,

    /// Initial user prompt
    #[arg(short = 'p', long, default_value = "What is the current time?")]
    prompt: String,

    /// Max generation tokens per assistant turn
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS)]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative)
    #[arg(short = 't', long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f32,

    /// Top-p sampling parameter
    #[arg(long, default_value_t = DEFAULT_TOP_P)]
    top_p: f32,

    /// Maximum rounds (assistant/tool exchanges) before stopping
    #[arg(long, default_value_t = DEFAULT_MAX_ROUNDS)]
    max_rounds: usize,

    /// Context size for the model (number of tokens)
    #[arg(long, default_value_t = DEFAULT_CONTEXT_SIZE)]
    context_size: u32,

    /// Optional custom chat template name in the GGUF (use default if empty)
    #[arg(long, default_value = "")]
    template: String,
}

/// Specification for a tool that can be called by the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Represents a tool invocation request from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCall {
    tool_name: String,
    arguments: serde_json::Value,
}

/// Initializes the llama.cpp backend
fn initialize_backend() -> Result<LlamaBackend, anyhow::Error> {
    let mut backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;
    backend.void_logs();
    debug!("initialized llama.cpp backend");
    Ok(backend)
}

/// Loads a GGUF model from the specified path
fn load_model(backend: &LlamaBackend, path: impl AsRef<Path>) -> Result<LlamaModel, anyhow::Error> {
    let path = path.as_ref();
    info!(path = %path.display(), "loading model");

    let model_params = LlamaModelParams::default();
    debug!("using default model parameters");
    let model = LlamaModel::load_from_file(backend, path, &model_params)
        .with_context(|| format!("Failed to load model from {}", path.display()))?;

    info!("model loaded successfully");
    Ok(model)
}

/// Creates an inference context with appropriate parameters
fn create_context<'a>(
    backend: &LlamaBackend,
    model: &'a LlamaModel,
    context_size: u32,
) -> anyhow::Result<LlamaContext<'a>> {
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(context_size))
        .with_offload_kqv(true)
        .with_flash_attention(true);

    let context = model
        .new_context(backend, params)
        .context("Failed to create inference context")?;

    debug!(
        context_size = context_size,
        "created context with token capacity"
    );
    Ok(context)
}

/// Returns the list of available tools for the LLM to use
fn builtin_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "get_time".into(),
            description: "Return the current local time (RFC9557)".into(),
            parameters: json!({"type":"object","properties":{},"additionalProperties":false}),
        },
        ToolSpec {
            name: "end_conversation".into(),
            description:
                "End the conversation when the task is complete or no further assistance is needed"
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final message to display before ending the conversation"
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
        },
    ]
}

/// Attempts to extract a JSON tool call from the assistant's response
fn try_parse_tool_call(text: &str) -> Option<ToolCall> {
    let position = text.find('{')?;
    let mut values = Deserializer::from_str(&text[position..]).into_iter::<serde_json::Value>();
    let value = match values.next()? {
        Ok(v) => v,
        Err(err) => {
            trace!(?err, "failed to parse JSON fragment for tool call");
            return None;
        }
    };

    trace!(value = ?value, "parsed tool call candidate");

    if value.get("tool_name").is_some() && value.get("arguments").is_some() {
        match serde_json::from_value::<ToolCall>(value) {
            Ok(call) => {
                debug!(call = ?call, "parsed tool call");
                Some(call)
            }
            Err(err) => {
                trace!(?err, "failed to deserialize tool call");
                None
            }
        }
    } else {
        None
    }
}

/// Executes a tool call and returns its result
fn run_tool(call: &ToolCall) -> anyhow::Result<String> {
    match call.tool_name.as_str() {
        "get_time" => {
            let timestamp = jiff::Zoned::now().to_string();
            Ok(timestamp)
        }
        "end_conversation" => {
            let message = call
                .arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Conversation ended.");
            Ok(message.to_string())
        }
        other => bail!("Unknown tool: {other}"),
    }
}

/// Downloads a model from `HuggingFace` Hub
fn download_model_from_hf(
    repo_id: &str,
    filename: &str,
    revision: &str,
) -> anyhow::Result<PathBuf> {
    info!(
        repo_id = repo_id,
        filename = filename,
        "downloading model from HuggingFace: {}/{}",
        repo_id,
        filename
    );
    let api = ApiBuilder::new()
        .build()
        .context("Failed to initialize HuggingFace API")?;
    let repo = Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    );
    let path = api
        .repo(repo)
        .get(filename)
        .with_context(|| format!("Failed to download {filename} from {repo_id}"))?;
    info!(path = ?path, "model downloaded successfully");
    Ok(path)
}

/// Resolves the model path, either from local file or `HuggingFace` download
fn get_model_path(args: &Args) -> anyhow::Result<PathBuf> {
    if let Some(ref path) = args.model {
        trace!(?path, "model path provided");
        Ok(path.clone())
    } else {
        let repo = args.hf_repo.clone().unwrap();
        let file = args.hf_file.clone().unwrap();

        download_model_from_hf(&repo, &file, &args.hf_revision)
    }
}

/// Constructs the initial chat messages including system prompt and user input
fn build_messages(tools: &[ToolSpec], user: &str) -> anyhow::Result<Vec<LlamaChatMessage>> {
    let tool_blob =
        serde_json::to_string_pretty(tools).context("serializing tool specifications")?;
    let system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{tool_blob}", &tool_blob);
    trace!(system_prompt, "loaded system prompt template");

    Ok(vec![
        LlamaChatMessage::new("system".into(), system_prompt)
            .context("constructing system message")?,
        LlamaChatMessage::new("user".into(), user.to_string())
            .context("constructing initial user message")?,
    ])
}

/// Generates a response from the model given the current context
fn generate_response(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    prompt_tokens: &[LlamaToken],
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
) -> anyhow::Result<String> {
    let mut n_past: i32 = prompt_tokens
        .len()
        .try_into()
        .context("Token count exceeds i32 capacity - prompt too long")?;

    // Warn if approaching context limit (80% threshold)
    let token_count = prompt_tokens.len();
    let context_size = ctx.n_ctx() as usize;
    if token_count > context_size * 8 / 10 {
        warn!(
            token_count = token_count,
            context_size = context_size,
            "prompt tokens approaching context limit ({}% used)",
            (token_count * 100) / context_size
        );
    }

    trace!(tokens = n_past, "tokenized prompt");

    let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
    batch.add_sequence(prompt_tokens, 0, true)?;
    ctx.decode(&mut batch)?;
    trace!("decoded initial prompt batch");

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::top_p(top_p, 1),
        LlamaSampler::dist(0),
    ]);

    let mut out = String::new();
    let mut prev = model.token_eos();
    let mut logits_ready = true;

    for i in 0..max_tokens {
        if !logits_ready {
            let mut batch = LlamaBatch::new(1, 1);
            batch.add(prev, n_past, &[0], true)?;
            ctx.decode(&mut batch)?;
            trace!("decoded token batch");
            n_past += 1;
        }

        let tok = sampler.sample(ctx, 0);
        if model.is_eog_token(tok) {
            debug!("end-of-generation token detected");
            break;
        }

        let piece = model.token_to_str(tok, Special::Plaintext)?;
        out.push_str(&piece);

        prev = tok;
        sampler.accept(tok);
        logits_ready = false;

        // Check if we've reached max tokens
        if i == max_tokens - 1 {
            debug!(max_tokens, "reached maximum token generation limit");
        }
    }

    trace!(response = out, "generated response");
    Ok(out)
}

/// Processes a tool call and updates the chat history
fn process_tool_call(
    call: &ToolCall,
    chat: &mut Vec<LlamaChatMessage>,
    assistant_response: String,
) -> anyhow::Result<RoundResult> {
    debug!(
        tool_name = call.tool_name,
        arguments = %call.arguments,
        "tool call detected"
    );

    let tool_result = run_tool(call).map_err(|e| {
        error!(tool_name = call.tool_name, error = %e, "tool execution error");
        anyhow!("Tool execution failed: {e}")
    })?;

    debug!(result = tool_result, "tool result");

    // Check if this is the end_conversation tool
    if call.tool_name == "end_conversation" {
        println!("{tool_result}");
        chat.push(
            LlamaChatMessage::new("assistant".into(), tool_result.clone())
                .context("recording final assistant message")?,
        );
        return Ok(RoundResult::Stop);
    }

    // For other tools, continue the conversation
    chat.push(
        LlamaChatMessage::new("assistant".into(), assistant_response)
            .context("recording assistant tool call")?,
    );
    chat.push(
        LlamaChatMessage::new(
            "tool".into(),
            json!({"tool_name": call.tool_name, "result": tool_result}).to_string(),
        )
        .context("recording tool result")?,
    );

    Ok(RoundResult::Continue)
}

/// Executes a single round of conversation (assistant response + potential tool call)
fn chat_round(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    tmpl: &LlamaChatTemplate,
    chat: &mut Vec<LlamaChatMessage>,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
) -> Result<RoundResult, anyhow::Error> {
    ctx.clear_kv_cache();
    trace!("cleared KV cache");
    trace!("starting new conversation round");

    // Prepare prompt and tokenize
    let prompt = model.apply_chat_template(tmpl, chat, true)?;
    trace!(?prompt, "prepared prompt");
    let toks = model.str_to_token(&prompt, AddBos::Never)?;

    // Generate response from model
    let response = generate_response(ctx, model, &toks, temperature, top_p, max_tokens)?;

    // Check if response contains a tool call
    if let Some(call) = try_parse_tool_call(&response) {
        process_tool_call(&call, chat, response)
    } else {
        // No tool call detected, display the response
        println!("Assistant: {response}");

        chat.push(
            LlamaChatMessage::new("assistant".into(), response)
                .context("recording assistant message")?,
        );

        debug!("conversation continues after assistant response");

        Ok(RoundResult::Continue)
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    info!("starting tool-demo CLI");

    // Parse command-line arguments
    let args = Args::parse();
    let model_path = get_model_path(&args)?;

    // Initialize LLM backend and model
    let backend = initialize_backend()?;
    let model = load_model(&backend, model_path)?;
    let mut ctx = create_context(&backend, &model, args.context_size)?;

    // Prepare tools and initial messages
    let tools = builtin_tools();
    let mut chat = build_messages(&tools, &args.prompt)?;

    // Get chat template
    let tmpl = if args.template.is_empty() {
        model.chat_template(None)?
    } else {
        debug!(template = args.template, "using custom chat template");
        model.chat_template(Some(&args.template))?
    };

    info!(prompt = args.prompt, "starting conversation");

    // Main conversation loop
    let mut completed_rounds = 0;
    for round in 1..=args.max_rounds {
        let span = info_span!("round", round);
        let _guard = span.enter();

        completed_rounds = round;
        if let RoundResult::Stop = chat_round(
            &mut ctx,
            &model,
            &tmpl,
            &mut chat,
            args.temperature,
            args.top_p,
            args.max_tokens,
        )? {
            info!(rounds = round, "conversation ended");
            break;
        }
    }

    if completed_rounds >= args.max_rounds {
        warn!(
            max_rounds = args.max_rounds,
            "reached maximum rounds without natural conversation end"
        );
    }

    Ok(())
}
