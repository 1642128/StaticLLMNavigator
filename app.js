import { CreateWebWorkerMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("agentInput");
const statusbarEl = document.getElementById("statusbar");
const modelStatusEl = document.getElementById("modelStatus");
const iframeEl = document.querySelector(".webview");

// --- helpers
function addLine(role, text) {
  const p = document.createElement("p");
  p.className = "line";
  p.innerHTML = `<strong>${role}:</strong> <span class="text"></span>`;
  p.querySelector(".text").textContent = text;
  messagesEl.appendChild(p);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return p;
}

function setStatus(text) {
  statusbarEl.textContent = text;
}

// --- create WebLLM engine in a worker
let engine;
let chatHistory = [
  {
    role: "system",
    content:
      "You are an in-browser assistant. Be concise. " +
      "If the user asks to open a website, respond with: OPEN_URL: <url> on its own line."
  }
];

// Choose a SMALL model for first-time UX.
// (Model IDs vary by WebLLM’s supported list; you can swap this as needed.)
const MODEL_ID = "Llama-3.1-8B-Instruct"; // shown in WebLLM worker example docs [2](https://onnxruntime.ai/docs/tutorials/web/build-web-app.html)[3](https://onnxruntime.ai/)

(async function init() {
  addLine("System", "Booting WebLLM…");

  const worker = new Worker("./worker.js", { type: "module" });

  engine = await CreateWebWorkerMLCEngine(worker, MODEL_ID, {
    initProgressCallback: (p) => {
      // docs show progress callback usage for model loading [2](https://onnxruntime.ai/docs/tutorials/web/build-web-app.html)[1](https://aibit.im/blog/post/webllm-run-llms-in-browser-with-webgpu-full-guide-here)
      setStatus(`Loading model…`);
      // p shape can differ; we avoid assuming fields
      console.log("initProgress", p);
    }
  });

  modelStatusEl.textContent = "Ready";
  modelStatusEl.classList.add("ok");
  addLine("System", "Model ready. Ask me something!");
  setStatus("Ready.");
})().catch((err) => {
  console.error(err);
  modelStatusEl.textContent = "Error";
  modelStatusEl.classList.add("bad");
  addLine("System", "Failed to load model. Check console.");
  setStatus("Model load error.");
});

// --- streaming chat
async function streamReply(userText) {
  chatHistory.push({ role: "user", content: userText });

  const assistantLine = addLine("Agent", "");
  const textSpan = assistantLine.querySelector(".text");

  // WebLLM uses OpenAI-compatible chat.completions.create and supports streaming [4](https://dev.to/hexshift/run-ai-models-entirely-in-the-browser-using-webassembly-onnx-runtime-no-backend-required-4lag)[5](https://github.com/ggml-org/llama.cpp/discussions/6055)
  const stream = await engine.chat.completions.create({
    messages: chatHistory,
    stream: true
  });

  let full = "";
  for await (const chunk of stream) {
    const delta = chunk?.choices?.[0]?.delta?.content ?? "";
    if (!delta) continue;
    full += delta;
    textSpan.textContent = full;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  chatHistory.push({ role: "assistant", content: full });
  return full;
}

// --- URL open convention (simple MVP tool)
// LLM outputs: OPEN_URL: https://...
function maybeHandleOpenUrl(replyText) {
  const match = replyText.match(/^\s*OPEN_URL:\s*(https?:\/\/\S+)\s*$/m);
  if (match) {
    const url = match[1];
    iframeEl.src = url;
    addLine("System", `Opened ${url}`);
    return true;
  }
  return false;
}

// --- input handling
inputEl.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  if (!engine) return;

  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  addLine("You", text);
  setStatus("Thinking…");

  try {
    const reply = await streamReply(text);
    maybeHandleOpenUrl(reply);
    setStatus("Ready.");
  } catch (err) {
    console.error(err);
    addLine("System", "Error generating reply. See console.");
    setStatus("Error.");
  }
});
