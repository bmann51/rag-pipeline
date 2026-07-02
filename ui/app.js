const els = {
  healthBadge: document.getElementById("healthBadge"),
  genBadge: document.getElementById("genBadge"),
  pdfFiles: document.getElementById("pdfFiles"),
  uploadBtn: document.getElementById("uploadBtn"),
  resetBtn: document.getElementById("resetBtn"),
  ingestionResult: document.getElementById("ingestionResult"),
  queryInput: document.getElementById("queryInput"),
  topK: document.getElementById("topK"),
  askBtn: document.getElementById("askBtn"),
  statusBanner: document.getElementById("statusBanner"),
  answerCard: document.getElementById("answerCard"),
  answerText: document.getElementById("answerText"),
  citationSection: document.getElementById("citationSection"),
  citationChips: document.getElementById("citationChips"),
  diagnostics: document.getElementById("diagnostics"),
  chunksContainer: document.getElementById("chunksContainer"),
};

const uiState = {
  chunksById: new Map(),
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function formatAnswerText(answer) {
  if (!answer) {
    return "";
  }

  // Remove inline chunk-id citations from displayed prose.
  const noBracketedCitations = answer.replace(/\[[0-9a-fA-F-]{8,}\]/g, "");
  const noBareUuidCitations = noBracketedCitations.replace(
    /\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b/g,
    ""
  );

  return noBareUuidCitations
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([,.;!?])/g, "$1")
    .trim();
}

function setStatus(message, kind = "ok") {
  els.statusBanner.textContent = `Status: ${message}`;
  els.statusBanner.classList.remove("ok", "warn");
  els.statusBanner.classList.add(kind);
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    els.healthBadge.textContent = `Health: ${data.status}`;
    els.genBadge.textContent = "Generation: check /query response";
  } catch (error) {
    els.healthBadge.textContent = "Health: unavailable";
  }
}

function renderIngestionResult(result) {
  els.ingestionResult.textContent = pretty(result);
}

function renderAnswer(answer, citedChunkIds) {
  if (!answer) {
    els.answerCard.classList.remove("hidden");
    els.answerText.textContent =
      "No generated answer returned. Check retrieved chunks and diagnostics below.";
    els.answerText.classList.add("answer-empty");
    els.citationSection.classList.add("hidden");
    els.citationChips.innerHTML = "";
    return;
  }

  els.answerCard.classList.remove("hidden");
  els.answerText.classList.remove("answer-empty");
  els.answerText.textContent = formatAnswerText(answer);
  els.citationChips.innerHTML = "";
  if (!citedChunkIds || citedChunkIds.length === 0) {
    els.citationSection.classList.add("hidden");
    return;
  }

  els.citationSection.classList.remove("hidden");

  for (const chunkId of citedChunkIds || []) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = chunkId;
    chip.addEventListener("click", () => {
      const node = document.getElementById(`chunk-${chunkId}`);
      if (node) {
        node.scrollIntoView({ behavior: "smooth", block: "center" });
        node.style.outline = "2px solid #0f6e61";
        setTimeout(() => {
          node.style.outline = "";
        }, 1400);
      }
    });
    els.citationChips.appendChild(chip);
  }
}

function formatScore(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(3);
}

function renderChunks(chunks) {
  uiState.chunksById.clear();
  els.chunksContainer.innerHTML = "";

  if (!chunks || chunks.length === 0) {
    els.chunksContainer.innerHTML = "<p class=\"subtle\">No chunks returned.</p>";
    return;
  }

  for (const chunk of chunks) {
    uiState.chunksById.set(chunk.chunk_id, chunk);

    const wrapper = document.createElement("article");
    wrapper.className = "chunk";
    wrapper.id = `chunk-${chunk.chunk_id}`;

    const head = document.createElement("div");
    head.className = "chunk-head";

    const title = document.createElement("strong");
    title.textContent = chunk.source_file;

    const meta = document.createElement("span");
    meta.className = "chunk-meta";
    meta.textContent = `p.${chunk.page_start}-${chunk.page_end}  | rel ${formatScore(chunk.relevance_score)} | kw ${formatScore(chunk.keyword_score)} | sem ${formatScore(chunk.semantic_score)}`;

    const text = document.createElement("p");
    text.className = "chunk-text";
    text.textContent = chunk.text;

    head.appendChild(title);
    head.appendChild(meta);
    wrapper.appendChild(head);
    wrapper.appendChild(text);
    els.chunksContainer.appendChild(wrapper);
  }
}

function renderDiagnostics(diag) {
  els.diagnostics.textContent = pretty(diag || {});
}

async function onUpload() {
  const files = els.pdfFiles.files;
  if (!files || files.length === 0) {
    setStatus("Please select at least one PDF before upload.", "warn");
    return;
  }

  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }

  setStatus("Uploading files...", "ok");
  try {
    const response = await fetch("/ingestion/pdfs", {
      method: "POST",
      body: form,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Upload failed");
    }

    renderIngestionResult(data);
    setStatus("Upload completed.", "ok");
  } catch (error) {
    renderIngestionResult({ error: String(error) });
    setStatus(`Upload failed: ${error}`, "warn");
  }
}

async function onReset() {
  setStatus("Resetting ingested data...", "ok");
  try {
    const response = await fetch("/ingestion/reset", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Reset failed");
    }

    renderIngestionResult(data);
    renderAnswer(null, []);
    renderChunks([]);
    renderDiagnostics({});
    setStatus("Ingestion data cleared.", "ok");
  } catch (error) {
    setStatus(`Reset failed: ${error}`, "warn");
  }
}

async function onAsk() {
  const query = els.queryInput.value.trim();
  if (!query) {
    setStatus("Enter a question first.", "warn");
    return;
  }

  const topK = Number(els.topK.value || 5);
  setStatus("Querying...", "ok");

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Query failed");
    }

    renderAnswer(data.generated_answer, data.cited_chunk_ids);
    renderChunks(data.retrieved_chunks || []);
    renderDiagnostics(data.diagnostics || {});

    if (data.status === "retrieval_complete") {
      setStatus(`retrieval_complete (${(data.retrieved_chunks || []).length} chunks)`, "ok");
    } else {
      setStatus(data.status, "warn");
    }
  } catch (error) {
    setStatus(`Query failed: ${error}`, "warn");
  }
}

els.uploadBtn.addEventListener("click", onUpload);
els.resetBtn.addEventListener("click", onReset);
els.askBtn.addEventListener("click", onAsk);

els.queryInput.value = "";
refreshHealth();
