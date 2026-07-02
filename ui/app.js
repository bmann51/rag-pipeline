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
  disclaimerBanner: document.getElementById("disclaimerBanner"),
  citationSection: document.getElementById("citationSection"),
  citationChips: document.getElementById("citationChips"),
  diagnostics: document.getElementById("diagnostics"),
  chunksContainer: document.getElementById("chunksContainer"),
  fileListDetails: document.getElementById("fileListDetails"),
  fileList: document.getElementById("fileList"),
  fileCount: document.getElementById("fileCount"),
};

const uiState = {
  chunksById: new Map(),
  chunkRankById: new Map(),
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

async function refreshFileList() {
  try {
    const response = await fetch("/ingestion/documents");
    if (!response.ok) return;
    const docs = await response.json();
    els.fileCount.textContent = docs.length;
    els.fileList.innerHTML = "";
    if (docs.length === 0) {
      const empty = document.createElement("li");
      empty.className = "file-list-empty";
      empty.textContent = "No files ingested yet.";
      els.fileList.appendChild(empty);
      return;
    }
    for (const doc of docs) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "file-name";
      name.title = doc.original_filename;
      name.textContent = doc.original_filename;
      const meta = document.createElement("span");
      meta.className = "file-meta";
      meta.textContent = `${doc.page_count} pp · ${doc.chunk_count} chunks`;
      li.appendChild(name);
      li.appendChild(meta);
      els.fileList.appendChild(li);
    }
    if (docs.length > 0) {
      els.fileListDetails.open = true;
    }
  } catch {
    // non-fatal — list stays stale
  }
}

function renderAnswer(answer, citedChunkIds, disclaimer) {
  if (disclaimer) {
    els.disclaimerBanner.textContent = disclaimer;
    els.disclaimerBanner.classList.remove("hidden");
  } else {
    els.disclaimerBanner.textContent = "";
    els.disclaimerBanner.classList.add("hidden");
  }

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
    const cited = uiState.chunksById.get(chunkId);
    const rank = uiState.chunkRankById.get(chunkId);
    chip.textContent = cited
      ? `${displayName(cited.source_file)} · p.${cited.page_start}–${cited.page_end}`
      : "unknown chunk";
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

function scoreClass(value) {
  if (value === null || value === undefined) return "low";
  return value >= 0.6 ? "high" : value >= 0.35 ? "mid" : "low";
}

const _UUID_PREFIX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i;
function displayName(sourceFile) {
  return sourceFile.split("/").pop().replace(_UUID_PREFIX, "");
}

function renderChunks(chunks) {
  uiState.chunksById.clear();
  uiState.chunkRankById.clear();
  els.chunksContainer.innerHTML = "";

  if (!chunks || chunks.length === 0) {
    els.chunksContainer.innerHTML = "<p class=\"subtle\">No chunks returned.</p>";
    return;
  }

  for (const [i, chunk] of chunks.entries()) {
    const rank = i + 1;
    uiState.chunksById.set(chunk.chunk_id, chunk);
    uiState.chunkRankById.set(chunk.chunk_id, rank);

    const wrapper = document.createElement("article");
    wrapper.className = "chunk";
    wrapper.id = `chunk-${chunk.chunk_id}`;

    const head = document.createElement("div");
    head.className = "chunk-head";

    const titleWrap = document.createElement("div");
    titleWrap.style.cssText = "display:flex;align-items:center;gap:0.4rem;min-width:0";

    const rankSpan = document.createElement("span");
    rankSpan.className = "chunk-rank";
    rankSpan.textContent = `#${rank}`;

    const title = document.createElement("strong");
    title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    title.textContent = displayName(chunk.source_file);

    titleWrap.appendChild(rankSpan);
    titleWrap.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "chunk-meta";

    const pages = document.createElement("span");
    pages.className = "chunk-pages";
    pages.textContent = `p.${chunk.page_start}–${chunk.page_end}`;
    meta.appendChild(pages);

    for (const [label, value] of [
      ["rel", chunk.relevance_score],
      ["kw", chunk.keyword_score],
      ["sem", chunk.semantic_score],
    ]) {
      const badge = document.createElement("span");
      badge.className = `score-badge ${scoreClass(value)}`;
      badge.textContent = `${label} ${formatScore(value)}`;
      meta.appendChild(badge);
    }

    const text = document.createElement("p");
    text.className = "chunk-text";
    text.textContent = chunk.text;

    head.appendChild(titleWrap);
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
    await refreshFileList();
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
    renderAnswer(null, [], null);
    renderChunks([]);
    renderDiagnostics({});
    setStatus("Ingestion data cleared.", "ok");
    await refreshFileList();
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

    renderChunks(data.retrieved_chunks || []);
    renderAnswer(data.generated_answer, data.cited_chunk_ids, data.disclaimer ?? null);
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
refreshFileList();
