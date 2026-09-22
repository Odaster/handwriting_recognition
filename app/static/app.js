const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const previewWrap = document.getElementById("preview-wrap");
const preview = document.getElementById("preview");
const filename = document.getElementById("filename");
const recognizeButton = document.getElementById("recognize");
const enhance = document.getElementById("enhance");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");
const meta = document.getElementById("meta");
const copyButton = document.getElementById("copy");
const downloadButton = document.getElementById("download");

let selectedFile = null;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function setResultEnabled(enabled) {
  copyButton.disabled = !enabled;
  downloadButton.disabled = !enabled;
}

function useFile(file) {
  if (!file) {
    return;
  }
  selectedFile = file;
  recognizeButton.disabled = false;
  filename.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
  setStatus("Файл выбран. Нажмите «Распознать».");
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  useFile(file);
});

fileInput.addEventListener("change", () => {
  useFile(fileInput.files[0]);
});

recognizeButton.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  const form = new FormData();
  form.append("file", selectedFile);

  recognizeButton.disabled = true;
  setResultEnabled(false);
  setStatus("Распознаём… Первый запуск может занять несколько минут.");

  try {
    const response = await fetch(`/api/recognize?enhance=${enhance.checked}`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || "Не удалось распознать текст.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    result.value = payload.text || "";
    const confidence =
      payload.average_confidence == null
        ? ""
        : ` · уверенность ${(payload.average_confidence * 100).toFixed(0)}%`;
    meta.textContent = `${payload.engine} · строк: ${payload.lines.length}${confidence}`;
    setResultEnabled(Boolean(result.value));
    setStatus("Готово.", "ok");
  } catch (error) {
    result.value = "";
    meta.textContent = "";
    setStatus(error.message || "Ошибка распознавания.", "error");
  } finally {
    recognizeButton.disabled = !selectedFile;
  }
});

copyButton.addEventListener("click", async () => {
  if (!result.value) {
    return;
  }
  await navigator.clipboard.writeText(result.value);
  setStatus("Текст скопирован.", "ok");
});

downloadButton.addEventListener("click", () => {
  if (!result.value) {
    return;
  }
  const blob = new Blob([result.value], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "recognized.txt";
  link.click();
  URL.revokeObjectURL(url);
});
