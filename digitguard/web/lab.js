const byId = (id) => document.getElementById(id);
const grid = byId("grid");
const pixels = Array(64).fill(0);
let erasing = false;
let activePointer = null;
let revision = 0;

function renderCell(index) {
  const value = pixels[index];
  const shade = 15 + Math.round(value * 15);
  const cell = grid.children[index];
  cell.style.backgroundColor = `rgb(${shade}, ${shade}, ${shade})`;
  cell.setAttribute("aria-label", `Row ${Math.floor(index / 8) + 1}, column ${index % 8 + 1}, intensity ${value}`);
}

function resetOutput(status) {
  revision++;
  byId("status").textContent = status;
  byId("candidates").textContent = "";
  byId("scores").replaceChildren();
}

function paint(cell) {
  if (!cell?.classList.contains("cell")) return;
  const index = Number(cell.dataset.index);
  pixels[index] = erasing ? 0 : Number(byId("brush").value);
  renderCell(index);
  resetOutput("Input changed. Run prediction.");
}

for (let index = 0; index < 64; index++) {
  const cell = document.createElement("button");
  cell.className = "cell";
  cell.dataset.index = index;
  cell.addEventListener("click", (event) => { if (event.detail === 0) paint(cell); });
  grid.append(cell);
  renderCell(index);
}
for (let digit = 0; digit <= 9; digit++) byId("digit").add(new Option(digit, digit));

grid.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || activePointer !== null) return;
  activePointer = event.pointerId;
  grid.setPointerCapture(activePointer);
  paint(event.target);
});
grid.addEventListener("pointermove", (event) => {
  if (event.pointerId === activePointer) paint(document.elementFromPoint(event.clientX, event.clientY));
});
grid.addEventListener("lostpointercapture", () => { activePointer = null; });

byId("erase").onclick = () => {
  erasing = !erasing;
  byId("erase").setAttribute("aria-pressed", erasing);
};
byId("clear").onclick = () => {
  pixels.fill(0);
  pixels.forEach((_, index) => renderCell(index));
  resetOutput("Grid cleared.");
};

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`Request failed (HTTP ${response.status})`);
  return response.json();
}

byId("sample").onclick = async () => {
  resetOutput("Loading sample…");
  const token = revision;
  try {
    const sample = await request(`/v1/example?digit=${byId("digit").value}`);
    if (token !== revision) return;
    pixels.splice(0, 64, ...sample.pixels);
    pixels.forEach((_, index) => renderCell(index));
    resetOutput(`Loaded a labeled ${sample.digit}. Run prediction.`);
  } catch (error) {
    if (token === revision) resetOutput(error.message);
  }
};

byId("predict").onclick = async () => {
  resetOutput("Predicting…");
  const token = revision;
  try {
    const result = await request("/v1/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pixels}),
    });
    if (token !== revision) return;
    byId("status").textContent = result.reason === "blank_input"
      ? "The grid is blank."
      : result.prediction === null
        ? "Abstained · uncertain input"
        : `Accepted digit ${result.prediction} · ${(result.confidence * 100).toFixed(1)}% confidence`;
    if (result.probabilities === null) return;
    byId("candidates").textContent = `Conformal set: ${result.candidates.join(", ") || "empty"}`;
    result.probabilities.forEach((probability, digit) => {
      const row = document.createElement("li");
      const bar = document.createElement("progress");
      bar.max = 1;
      bar.value = probability;
      bar.setAttribute("aria-label", `Digit ${digit} probability`);
      row.append(String(digit), bar, `${(probability * 100).toFixed(1)}%`);
      byId("scores").append(row);
    });
  } catch (error) {
    if (token === revision) resetOutput(error.message);
  }
};
