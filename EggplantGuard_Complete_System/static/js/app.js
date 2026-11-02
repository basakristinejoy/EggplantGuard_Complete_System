/**
 * EggplantGuard - Frontend Application Logic
 * ==========================================
 * Handles:
 *  - Whole-Dropzone Click & Drag-and-Drop Image Loading
 *  - Live Camera Stream, Viewfinder Alignment, and Snapshot Capture
 *  - Animated Laser Scanner Analysis
 *  - Out-of-Distribution / Non-Eggplant Sample Handling
 *  - Real-Time YOLO & Computer Vision Inference
 *  - Real-Time Audit Log Table with Live Search & Filtering
 *  - CSV Export & Record Reset
 */

const $ = (id) => document.getElementById(id);

let currentFile = null;
let cameraStream = null;
let allHistoryRecords = [];

// ============================================================================
// Notification Toast System
// ============================================================================
function showToast(message, type = "error") {
  const container = $("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${escapeHtml(message)}</span>
    <button style="background:none;border:none;cursor:pointer;font-weight:bold;color:inherit;font-size:1.1rem;" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    setTimeout(() => toast.remove(), 350);
  }, 4500);
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[m]);
}

// ============================================================================
// File Selection & Drag-and-Drop
// ============================================================================
const dropzone = $("dropzone");
const fileInput = $("fileInput");

// Allow clicking anywhere inside the dropzone box to open file explorer
dropzone.addEventListener("click", (e) => {
  if (e.target.id !== "fileInput" && !e.target.closest("label")) {
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("drag-active");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files && files.length > 0) {
    handleFileSelected(files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileSelected(e.target.files[0]);
  }
});

function handleFileSelected(file) {
  if (!file) return;

  const ext = file.name ? file.name.split(".").pop().toLowerCase() : "";
  const validExts = ["jpg", "jpeg", "png", "webp", "bmp"];
  const isImage = (file.type && file.type.startsWith("image/")) || validExts.includes(ext);

  if (!isImage) {
    showToast("Invalid format. Please upload a JPG, JPEG, PNG, or WEBP photo.", "error");
    return;
  }

  // 10 MB limit
  if (file.size > 10 * 1024 * 1024) {
    showToast("File size exceeds 10 MB. Please choose a smaller photo.", "warning");
    return;
  }

  currentFile = file;

  // Show preview
  const previewImg = $("previewImage");
  previewImg.src = URL.createObjectURL(file);
  $("previewFileName").textContent = file.name;
  $("previewFileSize").textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

  $("previewContainer").classList.remove("hidden");
  $("btnClassify").disabled = false;
  showToast("Eggplant image loaded. Ready for diagnostic scan.", "success");
}

$("btnClearImage").addEventListener("click", () => {
  clearImageSelection();
});

function clearImageSelection() {
  currentFile = null;
  fileInput.value = "";
  $("previewContainer").classList.add("hidden");
  $("btnClassify").disabled = true;
}

// ============================================================================
// Tab Navigation (Upload vs Camera)
// ============================================================================
$("tabUpload").addEventListener("click", () => {
  $("tabUpload").classList.add("active");
  $("tabCamera").classList.remove("active");
  $("uploadPane").classList.remove("hidden");
  $("cameraPane").classList.add("hidden");
  stopCameraStream();
});

$("tabCamera").addEventListener("click", () => {
  $("tabCamera").classList.add("active");
  $("tabUpload").classList.remove("active");
  $("uploadPane").classList.add("hidden");
  $("cameraPane").classList.remove("hidden");
});

// ============================================================================
// Live Camera Controls & Viewfinder
// ============================================================================
async function startCameraStream() {
  try {
    const constraints = {
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    $("cameraVideo").srcObject = cameraStream;
    $("cameraPlaceholder").classList.add("hidden");
    $("btnStartCamera").disabled = true;
    $("btnCapture").disabled = false;
    $("btnStopCamera").disabled = false;
    showToast("Live camera stream active. Align leaf/fruit inside reticle.", "success");
  } catch (err) {
    console.error("Camera access error:", err);
    showToast("Camera unavailable. Check browser permissions or use file upload.", "error");
  }
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  $("cameraVideo").srcObject = null;
  $("cameraPlaceholder").classList.remove("hidden");
  $("btnStartCamera").disabled = false;
  $("btnCapture").disabled = true;
  $("btnStopCamera").disabled = true;
}

$("btnStartCamera").addEventListener("click", startCameraStream);
$("btnStopCamera").addEventListener("click", stopCameraStream);

$("btnCapture").addEventListener("click", () => {
  const video = $("cameraVideo");
  if (!video.videoWidth) {
    showToast("Waiting for camera feed...", "warning");
    return;
  }

  const canvas = $("captureCanvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    const captureName = `eggplant_snap_${Date.now()}.jpg`;
    const capturedFile = new File([blob], captureName, { type: "image/jpeg" });
    handleFileSelected(capturedFile);
    showToast("Snapshot captured successfully!", "success");
  }, "image/jpeg", 0.95);
});

// ============================================================================
// AI Classification Execution with Laser Effect
// ============================================================================
$("btnClassify").addEventListener("click", async () => {
  if (!currentFile) {
    showToast("Please choose an image or capture a photo first.", "warning");
    return;
  }

  const btn = $("btnClassify");
  btn.disabled = true;
  $("btnScanText").textContent = "Scanning sample...";

  // Start animated laser scanner
  const laser = $("laserScanner");
  if (laser) laser.classList.add("scanning");

  const formData = new FormData();
  formData.append("image", currentFile);

  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Diagnostic request failed.");
    }

    displayClassificationResults(data);

    if (data.category === "Invalid / Non-Eggplant Sample") {
      showToast("Notice: Sample does not match eggplant characteristics.", "warning");
    } else {
      showToast(`Scan complete: ${data.category}`, "success");
    }

    // Refresh history & stats
    loadHistoryRecords();
    loadDashboardStats();

  } catch (err) {
    console.error("Classification error:", err);
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    $("btnScanText").textContent = "Analyze Sample with AI";
    if (laser) laser.classList.remove("scanning");
  }
});

function displayClassificationResults(result) {
  $("resultEmpty").classList.add("hidden");
  $("resultBody").classList.remove("hidden");

  // Category & Badge
  $("resultCategory").textContent = result.category;
  $("resultConditionText").textContent = result.condition;

  const badge = $("categoryBadge");
  badge.textContent = result.category;
  badge.style.backgroundColor = result.badge_color ? result.badge_color + "22" : "#f3e8ff";
  badge.style.color = result.badge_color || "#5c2d91";

  // Engine attribution
  const engineBadge = $("engineBadge");
  if (engineBadge) {
    engineBadge.textContent = result.model_source || "Vision Engine";
  }

  // Non-Eggplant Warning Box Toggle
  const invalidBox = $("invalidSampleNotice");
  if (result.category === "Invalid / Non-Eggplant Sample") {
    invalidBox.classList.remove("hidden");
    $("meterFill").style.background = "#64748b";
  } else {
    invalidBox.classList.add("hidden");
    $("meterFill").style.background = "linear-gradient(90deg, #16a34a, #7b2cbf)";
  }

  // Confidence Meter
  const conf = Number(result.confidence).toFixed(2);
  $("resultConfidence").textContent = `${conf}%`;
  $("meterFill").style.width = `${Math.min(100, Math.max(5, conf))}%`;

  // Actionable Recommendations Checklist
  const recList = $("recommendationList");
  recList.innerHTML = "";
  if (result.recommendations && result.recommendations.length > 0) {
    result.recommendations.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      recList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "Maintain regular crop monitoring and standard field management practices.";
    recList.appendChild(li);
  }

  const engineTag = result.model_source ? ` · Engine: ${result.model_source}` : "";
  $("resultTimestamp").textContent = `Audited at: ${result.scanned_at}${engineTag}`;

  // Smooth scroll into results view on mobile
  if (window.innerWidth <= 768) {
    $("resultsCard").scrollIntoView({ behavior: "smooth" });
  }
}

// ============================================================================
// History Table & Search / Filter
// ============================================================================
async function loadHistoryRecords() {
  const categoryFilter = $("filterCategory").value;
  const url = categoryFilter ? `/api/history?category=${encodeURIComponent(categoryFilter)}` : "/api/history";

  try {
    const res = await fetch(url);
    allHistoryRecords = await res.json();
    renderFilteredHistory();
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function renderFilteredHistory() {
  const searchQuery = ($("searchHistory") ? $("searchHistory").value : "").toLowerCase().trim();
  const tbody = $("historyTableBody");

  let records = allHistoryRecords;
  if (searchQuery) {
    records = records.filter(item => 
      item.filename.toLowerCase().includes(searchQuery) || 
      item.category.toLowerCase().includes(searchQuery)
    );
  }

  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No scan records matching current query.</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map((item) => {
    let badgeClass = "badge-healthy";
    if (item.category.includes("Miner")) badgeClass = "badge-warning";
    if (item.category.includes("Borer") || item.category.includes("Damaged")) badgeClass = "badge-danger";
    if (item.category.includes("Fresh")) badgeClass = "badge-fresh";
    if (item.category.includes("Invalid")) badgeClass = "badge-neutral";

    return `
      <tr>
        <td><strong>#${item.id}</strong></td>
        <td><code>${escapeHtml(item.filename)}</code></td>
        <td><span class="badge ${badgeClass}">${escapeHtml(item.category)}</span></td>
        <td><strong>${Number(item.confidence).toFixed(2)}%</strong></td>
        <td>${escapeHtml(item.scanned_at)}</td>
        <td>
          <button class="btn btn-sm btn-ghost text-danger" onclick="deleteHistoryItem(${item.id})">
            Delete
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

if ($("searchHistory")) {
  $("searchHistory").addEventListener("input", renderFilteredHistory);
}
$("filterCategory").addEventListener("change", loadHistoryRecords);

window.deleteHistoryItem = async function (id) {
  if (!confirm(`Are you sure you want to delete scan record #${id}?`)) return;

  try {
    const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      showToast(`Record #${id} removed.`, "success");
      loadHistoryRecords();
      loadDashboardStats();
    } else {
      showToast("Could not delete record.", "error");
    }
  } catch (err) {
    showToast("Error deleting scan record.", "error");
  }
};

$("btnClearAllHistory").addEventListener("click", async () => {
  if (!confirm("Are you sure you want to permanently clear all scan records?")) return;

  try {
    const res = await fetch("/api/history/clear", { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      showToast("Audit history database has been cleared.", "success");
      loadHistoryRecords();
      loadDashboardStats();
    }
  } catch (err) {
    showToast("Failed to clear database.", "error");
  }
});

// ============================================================================
// Dashboard Statistics & Status Check
// ============================================================================
async function loadDashboardStats() {
  try {
    const res = await fetch("/api/stats");
    const stats = await res.json();

    $("statTotalScans").textContent = stats.total_scans || 0;

    const b = stats.breakdown || {};
    $("statHealthyLeaf").textContent = b["Healthy Leaf"] || 0;
    $("statFreshFruit").textContent = b["Fresh-Market-Ready Fruit"] || 0;

    const issues = (b["Leaf Miner Damage"] || 0) + (b["Damaged-Borer-Infested Fruit"] || 0);
    $("statPestIssues").textContent = issues;
  } catch (err) {
    console.error("Failed to fetch statistics:", err);
  }
}

async function checkModelStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    const dot = $("statusDot");
    const text = $("modelStatusText");

    if (data.is_trained_yolo) {
      dot.classList.add("ready");
      text.textContent = "YOLO Model Active";
    } else {
      dot.classList.add("ready");
      text.textContent = "Precision Engine Active";
    }
  } catch (err) {
    $("modelStatusText").textContent = "API Offline";
  }
}

// ============================================================================
// Initialization on Page Load
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  checkModelStatus();
  loadDashboardStats();
  loadHistoryRecords();
});

// Clean up camera stream on page unload
window.addEventListener("beforeunload", stopCameraStream);
