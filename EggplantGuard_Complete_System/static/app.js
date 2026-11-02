const $ = id => document.getElementById(id);
let selectedFile = null, stream = null;
const imageInput = $("imageInput"), preview = $("preview");
function showError(msg=""){ $("error").textContent = msg; }
function setFile(file){
  if(!file) return;
  if(!["image/jpeg","image/png"].includes(file.type)){showError("Please choose a JPG, JPEG, or PNG image.");return;}
  if(file.size > 10*1024*1024){showError("Image must be 10 MB or smaller.");return;}
  selectedFile=file; showError("");
  preview.src=URL.createObjectURL(file);
  $("fileName").textContent=file.name;
  $("previewWrap").classList.remove("hidden");
  $("classifyBtn").disabled=false;
}
imageInput.addEventListener("change",e=>setFile(e.target.files[0]));
$("clearImage").onclick=()=>{selectedFile=null;imageInput.value="";$("previewWrap").classList.add("hidden");$("classifyBtn").disabled=true;};
$("uploadTab").onclick=()=>{stopCamera();$("uploadTab").classList.add("active");$("cameraTab").classList.remove("active");$("uploadArea").classList.remove("hidden");$("cameraArea").classList.add("hidden");};
$("cameraTab").onclick=()=>{$("cameraTab").classList.add("active");$("uploadTab").classList.remove("active");$("uploadArea").classList.add("hidden");$("cameraArea").classList.remove("hidden");};
async function startCamera(){
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});$("video").srcObject=stream;$("capture").disabled=false;showError("");}
 catch(e){showError("Camera unavailable. Check browser permission or use image upload.");}
}
function stopCamera(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}$("video").srcObject=null;$("capture").disabled=true;}
$("startCamera").onclick=startCamera;$("stopCamera").onclick=stopCamera;
$("capture").onclick=()=>{
 const v=$("video");if(!v.videoWidth){showError("Wait for the camera preview, then capture.");return;}
 const c=document.createElement("canvas");c.width=v.videoWidth;c.height=v.videoHeight;c.getContext("2d").drawImage(v,0,0);
 c.toBlob(blob=>{const f=new File([blob],"eggplant-camera-capture.jpg",{type:"image/jpeg"});setFile(f);$("previewWrap").classList.remove("hidden");},"image/jpeg",.92);
};
$("classifyBtn").onclick=async()=>{
 if(!selectedFile)return;
 showError("");$("classifyBtn").disabled=true;$("classifyBtn").textContent="Analyzing image…";
 const form=new FormData();form.append("image",selectedFile);
 try{
  const res=await fetch("/api/classify",{method:"POST",body:form});const data=await res.json();
  if(!res.ok)throw new Error(data.error||"Classification failed.");
  $("emptyResult").classList.add("hidden");$("result").classList.remove("hidden");
  $("category").textContent=data.category;$("confidence").textContent=data.confidence.toFixed(2)+"%";
  $("meterFill").style.width=Math.min(100,data.confidence)+"%";$("guidance").textContent=data.guidance;$("disclaimer").textContent=data.disclaimer;
  loadHistory();
 }catch(e){showError(e.message);}
 finally{$("classifyBtn").disabled=!selectedFile;$("classifyBtn").innerHTML='Classify image <span>→</span>';}
};
async function loadStatus(){
 try{const r=await fetch("/api/status"),d=await r.json();$("modelStatus").textContent=d.message;$("statusDot").style.background=d.model_ready?"#4d946a":"#d59a37";}
 catch(e){$("modelStatus").textContent="Could not check model status.";}
}
async function loadHistory(){
 const [h,s]=await Promise.all([fetch("/api/history").then(r=>r.json()),fetch("/api/stats").then(r=>r.json())]);
 $("totalScans").textContent=s.total;
 $("historyBody").innerHTML=h.length?h.map(x=>`<tr><td>${escapeHtml(x.filename)}</td><td>${escapeHtml(x.category)}</td><td>${Number(x.confidence).toFixed(1)}%</td><td>${escapeHtml(x.scanned_at)}</td><td><button class="text-btn" data-delete="${x.id}">Delete</button></td></tr>`).join(""):'<tr><td colspan="5" class="muted">No scans recorded yet.</td></tr>';
 document.querySelectorAll("[data-delete]").forEach(b=>b.onclick=async()=>{await fetch("/api/history/"+b.dataset.delete,{method:"DELETE"});loadHistory();});
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
$("refreshHistory").onclick=loadHistory;loadStatus();loadHistory();
window.addEventListener("beforeunload",stopCamera);
