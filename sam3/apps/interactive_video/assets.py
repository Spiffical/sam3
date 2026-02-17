
PLAYER_JS = r"""
(() => {
  // Prevent double initialization
  if (window.sam3_player_active) {
    console.log("[SAM3] Player already active, skipping init");
    return;
  }
  window.sam3_player_active = true;

  function log(msg, data) {
    // const ts = new Date().toISOString().split('T')[1].slice(0, -1);
    // const str = data ? `${msg} ${JSON.stringify(data)}` : msg;
    // console.log(`[SAM3 ${ts}] ${str}`);
  }

  log("Bootstrap v5 - Script Injected via DOM");

  const queryDeep = (selector) => {
    const visited = new Set();
    const queue = [document];
    while (queue.length) {
      const root = queue.shift();
      if (!root || visited.has(root)) {
        continue;
      }
      visited.add(root);
      let match = null;
      try {
        match = root.querySelector(selector);
      } catch (err) {
        // log("selector error", { selector, err: err.message });
      }
      if (match) {
        return match;
      }
      const getAll = root.querySelectorAll ? root.querySelectorAll("*") : [];
      for (const el of getAll) {
        if (el.shadowRoot && !visited.has(el.shadowRoot)) {
          queue.push(el.shadowRoot);
        }
      }
    }
    return null;
  };

  const queryBridge = (selector, innerSelector = null) => {
    const host = queryDeep(selector);
    if (!host) {
      return null;
    }
    if (!innerSelector) {
      return host.shadowRoot || host;
    }
    if (host.matches && host.matches(innerSelector)) {
      return host;
    }
    if (host.shadowRoot) {
      const inside = host.shadowRoot.querySelector(innerSelector);
      if (inside) {
        return inside;
      }
    }
    return host.querySelector(innerSelector);
  };

  const ensureElements = () => {
    const requiredSelectors = [
      { name: "frame-data", host: "#sam3-frame-data", target: "textarea" },
      { name: "frame-pointer", host: "#sam3-frame-pointer", target: "textarea" },
      { name: "click-payload", host: "#sam3-click-payload", target: "textarea" },
      { name: "click-trigger", host: "#sam3-click-trigger", target: "button" },
      { name: "propagate-trigger", host: "#sam3-propagate-trigger", target: "button" },
      { name: "canvas", host: "#sam3-canvas", target: "canvas" },
      { name: "play-pause-btn", host: "#sam3-play-pause", target: "button" },
      { name: "timeline", host: "#sam3-timeline", target: "input" },
    ];
    const missingList = requiredSelectors
      .map((entry) => ({ name: entry.name, node: queryBridge(entry.host, entry.target) }))
      .filter((entry) => !entry.node)
      .map((entry) => entry.name);
    const canvasPresent = !!queryDeep("#sam3-canvas");
    
    if (missingList.length || !canvasPresent) {
      log("Waiting for elements...", { missing: missingList, canvas: canvasPresent });
      setTimeout(ensureElements, 1000);
      return;
    }
    log("Elements ready, initializing player");
    setupPlayer();
  };

  window.sam3_manual_init = () => {
    log("Manual init triggered");
    ensureElements();
  };

  function setupPlayer() {
    log("setupPlayer starting");
    const canvas = queryDeep("#sam3-canvas");
    if (!canvas) {
      log("canvas not found in setupPlayer");
      ensureElements();
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      log("canvas context missing");
      ensureElements();
      return;
    }
    
    const playPauseBtn = queryDeep("#sam3-play-pause");
    const propagateBtn = queryDeep("#sam3-propagate-ui");
    const layoutBtn = queryDeep("#sam3-expand-btn");
    const timeline = queryDeep("#sam3-timeline");
    
    const frameDataBox = queryBridge("#sam3-frame-data", "textarea");
    const framePointerBox = queryBridge("#sam3-frame-pointer", "textarea");
    const clickPayloadBox = queryBridge("#sam3-click-payload", "textarea");
    const clickTriggerButton = queryBridge("#sam3-click-trigger", "button");
    const propagateTriggerButton = queryBridge("#sam3-propagate-trigger", "button");

    log("Setup found elements:", { playPauseBtn: !!playPauseBtn, timeline: !!timeline, propagateBtn: !!propagateBtn });

    if (!frameDataBox) {
      log("CRITICAL: frameDataBox missing despite ensureElements passing");
      return;
    }

    let frames = [];
    let images = [];
    let currentFrame = 0;
    let playing = false;
    let timerId = null;
    const fps = 24;
    let frameDir = "";

    function buildFrameUrl(name) {
      if (!name) return null;
      // If it's already a full URL, file= path, or data URI, leave it alone
      if (name.startsWith("http") || name.startsWith("/file=") || name.startsWith("data:")) return name;
      
      // Construct the full path
      let fullPath = frameDir ? `${frameDir}/${name}` : name;
      
      // Clean up double slashes
      fullPath = fullPath.replace(/\/+/g, "/");
      
      // Ensure it starts with / if it's an absolute path (which it is in /tmp)
      if (!fullPath.startsWith("/")) fullPath = "/" + fullPath;
      
      // Gradio expects /file=<absolute_path>
      return `/file=${fullPath}`;
    }

    function updatePointer() {
      if (framePointerBox) {
        framePointerBox.value = String(currentFrame);
        framePointerBox.dispatchEvent(new Event("input", { bubbles: true }));
      }
      if (timeline) timeline.value = String(currentFrame);
    }

    function drawFrame(index) {
      if (!images[index]) return;
      const img = images[index];
      if (!img.complete) {
        img.onload = () => drawFrame(index);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    function setFrames(payload) {
      log("setFrames called", { type: payload.type, count: payload.frames ? payload.frames.length : 0 });
      if (payload.dir) frameDir = payload.dir;
      if (payload.width && payload.height) {
        canvas.width = payload.width;
        canvas.height = payload.height;
      }
      if (payload.type === "clear") {
        frames = [];
        images = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (timeline) { timeline.max = "0"; timeline.value = "0"; }
        currentFrame = 0;
        return;
      }
      if (payload.type === "full" && Array.isArray(payload.frames)) {
        frames = payload.frames.map(buildFrameUrl);
        images = frames.map((src) => {
          const img = new Image();
          img.onload = () => {
            if (img === images[currentFrame]) drawFrame(currentFrame);
          };
          img.onerror = (err) => log("Image load error", src);
          img.src = src;
          return img;
        });
        if (timeline) {
          timeline.max = Math.max(0, frames.length - 1).toString();
          timeline.value = "0";
        }
        currentFrame = 0;
        drawFrame(currentFrame);
        updatePointer();
      } else if (payload.type === "patch" && Array.isArray(payload.patches)) {
        payload.patches.forEach((patch) => {
          if (typeof patch.index === "number" && patch.index >= 0 && patch.index < frames.length) {
            const url = buildFrameUrl(patch.frame);
            frames[patch.index] = url;
            const img = new Image();
            img.onload = () => {
              if (img === images[currentFrame]) drawFrame(currentFrame);
            };
            img.src = url;
            images[patch.index] = img;
          }
        });
        drawFrame(currentFrame);
      }
      // Reset Propagate Button UI
      const propagateBtn = queryDeep("#sam3-propagate-ui");
      const overlay = queryDeep("#sam3-loading-overlay");
      if (propagateBtn) {
          propagateBtn.innerText = "⚡ Propagate";
          propagateBtn.disabled = false;
      }
      if (overlay) overlay.classList.add("hidden");
    }

    let lastFramePayload = "";
    const pollFramePayload = () => {
      if (!frameDataBox) return;
      const val = frameDataBox.value;
      if (val && val !== lastFramePayload) {
        log("New frame payload detected", { length: val.length });
        lastFramePayload = val;
        try {
          const payload = JSON.parse(val);
          setFrames(payload);
        } catch (err) {
          log("Failed to parse frame payload", err.message);
        }
      }
      setTimeout(pollFramePayload, 200);
    };
    pollFramePayload();

    let lastPointerValue = "";
    const pollPointer = () => {
      const val = framePointerBox.value;
      if (val && val !== lastPointerValue) {
        lastPointerValue = val;
        const idx = Math.max(0, Math.min(images.length - 1, Number(val) || 0));
        currentFrame = idx;
        if (!playing) {
          drawFrame(currentFrame);
          if (timeline) timeline.value = String(currentFrame);
        }
      }
      requestAnimationFrame(pollPointer);
    };
    pollPointer();

    function stopPlayback() {
      playing = false;
      if (timerId) { clearTimeout(timerId); timerId = null; }
    }

    function stepPlayback() {
      if (!playing || images.length === 0) return;
      currentFrame = (currentFrame + 1) % images.length;
      if (currentFrame % 24 === 0) log("Playback", { frame: currentFrame });
      drawFrame(currentFrame);
      updatePointer();
      timerId = setTimeout(stepPlayback, 1000 / fps);
    }

    // Helper to update Play/Pause Text
    function updatePlayPauseUI() {
        if (!playPauseBtn) return;
        playPauseBtn.innerText = playing ? "⏸ Pause" : "▶ Play";
    }

    if (playPauseBtn) playPauseBtn.addEventListener("click", () => {
       if (!images.length) return;
       if (playing) {
           stopPlayback();
       } else {
           playing = true;
           updatePointer();
           stepPlayback();
       }
       updatePlayPauseUI();
    });

    // Timeline
    if (timeline) timeline.addEventListener("input", () => {
      if (!images.length) return;
      stopPlayback();
      playing = false;
      updatePlayPauseUI();
      currentFrame = Number(timeline.value) || 0;
      drawFrame(currentFrame);
      updatePointer();
    });

    // Canvas Interaction
    // Disable context menu for Right Click
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("pointerup", (event) => {
      if (!images.length) return;
      // Only handle Left (0) and Right (2)
      if (event.button !== 0 && event.button !== 2) return;
      
      stopPlayback();
      playing = false; 
      updatePlayPauseUI();

      const rect = canvas.getBoundingClientRect();
      const relX = (event.clientX - rect.left) / rect.width;
      const relY = (event.clientY - rect.top) / rect.height;
      
      // Right click = Negative (0), Left click = Positive (1)
      const label = (event.button === 2) ? 0 : 1; 
      log("Click detected", { button: event.button, label });

      let objIdEl = queryBridge("#sam3-obj-id", "input");
      if (!objIdEl) objIdEl = queryBridge("#sam3-obj-id", "textarea");
      
      if (!objIdEl) {
         const container = queryDeep("#sam3-obj-id");
         if (container) objIdEl = container.querySelector("input") || container.querySelector("textarea");
      }
      
      const objId = objIdEl ? Number(objIdEl.value) : 1;
      const payload = {
        frame_index: currentFrame,
        rel_x: relX,
        rel_y: relY,
        label: label,
        obj_id: objId,
      };
      
      if (clickPayloadBox) {
        clickPayloadBox.value = JSON.stringify(payload);
        clickPayloadBox.dispatchEvent(new Event("input", { bubbles: true }));
      }
      updatePointer();
      if (clickTriggerButton) {
          setTimeout(() => clickTriggerButton.click(), 50);
      }
    });

    if (layoutBtn) layoutBtn.addEventListener("click", () => {
       let wrapper = canvas.closest(".sam3-canvas-wrapper");
       if (!wrapper) wrapper = canvas.parentElement;
       if (wrapper) wrapper.classList.toggle("fullscreen");
    });
    
    // Close fullscreen on click background or ESC
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
             const w = document.querySelector(".sam3-canvas-wrapper.fullscreen");
             if (w) w.classList.remove("fullscreen");
        }
    });
    document.addEventListener("click", (e) => {
         if (e.target && e.target.classList.contains("sam3-canvas-wrapper") && e.target.classList.contains("fullscreen")) {
             e.target.classList.remove("fullscreen");
         }
    });

    if (propagateBtn) propagateBtn.addEventListener("click", () => {
      if (!images.length) return;
      stopPlayback();
      const propagateBtn = queryDeep("#sam3-propagate-ui");
      const overlay = queryDeep("#sam3-loading-overlay");
      
      propagateBtn.innerText = "⏳ Processing...";
      propagateBtn.disabled = true;
      if (overlay) {
          overlay.classList.remove("hidden");
          // Poll status box text
          const statusBox = document.getElementById("sam3-status-box");
          const overlayText = overlay.querySelector(".status-text");
          if (statusBox && overlayText) {
              const poller = setInterval(() => {
                  if (overlay.classList.contains("hidden")) {
                      clearInterval(poller);
                      return;
                  }
                  if (statusBox.innerText) {
                      if (statusBox.innerText.includes("Propagating")) {
                          overlayText.innerText = statusBox.innerText;
                      } else if (statusBox.innerText.includes("Propagation complete") || statusBox.innerText.includes("Error")) {
                          overlay.classList.add("hidden");
                          propagateBtn.innerText = "⚡ Propagate";
                          propagateBtn.disabled = false; 
                          clearInterval(poller);
                      }
                  }
              }, 200);
          }
      }

      if (propagateTriggerButton) {
          setTimeout(() => propagateTriggerButton.click(), 50);
      }
    });
  }

  ensureElements();
})();
"""

PLAYER_HTML = f"""
<style>
.sam3-player-container {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}}
.sam3-canvas-wrapper {{
  width: 100%;
  resize: both;
  overflow: auto;
  min-height: 200px;
  background: #111;
  border: 1px solid #333;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative; /* For overlay */
}}
.sam3-canvas-wrapper canvas {{
  max-width: 100%;
  height: auto;
}}
#sam3-loading-overlay {{
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.85); /* Darker background */
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 50;
  font-family: sans-serif;
  font-size: 1.5rem; /* Larger font */
  font-weight: bold;
  text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}}
.hidden {{ display: none !important; }}
.sam3-controls {{
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  align-items: center;
  background: #333;
  padding: 0.5rem;
  border-radius: 4px;
}}
.sam3-controls button {{
  padding: 0.4rem 0.8rem;
  background: #444;
  color: white;
  border: 1px solid #555;
  border-radius: 4px;
  cursor: pointer;
}}
.sam3-controls button:hover {{
  background: #555;
}}
.sam3-controls input[type="range"] {{
  flex-grow: 1;
  cursor: pointer;
}}
#sam3-frame-data,
#sam3-frame-pointer,
#sam3-click-payload,
#sam3-click-trigger,
#sam3-propagate-trigger {{
  display: none !important;
}}
#sam3-debug-console {{
  background: #222;
  color: #0f0;
  font-family: monospace;
  padding: 0.5rem;
  max-height: 100px; /* Smaller */
  overflow-y: auto;
  border: 1px solid #444;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  white-space: pre-wrap;
}}
.sam3-canvas-wrapper.fullscreen {{
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 10000;
  background: black;
  display: flex;
  justify-content: center;
  align-items: center;
}}
.sam3-canvas-wrapper.fullscreen canvas {{
   max-width: 100%;
   max-height: 100%;
   width: auto;
   height: auto;
}}
</style>
<div class="sam3-player-container">
  <div class="sam3-canvas-wrapper">
    <canvas id="sam3-canvas"></canvas>
    <div id="sam3-loading-overlay" class="hidden">
       <div style="margin-bottom: 10px; font-size: 2rem;">⏳</div>
       <div class="status-text">Propagating...</div>
       <div style="font-size: 0.8rem; margin-top:5px; color:#ccc;">(This may take a minute)</div>
    </div>
  </div>
  <div class="sam3-controls">
    <button id="sam3-play-pause" title="Play/Pause">▶ Play</button>
    <input id="sam3-timeline" type="range" min="0" value="0" step="1">
    <button id="sam3-propagate-ui" title="Propagate Annotations">⚡ Propagate</button>
    <button id="sam3-expand-btn" title="Fullscreen">⤢</button>
  </div>
  <div id="sam3-debug-console" style="display:none;"></div>
</div>

<!-- Hidden script source -->
<script id="sam3-script-source" type="text/plain">
{PLAYER_JS}
</script>

<!-- Trigger execution via onerror hack -->
<img src="_" style="display:none" onerror="
  console.log('[SAM3] Triggering script via onerror');
  try {{
    var scriptContent = document.getElementById('sam3-script-source').textContent;
    var s = document.createElement('script');
    s.textContent = scriptContent;
    document.body.appendChild(s);
  }} catch (e) {{
    console.error('[SAM3] Failed to inject script:', e);
  }}
"/>
"""
