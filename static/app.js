// ASTRA VISION 5.0 — Self-Learning AI Object Recognition Frontend

// Elements
const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const browseButton = document.querySelector('#browse-button');
const plusIcon = document.querySelector('#plus-icon');
const resultSection = document.querySelector('#result');
const errorBanner = document.querySelector('#error');

// Visual Image Elements & Toggles
const previewImg = document.querySelector('#preview');
const annotatedImg = document.querySelector('#annotated-image');
const heatmapImg = document.querySelector('#heatmap');
const viewAnnotatedBtn = document.querySelector('#view-annotated');
const viewOrigBtn = document.querySelector('#view-orig');
const viewCamBtn = document.querySelector('#view-cam');
const toggleContainer = document.querySelector('#toggle-container');

// Multi-Target UI Elements
const multiTargetBanner = document.querySelector('#multi-target-banner');
const targetCountBadge = document.querySelector('#target-count-badge');
const targetActiveIndicator = document.querySelector('#target-active-indicator');
const targetTabsContainer = document.querySelector('#target-tabs-container');
const allTargetsSection = document.querySelector('#all-targets-section');
const allTargetsCount = document.querySelector('#all-targets-count');
const allTargetsGrid = document.querySelector('#all-targets-grid');

// Optical Zoom Inspector Elements
const opticalZoomBox = document.querySelector('#optical-zoom-box');
const zoomCropImg = document.querySelector('#zoom-crop-img');
const zoomFactorBadge = document.querySelector('#zoom-factor-badge');
const zoomTargetTitle = document.querySelector('#zoom-target-title');
const zoomTargetDesc = document.querySelector('#zoom-target-desc');
const zoomCoords = document.querySelector('#zoom-coords');

// Verification & Punishment Card Elements
const vCard = document.querySelector('#verification-card');
const vPill = document.querySelector('#verdict-pill');
const compLocalName = document.querySelector('#comp-local-name');
const compLocalMeta = document.querySelector('#comp-local-meta');
const compCloudName = document.querySelector('#comp-cloud-name');
const compCloudMeta = document.querySelector('#comp-cloud-meta');
const punishmentAlert = document.querySelector('#punishment-alert');
const punishmentTitle = document.querySelector('#punishment-title');
const punishmentDesc = document.querySelector('#punishment-desc');
const rewardAlert = document.querySelector('#reward-alert');
const retrainLossDiff = document.querySelector('#retrain-loss-diff');
const retrainMemCount = document.querySelector('#retrain-mem-count');

// Mobile Tabs
const mobileTabsBar = document.querySelector('#mobile-tabs-bar');
const tabBtnPhoto = document.querySelector('#tab-btn-photo');
const tabBtnVerdict = document.querySelector('#tab-btn-verdict');
const tabBtnSpecs = document.querySelector('#tab-btn-specs');

const tryAgainBtn = document.querySelector('#try-again');
const browseCatalogBtn = document.querySelector('#browse-full-catalog-btn');

// Modals
const metricsBtn = document.querySelector('#metrics-btn');
const metricsModal = document.querySelector('#metrics-modal');
const modalClose = document.querySelector('#modal-close');

const catalogBtn = document.querySelector('#catalog-btn');
const catalogModal = document.querySelector('#catalog-modal');
const catalogModalClose = document.querySelector('#catalog-modal-close');
const catalogGrid = document.querySelector('#catalog-grid');
const catalogSearch = document.querySelector('#catalog-search');
const catalogCatTabs = document.querySelector('#catalog-category-tabs');

let cachedPlatforms = [];
let activeCatalogCat = 'all';
let currentReconData = null;
let activeTargetIdx = 0;

// Color palette matching backend HUD
const HUD_PALETTE = ['#00ffcc', '#ffb700', '#00e676', '#ff007f', '#00bfff', '#e040fb', '#76ff03'];

// Mobile Tab Switcher
function setMobileView(viewName) {
  if (!resultSection) return;
  resultSection.classList.remove('mobile-view-photo', 'mobile-view-verdict', 'mobile-view-specs');
  resultSection.classList.add(`mobile-view-${viewName}`);

  [tabBtnPhoto, tabBtnVerdict, tabBtnSpecs].forEach((btn) => {
    if (btn) btn.classList.remove('active');
  });

  if (viewName === 'photo' && tabBtnPhoto) tabBtnPhoto.classList.add('active');
  if (viewName === 'verdict' && tabBtnVerdict) tabBtnVerdict.classList.add('active');
  if (viewName === 'specs' && tabBtnSpecs) tabBtnSpecs.classList.add('active');
}

if (tabBtnPhoto) tabBtnPhoto.addEventListener('click', () => setMobileView('photo'));
if (tabBtnVerdict) tabBtnVerdict.addEventListener('click', () => setMobileView('verdict'));
if (tabBtnSpecs) tabBtnSpecs.addEventListener('click', () => setMobileView('specs'));

// File Select Handlers
browseButton.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});
plusIcon.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    analyse(fileInput.files[0]);
  }
});

// Drag and drop handlers
['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
  });
});

dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
    analyse(e.dataTransfer.files[0]);
  }
});

// Reset view
tryAgainBtn.addEventListener('click', () => {
  resultSection.classList.add('hidden');
  if (mobileTabsBar) mobileTabsBar.classList.add('hidden');
  dropZone.classList.remove('hidden');
  errorBanner.classList.add('hidden');
  fileInput.value = '';
  previewImg.src = '';
  if (annotatedImg) annotatedImg.src = '';
  heatmapImg.src = '';
  const uWarn = document.querySelector('#uncertain-warning');
  if (uWarn) uWarn.classList.add('hidden');
  currentReconData = null;
});

// View Toggles: Annotated HUD vs Original vs Grad-CAM
if (viewAnnotatedBtn) {
  viewAnnotatedBtn.addEventListener('click', () => {
    viewAnnotatedBtn.classList.add('active');
    viewOrigBtn.classList.remove('active');
    viewCamBtn.classList.remove('active');
    if (annotatedImg) annotatedImg.classList.remove('hidden');
    previewImg.classList.add('hidden');
    heatmapImg.classList.add('hidden');
  });
}

viewOrigBtn.addEventListener('click', () => {
  viewOrigBtn.classList.add('active');
  if (viewAnnotatedBtn) viewAnnotatedBtn.classList.remove('active');
  viewCamBtn.classList.remove('active');
  previewImg.classList.remove('hidden');
  if (annotatedImg) annotatedImg.classList.add('hidden');
  heatmapImg.classList.add('hidden');
});

viewCamBtn.addEventListener('click', () => {
  viewCamBtn.classList.add('active');
  if (viewAnnotatedBtn) viewAnnotatedBtn.classList.remove('active');
  viewOrigBtn.classList.remove('active');
  heatmapImg.classList.remove('hidden');
  if (annotatedImg) annotatedImg.classList.add('hidden');
  previewImg.classList.add('hidden');
});

// Metrics Modal
metricsBtn.addEventListener('click', async () => {
  metricsModal.classList.remove('hidden');
  try {
    const res = await fetch('/api/metrics');
    if (res.ok) {
      const data = await res.json();
      const ensemble = data.ensemble_model || {};
      if (ensemble.accuracy) {
        document.querySelector('#metric-acc').textContent = `${ensemble.accuracy}%`;
      }
      if (ensemble.macro_f1) {
        document.querySelector('#metric-f1').textContent = `${ensemble.macro_f1}`;
      }
      if (ensemble.macro_roc_auc) {
        document.querySelector('#metric-auc').textContent = `${ensemble.macro_roc_auc}`;
      }
    }
  } catch (err) {
    console.warn('Metrics fetch warning:', err);
  }
});
modalClose.addEventListener('click', () => metricsModal.classList.add('hidden'));
metricsModal.addEventListener('click', (e) => {
  if (e.target === metricsModal) metricsModal.classList.add('hidden');
});

// Platform Catalog Modal
catalogBtn.addEventListener('click', openCatalog);
if (browseCatalogBtn) browseCatalogBtn.addEventListener('click', openCatalog);
catalogModalClose.addEventListener('click', () => catalogModal.classList.add('hidden'));
catalogModal.addEventListener('click', (e) => {
  if (e.target === catalogModal) catalogModal.classList.add('hidden');
});

async function openCatalog() {
  catalogModal.classList.remove('hidden');
  if (cachedPlatforms.length === 0) {
    try {
      const res = await fetch('/api/platforms');
      if (res.ok) {
        const data = await res.json();
        cachedPlatforms = data.platforms || [];
        renderCatalog();
      }
    } catch (err) {
      console.warn('Catalog fetch warning:', err);
    }
  } else {
    renderCatalog();
  }
}

catalogSearch.addEventListener('input', () => renderCatalog());

catalogCatTabs.addEventListener('click', (e) => {
  if (e.target.classList.contains('cat-tab')) {
    catalogCatTabs.querySelectorAll('.cat-tab').forEach((tab) => tab.classList.remove('active'));
    e.target.classList.add('active');
    activeCatalogCat = e.target.dataset.cat;
    renderCatalog();
  }
});

function renderCatalog() {
  const query = catalogSearch.value.toLowerCase().trim();
  const filtered = cachedPlatforms.filter((p) => {
    const matchesCat = activeCatalogCat === 'all' || p.category === activeCatalogCat;
    const matchesQuery =
      !query ||
      p.name.toLowerCase().includes(query) ||
      p.designation.toLowerCase().includes(query) ||
      p.country.toLowerCase().includes(query) ||
      p.role.toLowerCase().includes(query);
    return matchesCat && matchesQuery;
  });

  if (filtered.length === 0) {
    catalogGrid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 30px; text-align: center; color: var(--muted);">
        No items matching "${query}". Try searching for F-22, Ontos, Su-57, Abrams, Apache, or Nimitz.
      </div>
    `;
    return;
  }

  catalogGrid.innerHTML = filtered
    .map(
      (p) => `
    <div class="catalog-card" title="Click to view details">
      <div class="catalog-card-header">
        <strong>${p.name}</strong>
        <span class="catalog-badge">${p.designation}</span>
      </div>
      <span class="catalog-card-country">${p.country}</span>
      <p class="catalog-card-role">${p.role}</p>
      <div class="catalog-card-stats">
        <span>Year: ${p.year_origin.split(' ')[0]}</span>
        <span>Fleet: ${p.operational_count.split(' ')[0]}</span>
      </div>
    </div>
  `
    )
    .join('');
}

// Switches displayed details to a specific target
function selectTarget(idx) {
  if (!currentReconData || !currentReconData.detected_targets || !currentReconData.detected_targets[idx]) return;
  activeTargetIdx = idx;
  const target = currentReconData.detected_targets[idx];
  const color = HUD_PALETTE[idx % HUD_PALETTE.length];

  // Update switcher tabs
  const tabBtns = document.querySelectorAll('.target-tab-btn');
  tabBtns.forEach((btn, bIdx) => {
    if (bIdx === idx) {
      btn.classList.add('active');
      btn.style.borderColor = color;
      btn.style.color = color;
    } else {
      btn.classList.remove('active');
      btn.style.borderColor = 'var(--panel-border)';
      btn.style.color = 'var(--text-dim)';
    }
  });

  // Update target cards in grid
  const cards = document.querySelectorAll('.target-intel-card');
  cards.forEach((c, cIdx) => {
    if (cIdx === idx) {
      c.classList.add('active');
      c.style.borderColor = color;
    } else {
      c.classList.remove('active');
      c.style.borderColor = 'var(--panel-border)';
    }
  });

  if (targetActiveIndicator) {
    targetActiveIndicator.textContent = `Target #${target.id} Active`;
    targetActiveIndicator.style.borderColor = color;
    targetActiveIndicator.style.color = color;
  }

  // Populate Dossier
  document.querySelector('#platform-name').textContent = target.exact_name || target.name;
  document.querySelector('#platform-designation').textContent = target.designation || 'TARGET';
  document.querySelector('#platform-variant').textContent = target.model_variant ? `Variant: ${target.model_variant}` : '';
  document.querySelector('#platform-role').textContent = target.role || target.tactical_analysis || 'Identified Entity';
  document.querySelector('#platform-confidence').textContent = `${target.confidence}%`;

  // Update KPIs with actual historical numbers
  const yr = target.year_origin || (currentReconData && currentReconData.platform && currentReconData.platform.year_origin);
  const blt = target.number_built || (currentReconData && currentReconData.platform && currentReconData.platform.number_built);
  const ops = target.operational_count || (currentReconData && currentReconData.platform && currentReconData.platform.operational_count);

  document.querySelector('#kpi-country').textContent = target.country || 'Global';
  document.querySelector('#kpi-manufacturer').textContent = target.manufacturer || 'Manufacturer / Developer';
  document.querySelector('#kpi-year').textContent = (yr && yr !== 'Modern' && yr !== 'Operational Era') ? yr : (currentReconData && currentReconData.platform && currentReconData.platform.year_origin && currentReconData.platform.year_origin !== 'Modern' ? currentReconData.platform.year_origin : (yr || '—'));
  document.querySelector('#kpi-built').textContent = (blt && blt !== 'In production' && blt !== 'Production Volume') ? blt : (currentReconData && currentReconData.platform && currentReconData.platform.number_built && currentReconData.platform.number_built !== 'In production' ? currentReconData.platform.number_built : (blt || '—'));
  document.querySelector('#kpi-operational').textContent = ops || target.status || 'Active';
  document.querySelector('#kpi-status').textContent = target.status || 'Active Service';

  // Technical Specs
  const specs = target.specs || {};
  document.querySelector('#spec-speed').textContent = specs.speed || 'Standard Rated Performance';
  document.querySelector('#spec-range').textContent =
    specs.combat_range || specs.ceiling ? `${specs.combat_range || ''} · Dimensions: ${specs.ceiling || ''}` : 'Standard operating range';
  document.querySelector('#spec-armament').textContent = specs.armament || 'Standard equipment';
  document.querySelector('#spec-avionics').textContent = specs.avionics || 'Sensors, onboard electronics, and control systems';
  document.querySelector('#spec-signatures').textContent = target.visual_signatures || 'Observable visual signatures.';

  // Optical Zoom Inspector
  if (target.crop_url) {
    opticalZoomBox.classList.remove('hidden');
    zoomCropImg.src = target.crop_url;
    zoomFactorBadge.textContent = target.zoom_factor || (target.is_distant_target ? 'Optical Zoom' : '1.0× Direct Optical');
    zoomTargetTitle.textContent = `Target #${target.id}: ${target.exact_name.split('/')[0].trim()}`;
    zoomTargetDesc.textContent = target.is_distant_target
      ? 'Distant silhouette crop enhanced with sub-pixel reconstruction.'
      : 'Optical crop isolated for visual inspection.';
    if (target.box_pixels) {
      zoomCoords.textContent = `Box Area: [Y: ${target.box_pixels[0]}..${target.box_pixels[2]}, X: ${target.box_pixels[1]}..${target.box_pixels[3]}]`;
    }
  } else {
    opticalZoomBox.classList.add('hidden');
  }
}

// Main Image Analysis & Reconnaissance
async function analyse(file) {
  errorBanner.classList.add('hidden');

  if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
    return showError('Invalid file format. Please upload a JPG, PNG, or WebP image.');
  }

  if (file.size > 10 * 1024 * 1024) {
    return showError('File is too large. The upload limit is 10 MB.');
  }

  browseButton.textContent = '1. Local Model Thinking → 2. AI Checking…';
  browseButton.disabled = true;

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || 'Visual recognition failed.');
    }

    currentReconData = body;
    const objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    saveToGallery(file, body);

    // Annotated HUD Image Setup
    if (body.annotated_image_url && annotatedImg) {
      annotatedImg.src = body.annotated_image_url;
      if (viewAnnotatedBtn) {
        viewAnnotatedBtn.classList.remove('hidden');
        viewAnnotatedBtn.click();
      }
    } else {
      if (viewAnnotatedBtn) viewAnnotatedBtn.classList.add('hidden');
      viewOrigBtn.click();
    }

    // -------------------------------------------------------------
    // NEW VERSION 5: Local Prediction, Verification & Punishment
    // -------------------------------------------------------------
    const vData = body.verification;
    if (vData && vCard) {
      vCard.classList.remove('hidden');

      if (vData.is_offline) {
        // OFFLINE MODE — Local model operating standalone without internet
        if (vPill) {
          vPill.textContent = '⚡ OFFLINE MODE (LOCAL MODEL)';
          vPill.className = 'verdict-pill offline';
        }
        if (compLocalName && vData.local_answer) {
          compLocalName.textContent = vData.local_answer.name || 'Local Prediction';
          compLocalMeta.textContent = `${vData.local_answer.category || 'Object'} (${vData.local_answer.confidence || body.confidence}%)`;
        }
        if (compCloudName) {
          compCloudName.textContent = 'API Verification Skipped';
          compCloudMeta.textContent = 'No internet connection — running standalone';
        }
        if (punishmentAlert) punishmentAlert.classList.add('hidden');
        if (rewardAlert) rewardAlert.classList.add('hidden');
        if (retrainLossDiff) retrainLossDiff.textContent = 'Offline Mode Active';
        if (retrainMemCount) {
          retrainMemCount.textContent = `${vData.learned_samples || (body.learning_status && body.learning_status.active_memory_slots) || 0} Exemplars in Memory`;
        }
      } else if (vData.is_correct) {
        // LOCAL MODEL CORRECT
        if (compLocalName && vData.local_answer) {
          compLocalName.textContent = vData.local_answer.name || 'Local Prediction';
          compLocalMeta.textContent = `${vData.local_answer.category || 'Object'} (${vData.local_answer.confidence || 75}%)`;
        }
        if (compCloudName && vData.verified_answer) {
          compCloudName.textContent = vData.verified_answer.name || 'Verified Entity';
          compCloudMeta.textContent = `${vData.verified_answer.category || 'Object'} (${vData.verified_answer.confidence || 99}%)`;
        }
        if (vPill) {
          vPill.textContent = '✅ CORRECT (+10 XP)';
          vPill.className = 'verdict-pill correct';
        }
        if (rewardAlert) rewardAlert.classList.remove('hidden');
        if (punishmentAlert) punishmentAlert.classList.add('hidden');
      } else {
        // LOCAL MODEL WRONG — SEVERE PUNISHMENT
        if (compLocalName && vData.local_answer) {
          compLocalName.textContent = vData.local_answer.name || 'Local Prediction';
          compLocalMeta.textContent = `${vData.local_answer.category || 'Object'} (${vData.local_answer.confidence || 75}%)`;
        }
        if (compCloudName && vData.verified_answer) {
          compCloudName.textContent = vData.verified_answer.name || 'Verified Entity';
          compCloudMeta.textContent = `${vData.verified_answer.category || 'Object'} (${vData.verified_answer.confidence || 99}%)`;
        }
        if (vPill) {
          vPill.textContent = '⚡ WRONG — PUNISHED';
          vPill.className = 'verdict-pill punished';
        }
        if (punishmentAlert) {
          punishmentAlert.classList.remove('hidden');
          if (punishmentTitle) {
            punishmentTitle.textContent = `SEVERE PUNISHMENT APPLIED (${vData.penalty_points || -50} Points)`;
          }
          if (punishmentDesc) {
            punishmentDesc.textContent = `Local model guessed '${vData.local_answer.name}'. AI check corrected it to '${vData.verified_answer.name}'. Neural weights updated & visual memory stored so it remembers next time!`;
          }
        }
        if (rewardAlert) rewardAlert.classList.add('hidden');

        if (retrainLossDiff) {
          if (vData.loss_before !== null && vData.loss_after !== null) {
            retrainLossDiff.textContent = `${vData.loss_before} → ${vData.loss_after} (Loss Reduced)`;
          } else {
            retrainLossDiff.textContent = `-50 Penalty Points`;
          }
        }
        if (retrainMemCount) {
          retrainMemCount.textContent = `${vData.learned_samples || 1} Exemplars Saved`;
        }
      }
    }

    // AI Check & Consensus Matrix
    const consensus = body.consensus || {};
    document.querySelector('#consensus-latency').textContent = body.latency_sec ? `${body.latency_sec}s latency` : '< 2.0s';

    if (consensus.gemini && consensus.gemini.detected) {
      document.querySelector('#node-gemini-detail').textContent = `${consensus.gemini.detected} (${consensus.gemini.confidence || 99}%)`;
    } else {
      document.querySelector('#node-gemini-detail').textContent = body.is_offline ? 'Offline (Skipped)' : 'AI Check Online';
    }

    if (consensus.openrouter && consensus.openrouter.detected) {
      document.querySelector('#node-openrouter-detail').textContent = `${consensus.openrouter.detected} (${consensus.openrouter.confidence || 98}%)`;
    } else {
      document.querySelector('#node-openrouter-detail').textContent = body.is_offline ? 'Offline (Skipped)' : 'Consensus Checked';
    }

    if (consensus.groq) {
      if (consensus.groq.detected) {
        document.querySelector('#node-groq-detail').textContent = `${consensus.groq.detected}`;
      } else {
        document.querySelector('#node-groq-detail').textContent = body.is_offline ? 'Offline (Skipped)' : 'Synthesized & Verified';
      }
    }

    if (consensus.local_neural) {
      if (consensus.local_neural.confidence) {
        document.querySelector('#node-neural-detail').textContent = `${consensus.local_neural.category} (${consensus.local_neural.confidence}%)`;
      } else {
        document.querySelector('#node-neural-detail').textContent = consensus.local_neural.category || 'Answered First';
      }
    }

    // Category Level Meta
    document.querySelector('#category-badge').textContent = body.label;
    document.querySelector('#category-conf').textContent = `${body.confidence}% Confidence`;
    document.querySelector('#model-tag').textContent = body.model;

    // Low-Confidence Warning ("Uncertain Prediction")
    const uncertainWarning = document.querySelector('#uncertain-warning');
    if (uncertainWarning) {
      if (body.uncertain || (body.confidence && body.confidence < 60)) {
        uncertainWarning.classList.remove('hidden');
        const uDesc = document.querySelector('#uncertain-desc');
        if (uDesc && body.explanation && body.explanation.includes('decision margin')) {
          uDesc.textContent = body.explanation;
        }
      } else {
        uncertainWarning.classList.add('hidden');
      }
    }

    // Multi-Target Reconnaissance Setup
    const targets = body.detected_targets || [];
    const targetCount = body.target_count || targets.length || 1;

    if (targetCount > 1 && multiTargetBanner && targetTabsContainer) {
      multiTargetBanner.classList.remove('hidden');
      targetCountBadge.textContent = `${targetCount} Objects Found`;
      targetTabsContainer.classList.remove('hidden');

      // Render Target Tabs
      targetTabsContainer.innerHTML = targets
        .map((t, idx) => {
          const color = HUD_PALETTE[idx % HUD_PALETTE.length];
          const short = t.exact_name.split('/')[0].trim();
          return `
          <button class="target-tab-btn ${idx === 0 ? 'active' : ''}" type="button" onclick="selectTarget(${idx})" style="border-color: ${idx === 0 ? color : 'var(--panel-border)'}; color: ${idx === 0 ? color : 'var(--text-dim)'};">
            <span class="target-tab-num" style="color: ${color};">[${t.id}]</span>
            <span>${short}</span>
            <span class="target-tab-conf">${t.confidence}%</span>
          </button>
        `;
        })
        .join('');
    } else {
      if (multiTargetBanner) multiTargetBanner.classList.add('hidden');
      if (targetTabsContainer) targetTabsContainer.classList.add('hidden');
    }

    // Render All Detected Targets Grid
    if (allTargetsSection && allTargetsGrid) {
      if (targets.length > 1) {
        allTargetsSection.classList.remove('hidden');
        if (allTargetsCount) allTargetsCount.textContent = targets.length;
        allTargetsGrid.innerHTML = targets
          .map((t, idx) => {
            const color = HUD_PALETTE[idx % HUD_PALETTE.length];
            const short = t.exact_name.split('/')[0].trim();
            return `
            <div class="target-intel-card ${idx === 0 ? 'active' : ''}" onclick="selectTarget(${idx})">
              <div class="target-card-top">
                <span class="target-id-pill" style="background: ${color}22; color: ${color}; border: 1px solid ${color};">
                  TARGET #${t.id}
                </span>
                <span style="font-family: ui-monospace, monospace; font-size: 0.72rem; color: ${color}; font-weight: 700;">
                  ${t.confidence}% Match
                </span>
              </div>
              <div class="target-card-crop-row">
                ${t.crop_url ? `<img src="${t.crop_url}" class="target-card-crop" alt="Target ${t.id}" style="border-color: ${color};" />` : ''}
                <div class="target-card-meta">
                  <strong class="target-card-name">${short}</strong>
                  <span class="target-card-desig">${t.designation || 'OBJECT'}</span>
                  <div style="font-size: 0.72rem; color: var(--signal-lime); margin-top: 2px;">${t.zoom_factor || 'Standard View'}</div>
                </div>
              </div>
              <p class="target-card-role">${t.role || 'Service operations'}</p>
              <div class="target-card-footer">
                <span>${t.country}</span>
                <span style="color: ${color};">Click to Inspect &rarr;</span>
              </div>
            </div>
          `;
          })
          .join('');
      } else {
        allTargetsSection.classList.add('hidden');
      }
    }

    // Initialize first target
    selectTarget(0);

    // AI Observations Box
    const tacticalBox = document.querySelector('#tactical-analysis-box');
    const tacticalText = document.querySelector('#tactical-analysis-text');
    if (body.platform && body.platform.tactical_analysis) {
      tacticalText.textContent = body.platform.tactical_analysis;
      tacticalBox.classList.remove('hidden');
    } else {
      tacticalBox.classList.add('hidden');
    }

    // Heatmap
    if (body.heatmap_url) {
      heatmapImg.src = body.heatmap_url;
      toggleContainer.classList.remove('hidden');
    }

    // Other Likely Matches
    const runnerUps = (body.platform && body.platform.runner_ups) || [];
    const runnerList = document.querySelector('#runner-ups-list');
    if (runnerUps.length > 0) {
      runnerList.innerHTML = runnerUps
        .map(
          (item) => `
        <div class="runner-up-row">
          <div class="runner-up-info">
            <span class="runner-up-name">${item.name}</span>
            <span class="runner-up-country">${item.country}</span>
          </div>
          <div class="mini-bar">
            <div class="mini-fill" style="width: ${item.confidence}%"></div>
          </div>
          <span class="runner-up-pct">${item.confidence}%</span>
        </div>
      `
        )
        .join('');
    } else {
      runnerList.innerHTML = `
        <div style="font-size: 0.82rem; color: var(--muted); padding: 8px;">
          Highest-confidence singular match with clear decision margin.
        </div>
      `;
    }

    dropZone.classList.add('hidden');
    resultSection.classList.remove('hidden');
    if (mobileTabsBar) mobileTabsBar.classList.remove('hidden');
    setMobileView('photo');
  } catch (err) {
    showError(err.message);
  } finally {
    browseButton.textContent = 'Select Image';
    browseButton.disabled = false;
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove('hidden');
}

// Global hook for target selection
window.selectTarget = selectTarget;

// ==========================================
// IMAGE HISTORY & SCAN GALLERY
// ==========================================
const galleryBtn = document.querySelector('#gallery-btn');
const galleryModal = document.querySelector('#gallery-modal');
const galleryModalClose = document.querySelector('#gallery-modal-close');
const clearGalleryBtn = document.querySelector('#clear-gallery-btn');
const galleryGrid = document.querySelector('#gallery-grid');
const galleryCountBadge = document.querySelector('#gallery-count');
const galleryModalCount = document.querySelector('#gallery-modal-count');

function getGalleryHistory() {
  try {
    return JSON.parse(localStorage.getItem('ASTRA_GALLERY') || '[]');
  } catch (e) {
    return [];
  }
}

function updateGalleryBadge() {
  const history = getGalleryHistory();
  if (galleryCountBadge) galleryCountBadge.textContent = history.length;
  if (galleryModalCount) galleryModalCount.textContent = history.length;
}

function saveToGallery(file, data) {
  try {
    const history = getGalleryHistory();
    const reader = new FileReader();
    reader.onload = function(e) {
      const entry = {
        id: 'scan_' + Date.now(),
        filename: file.name,
        date: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        name: (data.platform && data.platform.name) || data.label || 'Identified Object',
        category: data.label || 'Defense Asset',
        confidence: data.confidence || 0,
        thumb: e.target.result,
        data: data,
      };
      history.unshift(entry);
      if (history.length > 30) history.pop();
      try {
        localStorage.setItem('ASTRA_GALLERY', JSON.stringify(history));
      } catch (err) {
        entry.thumb = '';
        localStorage.setItem('ASTRA_GALLERY', JSON.stringify(history));
      }
      updateGalleryBadge();
    };
    reader.readAsDataURL(file);
  } catch (e) {
    console.warn('Gallery save exception:', e);
  }
}

function renderGallery() {
  const history = getGalleryHistory();
  updateGalleryBadge();
  if (!galleryGrid) return;

  if (history.length === 0) {
    galleryGrid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-dim);">
        <p style="font-size: 1.5rem; margin-bottom: 8px;">🖼️</p>
        <p>No scans recorded yet. Upload photos to build your tactical intelligence gallery!</p>
      </div>
    `;
    return;
  }

  galleryGrid.innerHTML = history.map((item) => `
    <div class="gallery-card" onclick="reinspectScan('${item.id}')" title="Click to view dossier">
      <img src="${item.thumb || '/static/placeholder.png'}" class="gallery-card-thumb" alt="${item.name}" />
      <div class="gallery-card-body">
        <span class="gallery-card-title">${item.name}</span>
        <div class="gallery-card-meta">
          <span>${item.category}</span>
          <span style="color: var(--signal-lime); font-weight: 700;">${item.confidence}%</span>
        </div>
        <div style="font-size: 0.68rem; color: var(--muted); margin-top: 2px;">
          ${item.filename} &middot; ${item.date}
        </div>
      </div>
    </div>
  `).join('');
}

function reinspectScan(scanId) {
  const history = getGalleryHistory();
  const found = history.find(h => h.id === scanId);
  if (!found || !found.data) return;

  currentReconData = found.data;
  if (found.thumb) {
    previewImg.src = found.thumb;
  }
  if (found.data.annotated_image_url && annotatedImg) {
    annotatedImg.src = found.data.annotated_image_url;
    if (viewAnnotatedBtn) {
      viewAnnotatedBtn.classList.remove('hidden');
      viewAnnotatedBtn.click();
    }
  } else {
    viewOrigBtn.click();
  }

  selectTarget(0);
  document.querySelector('#category-badge').textContent = found.data.label;
  document.querySelector('#category-conf').textContent = `${found.data.confidence}% Confidence`;
  document.querySelector('#model-tag').textContent = found.data.model;

  if (galleryModal) galleryModal.classList.add('hidden');
  dropZone.classList.add('hidden');
  resultSection.classList.remove('hidden');
  setMobileView('photo');
}

if (galleryBtn) {
  galleryBtn.addEventListener('click', () => {
    if (galleryModal) {
      galleryModal.classList.remove('hidden');
      renderGallery();
    }
  });
}
if (galleryModalClose) {
  galleryModalClose.addEventListener('click', () => galleryModal.classList.add('hidden'));
}
if (galleryModal) {
  galleryModal.addEventListener('click', (e) => {
    if (e.target === galleryModal) galleryModal.classList.add('hidden');
  });
}
if (clearGalleryBtn) {
  clearGalleryBtn.addEventListener('click', () => {
    if (confirm('Clear all saved scan history?')) {
      localStorage.removeItem('ASTRA_GALLERY');
      renderGallery();
    }
  });
}

// ==========================================
// BATCH IMAGE PROCESSING
// ==========================================
const batchBtn = document.querySelector('#batch-btn');
const batchModal = document.querySelector('#batch-modal');
const batchModalClose = document.querySelector('#batch-modal-close');
const batchDropZone = document.querySelector('#batch-drop-zone');
const batchFileInput = document.querySelector('#batch-file-input');
const batchSelectBtn = document.querySelector('#batch-select-btn');
const batchProgressSec = document.querySelector('#batch-progress-section');
const batchStatusText = document.querySelector('#batch-status-text');
const batchCountText = document.querySelector('#batch-count-text');
const batchProgressFill = document.querySelector('#batch-progress-bar-fill');
const batchResultsContainer = document.querySelector('#batch-results-container');
const batchResultsTbody = document.querySelector('#batch-results-tbody');
const batchTotalDone = document.querySelector('#batch-total-done');
const batchExportCsvBtn = document.querySelector('#batch-export-csv');

let currentBatchResults = [];

if (batchBtn) {
  batchBtn.addEventListener('click', () => {
    if (batchModal) batchModal.classList.remove('hidden');
  });
}
if (batchModalClose) {
  batchModalClose.addEventListener('click', () => batchModal.classList.add('hidden'));
}
if (batchModal) {
  batchModal.addEventListener('click', (e) => {
    if (e.target === batchModal) batchModal.classList.add('hidden');
  });
}
if (batchSelectBtn && batchFileInput) {
  batchSelectBtn.addEventListener('click', () => batchFileInput.click());
}

if (batchFileInput) {
  batchFileInput.addEventListener('change', () => {
    if (batchFileInput.files && batchFileInput.files.length > 0) {
      handleBatchUpload(batchFileInput.files);
    }
  });
}

if (batchDropZone) {
  ['dragenter', 'dragover'].forEach(name => {
    batchDropZone.addEventListener(name, (e) => {
      e.preventDefault();
      batchDropZone.classList.add('dragging');
    });
  });
  ['dragleave', 'drop'].forEach(name => {
    batchDropZone.addEventListener(name, (e) => {
      e.preventDefault();
      batchDropZone.classList.remove('dragging');
    });
  });
  batchDropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleBatchUpload(e.dataTransfer.files);
    }
  });
}

async function handleBatchUpload(fileList) {
  const files = Array.from(fileList).filter(f => f.type.match(/^image\//));
  if (files.length === 0) {
    alert('Please select valid image files (JPG, PNG, WebP).');
    return;
  }
  if (files.length > 30) {
    alert('Batch mode is limited to 30 images per request. Processing first 30.');
    files.length = 30;
  }

  batchProgressSec.classList.remove('hidden');
  batchResultsContainer.classList.add('hidden');
  batchStatusText.textContent = `Uploading & evaluating ${files.length} images...`;
  batchCountText.textContent = `0 / ${files.length}`;
  batchProgressFill.style.width = '30%';

  const formData = new FormData();
  files.forEach(f => formData.append('images', f));

  try {
    const res = await fetch('/api/predict/batch', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Batch processing failed.');
    }
    const data = await res.json();
    currentBatchResults = data.results || [];

    batchProgressFill.style.width = '100%';
    batchStatusText.textContent = `Batch evaluation completed!`;
    batchCountText.textContent = `${data.successful} / ${data.total} Succeeded`;

    batchTotalDone.textContent = currentBatchResults.length;
    batchResultsTbody.innerHTML = currentBatchResults.map(r => `
      <tr>
        <td><strong>${r.filename}</strong></td>
        <td><span class="catalog-badge">${r.label || r.prediction || 'Unknown'}</span></td>
        <td><strong style="color: var(--signal-lime);">${r.confidence || 0}%</strong></td>
        <td>${r.platform || '—'} (${r.designation || 'TARGET'})</td>
        <td>${r.country || 'Global'}</td>
      </tr>
    `).join('');

    batchResultsContainer.classList.remove('hidden');
  } catch (err) {
    alert(`Batch error: ${err.message}`);
    batchStatusText.textContent = `Batch error: ${err.message}`;
  }
}

if (batchExportCsvBtn) {
  batchExportCsvBtn.addEventListener('click', () => {
    if (!currentBatchResults || currentBatchResults.length === 0) return;
    const header = ['Filename', 'Category', 'Confidence', 'Platform', 'Designation', 'Country', 'Status'];
    const rows = currentBatchResults.map(r => [
      `"${r.filename || ''}"`,
      `"${r.label || r.prediction || ''}"`,
      r.confidence || 0,
      `"${r.platform || ''}"`,
      `"${r.designation || ''}"`,
      `"${r.country || ''}"`,
      `"${r.status || ''}"`
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [header.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `astra_batch_eval_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
}

updateGalleryBadge();
window.reinspectScan = reinspectScan;

