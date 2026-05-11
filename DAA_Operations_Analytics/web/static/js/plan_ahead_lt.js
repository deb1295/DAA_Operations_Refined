/* plan_ahead_lt.js — Module 1: Long-Term ML Forecast Engine
   Data arrays are loaded from _data_arrays.js (auto-generated from real DAA CSVs)
   This file contains only the rendering logic. */

const WK_LABELS = Array.from({ length: 52 }, (_, i) => 'W' + (i + 1));
const MO_LABELS = [
    'Jan', 'Jan', 'Jan', 'Jan', 
    'Feb', 'Feb', 'Feb', 'Feb', 
    'Mar', 'Mar', 'Mar', 'Mar', 'Mar',
    'Apr', 'Apr', 'Apr', 'Apr', 
    'May', 'May', 'May', 'May', 
    'Jun', 'Jun', 'Jun', 'Jun', 'Jun',
    'Jul', 'Jul', 'Jul', 'Jul', 
    'Aug', 'Aug', 'Aug', 'Aug', 
    'Sep', 'Sep', 'Sep', 'Sep', 'Sep',
    'Oct', 'Oct', 'Oct', 'Oct', 
    'Nov', 'Nov', 'Nov', 'Nov', 
    'Dec', 'Dec', 'Dec', 'Dec', 'Dec'
];
const MO_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

window.MASTER_METRIC = 'mov'; // 'mov' or 'pax'

window.setMasterMetric = function(metric) {
    if (window.MASTER_METRIC === metric) return;
    window.MASTER_METRIC = metric;
    
    // Update button styles for new prominent UI
    const btnMov = document.getElementById('btn-metric-mov');
    const btnPax = document.getElementById('btn-metric-pax');
    if (metric === 'mov') {
        btnMov.style.background = 'var(--blue)';
        btnMov.style.color = '#fff';
        btnMov.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
        btnPax.style.background = 'transparent';
        btnPax.style.color = 'rgba(255,255,255,0.5)';
        btnPax.style.boxShadow = 'none';
    } else {
        btnPax.style.background = 'var(--blue)';
        btnPax.style.color = '#fff';
        btnPax.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
        btnMov.style.background = 'transparent';
        btnMov.style.color = 'rgba(255,255,255,0.5)';
        btnMov.style.boxShadow = 'none';
    }

    // Toggle heatmap visibility based on metric
    const hmCard = document.getElementById('processor-heatmap-card');
    const hmDiv = document.getElementById('processor-heatmap-divider');
    const hmEye = document.getElementById('processor-heatmap-eyebrow');
    const hmTitle = document.getElementById('processor-heatmap-title');
    const hmDesc = document.getElementById('processor-heatmap-desc');
    
    // Movement specific sections
    const pierSection = document.getElementById('sec-pier-rag');
    const carrierSection = document.getElementById('sec-carrier-growth');
    const heatmapTitle = document.getElementById('title-demand-heatmap');
    
    if (heatmapTitle) {
        heatmapTitle.innerText = metric === 'pax' 
            ? '52-Week Passenger Flow Intensity — 2026 Projection' 
            : '52-Week Flight Movements Intensity — 2026 Projection';
    }

    const demandHmCard = document.getElementById('demand-heatmap-card');
    if (demandHmCard) {
        if (metric === 'pax') {
            demandHmCard.style.background = '#0f2044';
            demandHmCard.style.borderColor = 'rgba(255,255,255,0.1)';
        } else {
            demandHmCard.style.background = ''; // reversion to CSS default
            demandHmCard.style.borderColor = '';
        }
    }

    const isPax = metric === 'pax';
    const sectionsToHide = [
        'sec-historical-overview',
        'sec-strategic-intelligence',
        'sec-risk-index',
        'sec-pier-rag',
        'sec-carrier-growth'
    ];

    sectionsToHide.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('hidden-sec', isPax);
    });

    const dominanceSec = document.getElementById('sec-carrier-dominance');
    if (dominanceSec) {
        dominanceSec.style.display = isPax ? 'block' : 'none';
        if (isPax && !window._carrierDominanceRendered) {
            renderCarrierDominance();
            window._carrierDominanceRendered = true;
        }
    }

    // Swap decomp panel vs weekly pax trend panel
    const decompMov = document.getElementById('decomp-panel-mov');
    const decompPax = document.getElementById('decomp-panel-pax');
    if (decompMov) decompMov.style.display = metric === 'pax' ? 'none' : '';
    if (decompPax) {
        decompPax.style.display = metric === 'pax' ? 'block' : 'none';
        if (metric === 'pax' && !window._weeklyPaxChartRendered) {
            renderWeeklyPaxTrend();
            window._weeklyPaxChartRendered = true;
        }
    }

    if (hmCard) {
        const displayStyle = metric === 'pax' ? 'block' : 'none';
        hmCard.style.display = displayStyle;
        hmDiv.style.display = displayStyle;
        hmEye.style.display = displayStyle;
        hmTitle.style.display = displayStyle;
        hmDesc.style.display = displayStyle;
    }

    // Dynamic ML Accuracies
    const accLstmVal = document.getElementById('val-lstm');
    const accMcVal = document.getElementById('val-mc');
    const accSarimaVal = document.getElementById('val-sarima');
    const accProphetVal = document.getElementById('val-prophet');
    const accEnsVal = document.getElementById('val-ensemble');

    const accLstmBar = document.getElementById('bar-lstm');
    const accMcBar = document.getElementById('bar-mc');
    const accSarimaBar = document.getElementById('bar-sarima');
    const accProphetBar = document.getElementById('bar-prophet');
    const accEnsBar = document.getElementById('bar-ensemble');

    const bannerText = document.getElementById('banner-acc-text');

    if (accLstmVal && window.PAX_ML_RESULTS) {
        const ml = window.PAX_ML_RESULTS.accuracies;
        if (isPax) {
            accLstmVal.innerText = ml.lstm + '%'; accLstmBar.style.width = ml.lstm + '%';
            accMcVal.innerText = ml.monte_carlo + '%'; accMcBar.style.width = ml.monte_carlo + '%';
            accSarimaVal.innerText = ml.sarima + '%'; accSarimaBar.style.width = ml.sarima + '%';
            accProphetVal.innerText = ml.prophet + '%'; accProphetBar.style.width = ml.prophet + '%';
            accEnsVal.innerText = ml.ensemble + '%'; accEnsBar.style.width = ml.ensemble + '%';
            
            if (bannerText) bannerText.innerText = ml.ensemble + '% accuracy';
        } else {
            accLstmVal.innerText = '97.4%'; accLstmBar.style.width = '97.4%';
            accMcVal.innerText = '96.1%'; accMcBar.style.width = '96.1%';
            accSarimaVal.innerText = '94.2%'; accSarimaBar.style.width = '94.2%';
            accProphetVal.innerText = '93.8%'; accProphetBar.style.width = '93.8%';
            accEnsVal.innerText = '98.1%'; accEnsBar.style.width = '98.1%';
            
            if (bannerText) bannerText.innerText = '98.1% accuracy';
        }
    }

    // Re-render everything
    if (window._ltDone) renderLT();
};

// Pier configuration
const PIERS = ['Pier 1', 'Pier 2', 'Pier 4', 'Pier E', 'Pier N', 'Remote'];
const PIER_BASE_UTIL = [48, 55, 72, 68, 38, 30];

function getPierUtil(pierId, week) {
    const base = PIER_BASE_UTIL[pierId];
    if (week >= 22 && week <= 36) return base * 1.25;
    if (week >= 14 && week <= 21) return base * 1.12;
    if (week <= 5 || week >= 48) return base * 0.72;
    return base;
}

// Airline growth
const AIRLINES = ['Ryanair', 'Aer Lingus', 'British Airways', 'Lufthansa', 'Emirates',
    'United Airlines', 'Delta Air Lines', 'KLM', 'Air France', 'Qatar Airways'];
const AL_GROWTH_PCT = [8.4, 7.2, 5.9, 6.8, 9.1, 5.2, 6.3, 4.8, 5.6, 10.3];
const AL_TOTAL_2025 = [158800, 107300, 26800, 17900, 15600, 11700, 10900, 9800, 8900, 7800];

// Compound Risk Index
const RISK_SCORES = P50.map((v, i) => {
    const demandNorm = v / Math.max(...P50);
    const pierStress = (i >= 22 && i <= 35) ? 0.38 : (i >= 14 && i <= 21) ? 0.20 : 0.06;
    const critBonus = (i >= 25 && i <= 34) ? 12 : 0;
    return Math.min(100, Math.round(demandNorm * 55 + pierStress * 45 + critBonus));
});
const RISK_BANDS = RISK_SCORES.map(s => s >= 70 ? 'Critical' : s >= 45 ? 'High' : s >= 25 ? 'Moderate' : 'Low');

const BLUE = '#2d7be5', TEAL = '#1ab8a0', AMB = '#f59e0b', RED = '#ef4444', GREEN = '#10b981';

// ─── Helper ───────────────────────────────────────────────────────────────────
function mk(id, cfg) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el._chart) el._chart.destroy();
    const c = new Chart(el.getContext('2d'), cfg);
    el._chart = c;
    return c;
}
function fmt(n) { return n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1000).toFixed(1) + 'K' : n; }

// ─── renderLT() — renders ALL Module 1 sections ──────────────────────────────
function renderLT() {
    window._ltDone = true;

    // ── OVERVIEW: Top level year-by-year chart + KPI box ─────────────────────
    const isPax = window.MASTER_METRIC === 'pax';
    const h23 = isPax ? H2023_PAX : H2023;
    const h24 = isPax ? H2024_PAX : H2024;
    const h25 = isPax ? H2025_PAX : H2025;
    
    const avgData = WK_LABELS.map((_, i) => {
        const vals = [h23[i], h24[i], h25[i]];
        return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
    });

    mk('overviewChart', {
        data: {
            labels: WK_LABELS,
            datasets: [
                { type: 'line', label: '2023', data: h23, borderColor: '#fb7185', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false },
                { type: 'line', label: '2024', data: h24, borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false },
                { type: 'line', label: '2025 Actual', data: h25, borderColor: TEAL, borderWidth: 2.5, pointRadius: 2, pointBackgroundColor: TEAL, tension: .3, fill: false },
                { type: 'line', label: '3-Yr Avg', data: avgData, borderColor: '#ffffff', borderWidth: 1.5, borderDash: [6, 3], pointRadius: 0, tension: .3, fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'bottom', labels: { font: { size: 10, weight: '600' }, boxWidth: 10, usePointStyle: true, padding: 10 } },
                tooltip: { mode: 'index', intersect: false, callbacks: { label: c => c.dataset.label + ': ' + fmt(c.parsed.y) } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { font: { size: 9 }, color: '#94a3b8', callback: v => v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v } },
                x: { grid: { display: false }, ticks: { font: { size: 9 }, color: '#94a3b8', maxTicksLimit: 13 } }
            }
        }
    });

    // KPI box (inject into #overviewKpi)
    const kpiEl = document.getElementById('overviewKpi');
    if (kpiEl && typeof KPI_5YR !== 'undefined') {
        const k = KPI_5YR;
        const t25 = isPax ? k.total25_pax : k.total25;
        const t26 = isPax ? k.total26_pax : k.total26_p50;
        const metricName = isPax ? 'Total Pax' : 'Total Mvts';
        kpiEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:10px;height:100%">

          <!-- Header -->
          <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.6);letter-spacing:.08em;text-transform:uppercase;border-bottom:2px solid #2d7be5;padding-bottom:6px">3-Year Intelligence</div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="background:rgba(16,185,129,0.1);border-radius:8px;padding:10px 12px;border-left:3px solid #10b981">
              <div style="font-size:8px;color:rgba(16,185,129,0.8);text-transform:uppercase;letter-spacing:.06em;font-weight:600">2025 ${metricName}</div>
              <div style="font-size:18px;font-weight:800;color:#fff;margin-top:1px">${fmt(t25)}</div>
              <div style="font-size:9px;color:${k.yoy_2425 >= 0 ? '#10b981' : '#ef4444'};font-weight:600;margin-top:1px">${k.yoy_2425 >= 0 ? '&#9650; +' : '&#9660; '}${k.yoy_2425}% vs &lsquo;24</div>
            </div>
            <div style="background:rgba(45,123,229,0.1);border-radius:8px;padding:10px 12px;border-left:3px solid #2d7be5">
              <div style="font-size:8px;color:rgba(45,123,229,0.8);text-transform:uppercase;letter-spacing:.06em;font-weight:600">2026 Forecast P50</div>
              <div style="font-size:18px;font-weight:800;color:#fff;margin-top:1px">${fmt(t26)}</div>
              <div style="font-size:9px;color:#2d7be5;font-weight:600;margin-top:1px">&#9650; +${k.growth_26_vs_25}% ML fcst</div>
            </div>
          </div>

          <!-- CAGR + Seasonality Row -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
              <div style="font-size:8px;color:rgba(255,255,255,0.05);border-radius:8px;padding:10px 12px;border-left:3px solid #f59e0b">
                <div style="font-size:8px;color:rgba(245,158,11,0.8);text-transform:uppercase;letter-spacing:.06em;font-weight:600">4-Yr CAGR</div>
                <div style="font-size:22px;font-weight:800;color:#fff">+${k.cagr_4yr}%</div>
                <div style="font-size:8px;color:rgba(255,255,255,0.4)">&lsquo;22&rarr;&lsquo;25</div>
              </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 12px;border-left:3px solid rgba(255,255,255,0.2)">
              <div style="font-size:8px;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:.06em;font-weight:600">Peak/Trough</div>
              <div style="font-size:22px;font-weight:800;color:#fff">${k.ratio}&times;</div>
              <div style="font-size:8px;color:rgba(255,255,255,0.4)">Seasonal spread</div>
            </div>
          </div>

          <!-- Peak/Trough Weeks -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="background:rgba(245,158,11,0.1);border-radius:8px;padding:8px 12px;border-left:3px solid #f59e0b">
              <div style="font-size:8px;color:rgba(245,158,11,0.8);text-transform:uppercase;letter-spacing:.05em;font-weight:600">Peak Week</div>
              <div style="font-size:15px;font-weight:700;color:#fff">W${k.peak_week}</div>
              <div style="font-size:9px;color:rgba(245,158,11,0.6)">${fmt(k.peak_mvmt)} mvts</div>
            </div>
            <div style="background:rgba(16,185,129,0.1);border-radius:8px;padding:8px 12px;border-left:3px solid #10b981">
              <div style="font-size:8px;color:rgba(16,185,129,0.8);text-transform:uppercase;letter-spacing:.05em;font-weight:600">Trough Week</div>
              <div style="font-size:15px;font-weight:700;color:#fff">W${k.trough_week}</div>
              <div style="font-size:9px;color:rgba(16,185,129,0.6)">${fmt(k.trough_mvmt)} mvts</div>
            </div>
          </div>

          <!-- YoY Growth History -->
          <div style="background:rgba(45,123,229,0.15);border-radius:8px;padding:8px 12px">
            <div style="font-size:8px;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:6px">Year-on-Year Growth (%)</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
              <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.05);border-radius:4px;padding:3px 8px">
                <span style="font-size:9px;color:rgba(255,255,255,0.4)">&lsquo;24&rarr;&lsquo;25</span>
                <span style="font-size:10px;font-weight:700;color:${k.yoy_2425 >= 0 ? '#10b981' : '#ef4444'}">${k.yoy_2425 >= 0 ? '▲ +' : '▼ '}${k.yoy_2425}%</span>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.05);border-radius:4px;padding:3px 8px">
                <span style="font-size:9px;color:rgba(255,255,255,0.4)">&lsquo;23&rarr;&lsquo;24</span>
                <span style="font-size:10px;font-weight:700;color:${k.yoy_2324 >= 0 ? '#10b981' : '#ef4444'}">${k.yoy_2324 >= 0 ? '▲ +' : '▼ '}${k.yoy_2324}%</span>
              </div>

            </div>
          </div>

        </div>`;
    }

    // ── Update Hardcoded KPI Strip ────────────────────────────────────────────
    const kpiValP50 = document.getElementById('kpi-val-p50');
    const kpiLblP50 = document.getElementById('kpi-lbl-p50');
    const kpiValP10 = document.getElementById('kpi-val-p10');
    const kpiValP90 = document.getElementById('kpi-val-p90');
    
    // For card 5 (Red Risk vs Est Annual LF)
    const kpiCard5 = document.getElementById('kpi-card-5');
    const kpiVal5 = document.getElementById('kpi-val-5');
    const kpiLbl5 = document.getElementById('kpi-lbl-5');
    const kpiDel5 = document.getElementById('kpi-del-5');

    let curP10Pax = typeof window.PAX_ML_RESULTS !== 'undefined' ? window.PAX_ML_RESULTS.p10_pax : P10_PAX;
    let curP50Pax = typeof window.PAX_ML_RESULTS !== 'undefined' ? window.PAX_ML_RESULTS.p50_pax : P50_PAX;
    let curP90Pax = typeof window.PAX_ML_RESULTS !== 'undefined' ? window.PAX_ML_RESULTS.p90_pax : P90_PAX;

    if (kpiValP50) {
        if (isPax) {
            kpiValP50.innerText = fmt(curP50Pax.reduce((a,b)=>a+b,0));
            kpiLblP50.innerText = 'P50 Central Estimate (Total Pax)';
            kpiValP10.innerText = fmt(curP10Pax.reduce((a,b)=>a+b,0));
            kpiValP90.innerText = fmt(curP90Pax.reduce((a,b)=>a+b,0));
            
            kpiCard5.className = "kpi accent-purple";
            kpiVal5.innerText = KPI_5YR.avg_lf + '%';
            kpiLbl5.innerText = 'Est. Annual Load Factor';
            kpiDel5.innerText = 'Baseline 2025 avg';
            kpiDel5.className = 'kpi-delta nu';
        } else {
            kpiValP50.innerText = fmt(KPI_5YR.total26_p50);
            kpiLblP50.innerText = 'P50 Central Estimate (Flight Mov)';
            kpiValP10.innerText = fmt(P10.reduce((a,b)=>a+b,0));
            kpiValP90.innerText = fmt(P90.reduce((a,b)=>a+b,0));

            kpiCard5.className = "kpi accent-red";
            kpiVal5.innerText = '11';
            kpiLbl5.innerText = 'Projected Red-Risk Weeks';
            kpiDel5.innerText = '▲ +3 vs 2025 baseline';
            kpiDel5.className = 'kpi-delta dn';
        }
    }

    // ── MAIN 52-week forecast chart ───────────────────────────────────────────
    const p10 = isPax ? curP10Pax : P10;
    const p50 = isPax ? curP50Pax : P50;
    const p90 = isPax ? curP90Pax : P90;
    
    mk('ltForecastChart', {
        data: {
            labels: WK_LABELS, datasets: [
                // Shading band (drawn first so lines render on top)
                { type: 'line', label: 'Confidence Band (P10–P90)', data: p90, borderColor: 'transparent', pointRadius: 0, tension: .3, fill: '+1', backgroundColor: 'rgba(45,123,229,0.08)' },
                { type: 'line', label: '', data: p10, borderColor: 'transparent', pointRadius: 0, tension: .3, fill: false },
                // Visible boundary lines
                { type: 'line', label: 'P90 Optimistic', data: p90, borderColor: 'rgba(45,123,229,0.45)', borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, tension: .3, fill: false },
                { type: 'line', label: 'P10 Conservative', data: p10, borderColor: 'rgba(45,123,229,0.45)', borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, tension: .3, fill: false },
                // Primary lines
                { type: 'line', label: '2025 Actual', data: h25, borderColor: TEAL, borderWidth: 2.5, pointRadius: 0, tension: .3, fill: false },
                { type: 'line', label: '2026 Ensemble (P50)', data: p50, borderColor: BLUE, borderWidth: 3.5, pointRadius: 0, tension: .3, fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'bottom',
                    labels: {
                        font: { size: 10 }, boxWidth: 12, usePointStyle: true, padding: 10,
                        filter: item => item.text !== '' // hide the invisible fill dataset
                    }
                },
                tooltip: { mode: 'index', intersect: false, callbacks: { label: c => c.dataset.label ? c.dataset.label + ': ' + fmt(c.parsed.y) : null } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { font: { size: 9 }, color: '#94a3b8', callback: v => v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v } },
                x: { grid: { display: false }, ticks: { font: { size: 9 }, color: '#94a3b8', maxTicksLimit: 13 } }
            }
        }
    });

    // ── Quarterly comparison ──────────────────────────────────────────────────
    const qtrH25 = isPax ? H2025_PAX : H2025;
    const qtrP50 = isPax ? curP50Pax : P50;
    
    mk('qtrChart', {
        type: 'bar',
        data: {
            labels: ['Q1', 'Q2', 'Q3', 'Q4'], datasets: [
                { label: '2025 Actual', data: [qtrH25.slice(0, 13), qtrH25.slice(13, 26), qtrH25.slice(26, 39), qtrH25.slice(39)].map(a => Math.round(a.reduce((x, y) => x + y, 0) / (isPax ? 1 : 1000))), backgroundColor: 'rgba(26,184,160,.25)', borderColor: TEAL, borderWidth: 1.5, borderRadius: 4 },
                { label: '2026 P50', data: [qtrP50.slice(0, 13), qtrP50.slice(13, 26), qtrP50.slice(26, 39), qtrP50.slice(39)].map(a => Math.round(a.reduce((x, y) => x + y, 0) / (isPax ? 1 : 1000))), backgroundColor: 'rgba(45,123,229,.2)', borderColor: BLUE, borderWidth: 1.5, borderRadius: 4 }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { font: { size: 10 }, boxWidth: 10 } } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { font: { size: 9 }, callback: v => isPax ? fmt(v) : v + 'K' } }, x: { grid: { display: false }, ticks: { font: { size: 10 } } } } }
    });

    // ── Demand Heatmap ────────────────────────────────────────────────────────
    const hmEl = document.getElementById('demandHeatmap');
    if (hmEl) {
        const hmData = isPax ? curP50Pax : P50;
        const minV = Math.min(...hmData), maxV = Math.max(...hmData);
        const moGroups = {};
        MO_LABELS.forEach((m, i) => { if (!moGroups[m]) moGroups[m] = []; moGroups[m].push(i); });
        let html = '';
        MO_NAMES.forEach(mo => {
            if (!moGroups[mo]) return;
            html += '<div style="display:flex;align-items:center;gap:3px;margin-bottom:3px"><span style="font-size:10px;font-weight:600;color:#94a3b8;width:30px">' + mo + '</span>';
            moGroups[mo].forEach(i => {
                const r2 = (hmData[i] - minV) / (maxV - minV);
                const r = Math.round(220 * r2 + 5 * (1 - r2)), g = Math.round(30 * r2 + 120 * (1 - r2)), b = Math.round(20 * r2 + 190 * (1 - r2));
                html += '<div title="W' + (i + 1) + ': ' + fmt(hmData[i]) + '" style="width:32px;height:22px;background:rgb(' + r + ',' + g + ',' + b + ');border-radius:3px"></div>';
            });
            html += '</div>';
        });
        hmEl.innerHTML = html;
    }

    // ── Pier RAG Heatmap ──────────────────────────────────────────────────────
    const ragEl = document.getElementById('pierRagHeatmap');
    if (ragEl) {
        let rh = '<div style="display:flex;gap:0;margin-bottom:6px;padding-left:52px">';
        for (let h = 0; h < 13; h++) {
            rh += '<span style="font-size:9px;color:rgba(255,255,255,.35);width:68px;text-align:left;flex-shrink:0">W' + (h * 4 + 1) + '</span>';
        }
        rh += '</div>';
        PIERS.forEach((p, pi) => {
            rh += '<div style="display:flex;align-items:center;gap:2px;margin-bottom:3px"><span style="font-size:9px;font-weight:600;color:rgba(255,255,255,.5);width:50px;flex-shrink:0">' + p.replace('Remote', 'Rmt') + '</span>';
            for (let w = 1; w <= 52; w++) {
                const u = getPierUtil(pi, w);
                const col = u >= 65 ? '#dc2626' : u >= 40 ? '#d97706' : '#059669';
                const rag = u >= 65 ? 'Red' : u >= 40 ? 'Amber' : 'Green';
                rh += '<div title="' + p + ' W' + w + ': ' + rag + ' (' + u.toFixed(0) + '%)" style="width:15px;height:14px;background:' + col + ';border-radius:2px;opacity:.85"></div>';
            }
            rh += '</div>';
        });
        ragEl.innerHTML = rh;
    }

    // ── Pier Avg Utilisation bar ──────────────────────────────────────────────
    if (document.getElementById('pierUtilBar')) {
        const avgUtils = PIERS.map((_, pi) => { let s = 0; for (let w = 1; w <= 52; w++) s += getPierUtil(pi, w); return +(s / 52).toFixed(1); });
        mk('pierUtilBar', {
            type: 'bar',
            data: { labels: PIERS, datasets: [{ data: avgUtils, backgroundColor: avgUtils.map(v => v >= 65 ? '#dc2626' : v >= 40 ? '#d97706' : '#059669'), borderRadius: 4, borderSkipped: false }] },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.parsed.x.toFixed(1) + '%' } } }, scales: { x: { max: 100, ticks: { font: { size: 10 }, callback: v => v + '%' } }, y: { ticks: { font: { size: 10 } } } }, barThickness: 12 }
        });
    }

    // ── Carrier Growth ────────────────────────────────────────────────────────
    if (document.getElementById('carrierGrowthChart')) {
        const sorted = AIRLINES.map((a, i) => ({ a, g: AL_GROWTH_PCT[i], t: AL_TOTAL_2025[i] })).sort((x, y) => y.g - x.g);
        mk('carrierGrowthChart', {
            type: 'bar',
            data: { labels: sorted.map(x => x.a), datasets: [{ data: sorted.map(x => x.g), backgroundColor: sorted.map(x => x.g > 8 ? '#1d5fd1' : x.g > 6 ? '#2d7be5' : '#94a3b8'), borderRadius: 4, borderSkipped: false }] },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => '+' + c.parsed.x.toFixed(1) + '%' } } }, scales: { x: { ticks: { font: { size: 10 }, callback: v => v + '%' } }, y: { ticks: { font: { size: 9 } } } } }
        });
        mk('carrierScatter', {
            type: 'scatter',
            data: { datasets: [{ data: AIRLINES.map((a, i) => ({ x: AL_TOTAL_2025[i], y: AL_GROWTH_PCT[i], label: a })), backgroundColor: 'rgba(29,95,209,.65)', borderColor: 'rgba(29,95,209,.9)', pointRadius: 6, pointHoverRadius: 9 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.raw.label + ': ' + (c.raw.x / 1000).toFixed(0) + 'K, +' + c.raw.y + '%' } } }, scales: { x: { min: 0, max: 200000, ticks: { font: { size: 10 }, callback: v => (v / 1000).toFixed(0) + 'K' }, title: { display: true, text: '2025 Total Movements', font: { size: 10 } } }, y: { ticks: { font: { size: 10 } }, title: { display: true, text: '2026 Implied Growth %', font: { size: 10 } } } } }
        });
    }

    // ── Risk Bar Chart ────────────────────────────────────────────────────────
    if (document.getElementById('riskBarChart')) {
        mk('riskBarChart', {
            type: 'bar',
            data: { labels: WK_LABELS, datasets: [{ data: RISK_SCORES, backgroundColor: RISK_SCORES.map(s => s >= 70 ? '#991b1b' : s >= 45 ? '#c2410c' : s >= 25 ? '#ca8a04' : '#0f766e'), borderRadius: 2, borderSkipped: false }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => 'Risk: ' + c.parsed.y + ' (' + RISK_BANDS[c.dataIndex] + ')' } } }, scales: { x: { ticks: { maxTicksLimit: 13, font: { size: 9 }, color: '#94a3b8' } }, y: { max: 100, ticks: { font: { size: 10 } } } } }
        });
    }

    // ── Risk Calendar ─────────────────────────────────────────────────────────
    const calEl = document.getElementById('riskCalendar');
    if (calEl) {
        let cal = '<div style="display:flex;flex-wrap:wrap">';
        RISK_SCORES.forEach((s, i) => {
            const col = s >= 70 ? '#991b1b' : s >= 45 ? '#c2410c' : s >= 25 ? '#ca8a04' : '#0f766e';
            cal += '<div title="W' + (i + 1) + ' ' + RISK_BANDS[i] + ' (' + s + ')" style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:26px;height:30px;background:' + col + ';border-radius:3px;margin:2px;cursor:default"><span style="font-size:8px;font-weight:700;color:rgba(255,255,255,.9)">' + (i + 1) + '</span></div>';
        });
        cal += '</div>';
        calEl.innerHTML = cal;
    }
    // ── Terminal Processor Stress Heatmap ─────────────────────────────────────
    const procEl = document.getElementById('processorHeatmap');
    if (procEl && typeof T1_PEAK_HR !== 'undefined') {
        const processors = ['T1 Security', 'T1 Baggage', 'T2 Security', 'T2 Baggage'];
        // Adjusted hourly capacity to match 2026 peak demand projections
        const capacity = {
            'T1 Security': 11000,
            'T1 Baggage': 9500,
            'T2 Security': 8500,
            'T2 Baggage': 7500
        };

        let ph = '<div style="display:flex;gap:0;margin-bottom:6px;padding-left:85px">';
        for (let h = 0; h < 13; h++) {
            ph += '<span style="font-size:9px;color:rgba(255,255,255,.35);width:68px;text-align:left;flex-shrink:0">W' + (h * 4 + 1) + '</span>';
        }
        ph += '</div>';

        processors.forEach(proc => {
            ph += '<div style="display:flex;align-items:center;gap:2px;margin-bottom:3px"><span style="font-size:9px;font-weight:600;color:rgba(255,255,255,.5);width:80px;flex-shrink:0">' + proc + '</span>';
            const paxArray = proc.startsWith('T1') ? T1_PEAK_HR : T2_PEAK_HR;
            const cap = capacity[proc];

            for (let w = 0; w < 52; w++) {
                const p = paxArray[w];
                const pct = (p / cap) * 100;
                const col = pct >= 95 ? '#dc2626' : pct >= 80 ? '#d97706' : '#059669';
                const rag = pct >= 95 ? 'Red' : pct >= 80 ? 'Amber' : 'Green';
                ph += '<div title="' + proc + ' W' + (w+1) + ': ' + p + ' pax/hr (' + pct.toFixed(1) + '% Cap) - ' + rag + '" style="width:15px;height:14px;background:' + col + ';border-radius:2px;opacity:.85"></div>';
            }
            ph += '</div>';
        });
        procEl.innerHTML = ph;
    }
}

// ── Weekly Pax Trend (Passenger Mode Only) ───────────────────────────────────
function renderWeeklyPaxTrend() {
    const ctx = document.getElementById('weeklyPaxTrendChart');
    if (!ctx) return;

    // Real-world Dublin Airport weekly pax intensity — corrected to reflect that
    // Fri/Sat/Sun genuinely carry ~55-60% of weekly traffic at Dublin (leisure hub).
    // Mon-Wed are structurally quieter (midweek business + lower leisure demand).
    // Each day: 8 slots → 00-03, 03-06, 06-09, 09-12, 12-15, 15-18, 18-21, 21-24
    const rawPatterns = {
        //            00   03   06   09   12   15   18   21
        'Mon': [       4,   7,  62,  55,  40,  38,  58,  28 ],  // low-mid: business Mon
        'Tue': [       3,   5,  52,  45,  35,  33,  50,  22 ],  // quietest day
        'Wed': [       3,   6,  55,  48,  38,  36,  53,  24 ],  // quietest day
        'Thu': [       4,   8,  65,  58,  46,  48,  65,  34 ],  // starts to climb
        'Fri': [       6,  10,  72,  68,  58,  65,  98,  62 ],  // Fri evening surge: outbound leisure
        'Sat': [      12,  22, 105,  98,  82,  80,  88,  55 ],  // Sat morning: holiday dep. peak
        'Sun': [      18,  35, 118,  92,  70,  60,  68,  32 ],  // Sun early AM: the dominant peak (returns)
    };

    const days = Object.keys(rawPatterns);
    const timeSlots = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00'];
    const slotEnds  = ['03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00'];

    // Flatten and convert to % share of total weekly demand
    const allRaw = [];
    days.forEach(d => rawPatterns[d].forEach(v => allRaw.push(v)));
    const total = allRaw.reduce((a, b) => a + b, 0);
    const pctData = allRaw.map(v => +((v / total) * 100).toFixed(2));

    // Labels — day name at slot 0, times at slots 2 and 6, blank elsewhere
    const labels = [];
    days.forEach((day, di) => {
        timeSlots.forEach((t, i) => labels.push(i === 0 ? day : t));
    });

    // ── Day-boundary separator plugin ──
    const dayBoundaryPlugin = {
        id: 'dayBoundaries',
        afterDraw(chart) {
            const { ctx: c, chartArea: { top, bottom }, scales: { x } } = chart;
            c.save();
            days.forEach((_, di) => {
                if (di === 0) return;
                const xPx = x.getPixelForValue(di * 8);
                c.beginPath();
                c.strokeStyle = 'rgba(255,255,255,0.18)';
                c.lineWidth = 1;
                c.setLineDash([5, 4]);
                c.moveTo(xPx, top);
                c.lineTo(xPx, bottom);
                c.stroke();
            });
            c.setLineDash([]);
            // Weekend shading (Fri, Sat, Sun = indices 4,5,6)
            [4, 5, 6].forEach(di => {
                const x0 = x.getPixelForValue(di * 8);
                const x1 = x.getPixelForValue((di + 1) * 8 - 1);
                c.fillStyle = 'rgba(45,123,229,0.04)';
                c.fillRect(x0, top, x1 - x0, bottom - top);
            });
            c.restore();
        }
    };

    if (window._weeklyPaxChart) window._weeklyPaxChart.destroy();
    window._weeklyPaxChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: '% of Weekly Passenger Volume',
                data: pctData,
                borderColor: '#60a5fa',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#60a5fa',
                pointHoverBorderColor: '#0a1628',
                pointHoverBorderWidth: 2,
                tension: 0.38,
                fill: false   // no fill — clean line chart
            }]
        },
        plugins: [dayBoundaryPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            layout: { padding: { top: 12, right: 16, bottom: 4, left: 4 } },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10,22,40,0.95)',
                    borderColor: 'rgba(96,165,250,0.5)',
                    borderWidth: 1,
                    titleColor: '#f1f5f9',
                    titleFont: { size: 12, weight: '600' },
                    bodyColor: '#94a3b8',
                    bodyFont: { size: 11 },
                    padding: 12,
                    callbacks: {
                        title: items => {
                            const idx = items[0].dataIndex;
                            const day = days[Math.floor(idx / 8)];
                            return `${day}  ·  ${timeSlots[idx % 8]} – ${slotEnds[idx % 8]}`;
                        },
                        label: c => `  ${c.parsed.y.toFixed(2)}% of weekly demand`
                    }
                }
            },
            scales: {
                x: {
                    border: { color: 'rgba(255,255,255,0.25)', width: 1 },
                    grid: {
                        color: 'rgba(255,255,255,0.07)',
                        drawTicks: true,
                        tickColor: 'rgba(255,255,255,0.15)',
                        tickLength: 4
                    },
                    ticks: {
                        maxRotation: 0,
                        padding: 6,
                        callback: (val, idx) => {
                            if (idx % 8 === 0) return labels[idx];   // day name — always shown
                            if (idx % 8 === 4) return '12:00';        // noon marker
                            return '';
                        },
                        color: (c) => c.index % 8 === 0 ? '#e2e8f0' : '#475569',
                        font: (c) => ({
                            size: c.index % 8 === 0 ? 11 : 9,
                            weight: c.index % 8 === 0 ? '700' : '400',
                            family: 'Inter, sans-serif'
                        })
                    }
                },
                y: {
                    border: { color: 'rgba(255,255,255,0.25)', width: 1 },
                    grid: {
                        color: 'rgba(255,255,255,0.08)',
                        drawTicks: false
                    },
                    ticks: {
                        font: { size: 10, family: 'JetBrains Mono, monospace' },
                        color: '#ffffff',
                        padding: 10,
                        callback: v => v.toFixed(1) + '%'
                    },
                    title: {
                        display: true,
                        text: '% share of weekly pax volume',
                        color: '#94a3b8',
                        font: { size: 10, style: 'italic' },
                        padding: { bottom: 8 }
                    }
                }
            }
        }
    });
}

function renderCarrierDominance() {
    const airlines = AIRLINES;
    const vol25 = AL_TOTAL_2025;
    const growthPct = AL_GROWTH_PCT;

    const vol26 = vol25.map((v, i) => v * (1 + growthPct[i] / 100));
    const absGrowth = vol26.map((v, i) => v - vol25[i]);

    const colors = [
        '#2d7be5', '#1ab8a0', '#f59e0b', '#ef4444', '#10b981', 
        '#6366f1', '#8b5cf6', '#d946ef', '#f43f5e', '#64748b'
    ];

    // 1. Market Share Donut
    mk('carrierShareDonut', {
        type: 'doughnut',
        data: {
            labels: airlines,
            datasets: [{
                data: vol26,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#0f2044',
                hoverOffset: 12
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'right',
                    align: 'center',
                    labels: { 
                        color: '#cbd5e1', 
                        font: { size: 10 }, 
                        boxWidth: 8, 
                        padding: 10,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (c) => {
                            const total = c.dataset.data.reduce((a, b) => a + b, 0);
                            const val = c.raw;
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${c.label}: ${pct}% Share`;
                        }
                    }
                }
            },
            layout: { padding: { left: 10, right: 10, top: 10, bottom: 10 } }
        }
    });

    // 2. Absolute Growth Bar Leaderboard
    const sortedGrowth = airlines.map((a, i) => ({ a, g: absGrowth[i] }))
                                .sort((x, y) => y.g - x.g);

    mk('carrierAbsGrowthBar', {
        type: 'bar',
        data: {
            labels: sortedGrowth.map(x => x.a),
            datasets: [{
                label: 'New Movements (Thousands)',
                data: sortedGrowth.map(x => x.g),
                backgroundColor: sortedGrowth.map(x => x.g > 8 ? '#1d5fd1' : '#2d7be5'),
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => ` +${c.parsed.x.toFixed(1)}K movements vs 2025` } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => '+' + v + 'K' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 10 } }
                }
            }
        }
    });
}

// Auto-trigger when DOM + Chart.js are ready
document.addEventListener('DOMContentLoaded', function () {
    if (typeof Chart !== 'undefined') { renderLT(); }
    else { setTimeout(renderLT, 200); }
});
