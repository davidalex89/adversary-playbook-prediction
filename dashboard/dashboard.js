(async function() {
    const BASE = "data/";

    async function loadJSON(file) {
        const resp = await fetch(BASE + file);
        return resp.json();
    }

    const [jaccardData, liftData, groupClusters, techClusters, nlpStats] = await Promise.all([
        loadJSON("cooccurrence_top10_jaccard.json"),
        loadJSON("cooccurrence_top10_lift.json"),
        loadJSON("group_clusters.json"),
        loadJSON("technique_clusters.json"),
        loadJSON("nlp_correlation_stats.json").catch(() => null),
    ]);

    const techniques = Object.keys(jaccardData).sort();
    const nGroups = groupClusters.points.length;

    document.getElementById("stat-techniques").textContent = techniques.length.toLocaleString();
    document.getElementById("stat-groups").textContent = nGroups.toLocaleString();
    document.getElementById("stat-pairs").textContent = "4,546";
    document.getElementById("stat-avg-ttps").textContent = "26.1";
    document.getElementById("stat-group-clusters").textContent = groupClusters.k;
    document.getElementById("stat-tech-clusters").textContent = techClusters.k;

    // --- Co-occurrence Predictor ---
    const select = document.getElementById("tech-select");
    select.innerHTML = '<option value="">— Select a technique —</option>';
    techniques.forEach(tid => {
        const items = jaccardData[tid];
        const name = items && items.length > 0 ? items[0]?.name || "" : "";
        const opt = document.createElement("option");
        opt.value = tid;
        opt.textContent = tid;
        select.appendChild(opt);
    });

    let currentMethod = "jaccard";
    let coChart = null;

    function renderCooccurrence(techId) {
        const data = currentMethod === "jaccard" ? jaccardData[techId] : liftData[techId];
        if (!data || data.length === 0) return;

        const labels = data.map(d => `${d.id} ${d.name}`);
        const values = data.map(d => d.score);

        const ctx = document.getElementById("cooccurrence-chart").getContext("2d");
        if (coChart) coChart.destroy();

        coChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: "rgba(34, 211, 238, 0.5)",
                    borderColor: "rgba(6, 182, 212, 1)",
                    borderWidth: 1,
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: `Top co-occurring techniques for ${techId} (${currentMethod})`,
                        color: "#e6edf3",
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => items[0]?.label || "",
                            label: (ctx) => ` Score: ${ctx.raw.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
                    y: {
                        ticks: {
                            color: "#e6edf3",
                            font: { size: 10 },
                            callback: function(value) {
                                const label = this.getLabelForValue(value);
                                return label.length > 40 ? label.substring(0, 38) + "…" : label;
                            }
                        },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    select.addEventListener("change", () => {
        if (select.value) renderCooccurrence(select.value);
    });

    document.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentMethod = btn.dataset.method;
            if (select.value) renderCooccurrence(select.value);
        });
    });

    // --- Cluster Scatter Plots ---
    const clusterColors = [
        "rgba(34, 211, 238, 0.7)",
        "rgba(248, 113, 113, 0.7)",
        "rgba(52, 211, 153, 0.7)",
        "rgba(167, 139, 250, 0.7)",
        "rgba(251, 146, 60, 0.7)",
        "rgba(236, 72, 153, 0.7)",
        "rgba(96, 165, 250, 0.7)",
        "rgba(163, 230, 53, 0.7)",
    ];

    function renderScatter(canvasId, clusterData, title) {
        const datasets = [];
        for (let c = 0; c < clusterData.k; c++) {
            const pts = clusterData.points.filter(p => p.cluster === c);
            datasets.push({
                label: `Cluster ${c}`,
                data: pts.map(p => ({ x: p.x, y: p.y, label: p.name || p.id, id: p.id })),
                backgroundColor: clusterColors[c % clusterColors.length],
                pointRadius: canvasId === "technique-cluster-chart" ? 3 : 5,
                pointHoverRadius: 7,
            });
        }

        const ctx = document.getElementById(canvasId).getContext("2d");
        new Chart(ctx, {
            type: "scatter",
            data: { datasets },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: title, color: "#e6edf3" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const d = ctx.raw;
                                return d.id ? `${d.id} — ${d.label}` : d.label;
                            }
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" }, title: { display: true, text: "UMAP-1", color: "#8b949e" } },
                    y: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" }, title: { display: true, text: "UMAP-2", color: "#8b949e" } }
                }
            }
        });
    }

    renderScatter("group-cluster-chart", groupClusters, `Threat Groups by TTP Similarity (k=${groupClusters.k})`);
    renderScatter("technique-cluster-chart", techClusters, `Techniques by Co-usage Pattern (k=${techClusters.k})`);

    // --- NLP Stats ---
    const nlpDiv = document.getElementById("nlp-stats");
    if (nlpStats) {
        const pDisplay = nlpStats.p_value === 0 ? "< 1e-300" : nlpStats.p_value.toExponential(1);
        nlpDiv.innerHTML = `
            <div class="nlp-stat"><div class="nlp-stat-value">${nlpStats.spearman_correlation}</div><div class="nlp-stat-label">Spearman ρ</div></div>
            <div class="nlp-stat"><div class="nlp-stat-value">${pDisplay}</div><div class="nlp-stat-label">p-value</div></div>
            <div class="nlp-stat"><div class="nlp-stat-value">${nlpStats.n_techniques_compared.toLocaleString()}</div><div class="nlp-stat-label">Techniques</div></div>
            <div class="nlp-stat"><div class="nlp-stat-value">${nlpStats.n_pairs.toLocaleString()}</div><div class="nlp-stat-label">Pairs Compared</div></div>
        `;
    } else {
        nlpDiv.innerHTML = '<p style="color:var(--muted)">NLP data not available.</p>';
    }

    // Pre-load T1566 (Phishing) so the chart isn't empty on first visit
    const defaultTech = "T1566";
    if (jaccardData[defaultTech]) {
        select.value = defaultTech;
        renderCooccurrence(defaultTech);
    }
})();
