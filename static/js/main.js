let historyLogs = [];

function switchTab(tabId, element) {
    document.querySelectorAll('.nav-item-link').forEach(el => el.classList.remove('active'));
    if (element) element.classList.add('active');

    if (tabId === 'tab-home') {
        document.getElementById('tab-home').style.display = 'block';
        document.getElementById('tab-predict-section').style.display = 'block';
        document.querySelectorAll('.tab-section').forEach(el => {
            if (el.id !== 'tab-home' && el.id !== 'tab-predict-section') el.style.display = 'none';
        });
    } else {
        document.querySelectorAll('.tab-section').forEach(el => el.style.display = 'none');
        const target = document.getElementById(tabId);
        if (target) target.style.display = 'block';
    }

    if (tabId === 'tab-stats') renderBarChart();
}

function updateVal(id, lblId) {
    document.getElementById(lblId).innerText = document.getElementById(id).value + " cm";
}

function onKernelChange() {
    const k = document.getElementById('kernelSelect').value;
    document.getElementById('kernelPlotTitle').innerText = k.toUpperCase();
    renderBoundaryChart(k);
}

async function runPredict() {
    const sl = parseFloat(document.getElementById('sl').value);
    const sw = parseFloat(document.getElementById('sw').value);
    const pl = parseFloat(document.getElementById('pl').value);
    const pw = parseFloat(document.getElementById('pw').value);
    const selectedKernel = document.getElementById('kernelSelect').value;

    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw, kernel: selectedKernel
        })
    });
    const data = await res.json();
    applyPredictResult(data);
}

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    uploadAndPredictImage(file);
}

async function uploadAndPredictImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        const reader = new FileReader();
        reader.onload = function(e) {
            data.img = e.target.result;
            applyPredictResult(data);

            const pastePreviewContainer = document.getElementById('pastePreviewContainer');
            if (pastePreviewContainer) {
                pastePreviewContainer.innerHTML = 
                    `<img src="${e.target.result}" class="preview-pasted-img d-block mx-auto mt-2"><span class="badge bg-success mt-2">Đã nhận diện ảnh</span>`;
            }
        };
        reader.readAsDataURL(file);
    } catch(e) {
        alert("Không thể phân loại ảnh này!");
    }
}

function applyPredictResult(data) {
    if (data.name) document.getElementById('resName').innerText = data.name;
    if (data.desc) document.getElementById('resDesc').innerText = data.desc;
    if (data.img) document.getElementById('resImg').src = data.img;
    if (data.used_kernel) {
        document.getElementById('usedKernelBadge').innerHTML = `<i class="bi bi-gear-wide-connected me-1"></i> Kernel: ${data.used_kernel}`;
        document.getElementById('kernelPlotTitle').innerText = data.used_kernel;
    }

    if (data.probs) {
        document.getElementById('prob_0').innerText = data.probs[0] + "%";
        document.getElementById('bar_0').style.width = data.probs[0] + "%";

        document.getElementById('prob_1').innerText = data.probs[1] + "%";
        document.getElementById('bar_1').style.width = data.probs[1] + "%";

        document.getElementById('prob_2').innerText = data.probs[2] + "%";
        document.getElementById('bar_2').style.width = data.probs[2] + "%";
    }

    const currentKernel = document.getElementById('kernelSelect').value;
    const currentPetalLen = parseFloat(document.getElementById('pl').value);
    const currentSepalLen = parseFloat(document.getElementById('sl').value);
    renderBoundaryChart(currentKernel, currentSepalLen, currentPetalLen);

    addHistory(data.used_kernel || "LINEAR", data.name);
}

function addHistory(method, result) {
    const now = new Date().toLocaleTimeString();
    historyLogs.unshift({ time: now, method: method, result: result });
    updateHistoryTable();
}

function updateHistoryTable() {
    const tbody = document.getElementById('historyTableBody');
    if (!tbody) return;
    tbody.innerHTML = historyLogs.map(item => `
        <tr>
            <td>${item.time}</td>
            <td><span class="badge bg-warning text-dark">${item.method}</span></td>
            <td><span class="badge bg-primary">${item.result}</span></td>
        </tr>
    `).join('');
}

function clearHistory() {
    historyLogs = [];
    updateHistoryTable();
}

window.addEventListener('paste', (e) => {
    const clipboardData = e.clipboardData || window.clipboardData;
    if (!clipboardData || !clipboardData.items) return;

    for (let item of clipboardData.items) {
        if (item.type.indexOf('image') !== -1) {
            const file = item.getAsFile();
            if (file) {
                const pasteTabBtn = document.getElementById('paste-tab-btn');
                if (pasteTabBtn) {
                    const bsTab = new bootstrap.Tab(pasteTabBtn);
                    bsTab.show();
                }
                uploadAndPredictImage(file);
            }
            break;
        }
    }
});

window.onload = function() {
    renderBoundaryChart('linear');
};
