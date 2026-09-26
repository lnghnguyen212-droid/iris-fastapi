let barChartInstance = null;

window.onload = function() {
    initChart();
    updatePredict(); // Chạy dự đoán lần đầu khi vừa mở trang

    window.addEventListener('paste', e => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf("image") === 0) {
                const blob = item.getAsFile();
                uploadImage(blob);
            }
        }
    });
};

function initChart() {
    const ctx = document.getElementById('barChart').getContext('2d');
    barChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Linear', 'RBF', 'Poly', 'Sigmoid'],
            datasets: [{
                data: [98.5, 96.2, 92.4, 75.0],
                backgroundColor: ['#38bdf8', '#a855f7', '#ec4899', '#f59e0b'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 50, max: 100, grid: { color: '#232d42' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function onKernelChange() {
    const k = document.getElementById('kernelSelect').value.toUpperCase();
    document.getElementById('kernelBadge').innerText = 'KERNEL: ' + k;
    updatePredict();
}

async function updatePredict() {
    const sl = parseFloat(document.getElementById('sepal_length').value);
    const sw = parseFloat(document.getElementById('sepal_width').value);
    const pl = parseFloat(document.getElementById('petal_length').value);
    const pw = parseFloat(document.getElementById('petal_width').value);
    const kernel = document.getElementById('kernelSelect').value;

    document.getElementById('sl_val').innerText = sl + ' cm';
    document.getElementById('sw_val').innerText = sw + ' cm';
    document.getElementById('pl_val').innerText = pl + ' cm';
    document.getElementById('pw_val').innerText = pw + ' cm';

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sepal_length: sl,
                sepal_width: sw,
                petal_length: pl,
                petal_width: pw,
                kernel: kernel
            })
        });
        const data = await res.json();
        updateUI(data);
    } catch (err) {
        console.error("Lỗi gửi dữ liệu:", err);
    }
}

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) uploadImage(file);
}

async function uploadImage(file) {
    const preview = document.getElementById('uploadPreview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict-image', { method: 'POST', body: formData });
        const data = await res.json();
        updateUI(data);
    } catch (err) {
        console.error("Lỗi gửi ảnh:", err);
    }
}

function updateUI(data) {
    document.getElementById('resName').innerText = data.name;
    
    const imgEl = document.getElementById('resImg');
    const placeholder = document.getElementById('imgPlaceholder');
    imgEl.src = data.img;
    imgEl.style.display = 'block';
    placeholder.style.display = 'none';

    const probs = data.probs || [0, 0, 0];
    for (let i = 0; i < 3; i++) {
        document.getElementById(`prob_${i}`).innerText = probs[i] + '%';
        document.getElementById(`bar_${i}`).style.width = probs[i] + '%';
    }

    const tbody = document.getElementById('historyBody');
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const row = `<tr>
        <td>${timeStr}</td>
        <td><span class="badge bg-secondary">${data.used_kernel}</span></td>
        <td class="text-warning">${data.name}</td>
    </tr>`;
    tbody.innerHTML = row + tbody.innerHTML;
}

function clearHistory() {
    document.getElementById('historyBody').innerHTML = '';
}
