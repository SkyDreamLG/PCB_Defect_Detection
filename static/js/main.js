const POLLING_INTERVAL = 1000;
let uploadedFiles = [], currentTaskId = null, pollingInterval = null, isPolling = false, results = [];
let currentMode = 'detect';
let currentResultTab = 'normal';

const uploadSection = document.getElementById('uploadSection');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileListSection = document.getElementById('fileListSection');
const fileList = document.getElementById('fileList');
const fileCount = document.getElementById('fileCount');
const addMoreBtn = document.getElementById('addMoreBtn');
const clearAllFilesBtn = document.getElementById('clearAllFilesBtn');
const confThreshold = document.getElementById('confThreshold');
const iouThreshold = document.getElementById('iouThreshold');
const confValue = document.getElementById('confValue');
const iouValue = document.getElementById('iouValue');
const detectBtn = document.getElementById('detectBtn');
const downloadResultsBtn = document.getElementById('downloadResultsBtn');
const clearResultsBtn = document.getElementById('clearResultsBtn');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressDetail = document.getElementById('progressDetail');
const normalTableBody = document.getElementById('normalTableBody');
const defectTableBody = document.getElementById('defectTableBody');
const normalEmpty = document.getElementById('normalEmpty');
const defectEmpty = document.getElementById('defectEmpty');
const normalCount = document.getElementById('normalCount');
const defectCount = document.getElementById('defectCount');
const normalCard = document.getElementById('normalCard');
const defectCard = document.getElementById('defectCard');
const resultsCard = document.getElementById('resultsCard');
const historyCard = document.getElementById('historyCard');
const historyTableBody = document.getElementById('historyTableBody');
const historyCount = document.getElementById('historyCount');
const navDetect = document.getElementById('navDetect');
const navHistory = document.getElementById('navHistory');
const detectMode = document.getElementById('detectMode');
const historyMode = document.getElementById('historyMode');
const resultTabs = document.getElementById('resultTabs');

document.addEventListener('DOMContentLoaded', ()=>{
    confValue.textContent = confThreshold.value;
    iouValue.textContent = iouThreshold.value;
    updateLayout();
    downloadResultsBtn.disabled = true;

    //设置默认日期
    const today = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(today.getDate() - 7);

    document.getElementById('startDate').value = formatDate(sevenDaysAgo);
    document.getElementById('endDate').value = formatDate(today);
});

function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}



//切换
window.switchMode = (mode) => {
    currentMode = mode;
    if (mode === 'detect') {
        navDetect.classList.add('active');
        navHistory.classList.remove('active');
        detectMode.style.display = 'block';
        historyMode.style.display = 'none';
        resultTabs.style.display = 'flex';
        resultsCard.style.display = 'block';
        historyCard.style.display = 'none';
    } else {
        navDetect.classList.remove('active');
        navHistory.classList.add('active');
        detectMode.style.display = 'none';
        historyMode.style.display = 'block';
        resultTabs.style.display = 'none';
        resultsCard.style.display = 'none';
        historyCard.style.display = 'block';
        queryHistory();
    }
};

//正常/缺陷 tab
function updateTabActive(tab) {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach((el, index) => {
        if (index === 0) {
            el.classList.toggle('active', tab === 'normal');
        } else {
            el.classList.toggle('active', tab === 'defect');
        }
    });
}

window.switchResultTab = (tab) => {
    currentResultTab = tab;
    updateTabActive(tab);

    if (tab === 'normal') {
        normalCard.style.display = 'block';
        defectCard.style.display = 'none';
    } else {
        normalCard.style.display = 'none';
        defectCard.style.display = 'block';
    }
};

function updateLayout() {
    const n = parseInt(normalCount.textContent);
    const d = parseInt(defectCount.textContent);

    if(n === 0 && d === 0) {
        normalCard.style.display = 'block';
        defectCard.style.display = 'none';
        currentResultTab = 'normal';
        updateTabActive('normal');
    }
    else if(n === 0) {
        normalCard.style.display = 'none';
        defectCard.style.display = 'block';
        currentResultTab = 'defect';
        updateTabActive('defect');
    }
    else if(d === 0) {
        defectCard.style.display = 'none';
        normalCard.style.display = 'block';
        currentResultTab = 'normal';
        updateTabActive('normal');
    }
    else {
        if (currentResultTab === 'normal') {
            normalCard.style.display = 'block';
            defectCard.style.display = 'none';
        } else {
            normalCard.style.display = 'none';
            defectCard.style.display = 'block';
        }
        updateTabActive(currentResultTab);
    }
}

//文件上传处理
confThreshold.addEventListener('input', ()=> confValue.textContent = confThreshold.value);
iouThreshold.addEventListener('input', ()=> iouValue.textContent = iouThreshold.value);
dropZone.addEventListener('click', ()=> fileInput.click());

fileInput.addEventListener('change', (e)=> {
    handleFiles(e.target.files);
    fileInput.value = '';
});

addMoreBtn.addEventListener('click', ()=> fileInput.click());
clearAllFilesBtn.addEventListener('click', ()=> clearAllFiles());

['dragenter','dragover','dragleave','drop'].forEach(ev => dropZone.addEventListener(ev, preventDefaults));
['dragenter','dragover'].forEach(ev => dropZone.addEventListener(ev, ()=> dropZone.classList.add('drag-over')));
['dragleave','drop'].forEach(ev => dropZone.addEventListener(ev, ()=> dropZone.classList.remove('drag-over')));

dropZone.addEventListener('drop', (e)=>{
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    const imageFiles = Array.from(files).filter(f=> f.type.startsWith('image/'));
    if(imageFiles.length===0) {
        showToast('请选择图片文件','error');
        return;
    }
    if(uploadedFiles.length+imageFiles.length>1000) {
        showToast('最多处理1000张图片','error');
        return;
    }
    imageFiles.forEach(file=>{
        const exists = uploadedFiles.some(f=> f.name===file.name && f.size===file.size);
        if(!exists) uploadedFiles.push(file);
    });
    updateFileList();
    showToast(`已添加 ${imageFiles.length} 张`,'success');
}

function updateFileList() {
    if(uploadedFiles.length===0) {
        uploadSection.classList.remove('hidden');
        fileListSection.style.display='none';
        return;
    }
    uploadSection.classList.add('hidden');
    fileListSection.style.display='block';

    let html='';
    uploadedFiles.forEach((file,idx)=>{
        const size=(file.size/1024).toFixed(1);
        const ext=file.name.split('.').pop().toUpperCase();
        html+=`<div class="file-item" data-index="${idx}">
            <div class="file-icon">${ext}</div>
            <div class="file-info">
                <div class="file-name" title="${file.name}">${file.name}</div>
                <div class="file-meta">${size}KB</div>
            </div>
            <button class="file-remove" onclick="removeFile(${idx})">✕</button>
        </div>`;
    });
    fileList.innerHTML=html;
    fileCount.textContent=uploadedFiles.length;
}

window.removeFile = (index) => {
    uploadedFiles.splice(index,1);
    updateFileList();
    showToast('已移除','info');
};

function clearAllFiles() {
    if(uploadedFiles.length===0) return;
    uploadedFiles=[];
    updateFileList();
    showToast('清空所有文件','info');
}

//检测任务
detectBtn.addEventListener('click', async()=>{
    if(uploadedFiles.length===0) {
        showToast('请上传图片','error');
        return;
    }
    stopPolling();
    clearResults();
    resetProgress();
    setButtonsDisabled(true);

    document.getElementById('pendingTasks').textContent = uploadedFiles.length;

    const formData = new FormData();
    uploadedFiles.forEach(f=> formData.append('files',f));
    formData.append('conf_threshold',confThreshold.value);
    formData.append('iou_threshold',iouThreshold.value);

    showToast('任务提交中...','info');

    try{
        const resp = await fetch('/start_batch_detect',{method:'POST',body:formData});
        if(!resp.ok) throw new Error('请求失败');
        const res = await resp.json();
        if(res.error) throw new Error(res.error);
        currentTaskId = res.task_id;
        startPolling();
    } catch(e) {
        showToast('检测失败: '+e.message,'error');
        setButtonsDisabled(false);
    }
});

downloadResultsBtn.addEventListener('click', async()=>{
    if(!results || results.length===0) {
        showToast('无结果可下载','error');
        return;
    }
    try{
        showToast('打包下载中...','info');
        const resp = await fetch('/download_results',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({results})
        });
        if(!resp.ok) throw new Error('下载失败');
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pcb_defect_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showToast('下载成功','success');
    } catch(e) {
        showToast('下载失败: '+e.message,'error');
    }
});

clearResultsBtn.addEventListener('click',()=>{
    clearResults();
    showToast('结果已清空','info');
});

function clearResults() {
    normalTableBody.innerHTML='';
    defectTableBody.innerHTML='';
    normalEmpty.style.display='none';
    defectEmpty.style.display='none';
    results=[];
    downloadResultsBtn.disabled=true;
    updateLayout();
}

function resetProgress() {
    progressFill.style.width='0%';
    progressText.textContent='0%';
    progressDetail.textContent='0/0';
}

function setButtonsDisabled(disabled){
    detectBtn.disabled = disabled;
    detectBtn.innerHTML = disabled ? '⏳ 处理中...' : '▶ 启动检测';
}

function startPolling() {
    if(pollingInterval) clearInterval(pollingInterval);
    isPolling=true;
    pollingInterval=setInterval(pollForResults,POLLING_INTERVAL);
}

function stopPolling() {
    if(pollingInterval) clearInterval(pollingInterval);
    pollingInterval=null;
    isPolling=false;
    setButtonsDisabled(false);
}

async function pollForResults(){
    if(!currentTaskId || !isPolling) return;
    try{
        const resp = await fetch(`/get_detection_result/${currentTaskId}`);
        const data = await resp.json();
        if(data.status==='completed') {
            updateResults(data.results);
            updateProgress(100, data.results.length, uploadedFiles.length);
            document.getElementById('pendingTasks').textContent = '0';
            stopPolling();
            showToast('检测完成！','success');
            downloadResultsBtn.disabled=false;
        }
        else if(data.status==='processing' && data.results){
            updateResults(data.results);
            const completed = data.results.filter(r=> r.status==='completed'||r.status==='error').length;
            const total = uploadedFiles.length;
            const pending = total - completed;
            document.getElementById('pendingTasks').textContent = pending;
            updateProgress(data.progress, completed, total);
        } else if(data.status==='error') {
            showToast('处理出错: '+data.error,'error');
            stopPolling();
        }
    } catch(e) {
        console.error(e);
    }
}

function updateResults(newResults){
    if(!newResults||!newResults.length) return;
    results = newResults;
    const normal = [], defect = [];
    results.forEach(item=>{
        if(item.status==='processing') return;
        (item.detections && item.detections.length>0 ? defect : normal).push(item);
    });

    if(normal.length>0){
        let html='';
        normal.forEach(item=>{
            html+=`<tr>
                <td><img src="${item.original_image}" class="thumbnail" onclick="showLargeImage('${item.original_image}')" onerror="this.src='data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2264%22%20height%3D%2264%22%20viewBox%3D%220%200%2064%2064%22%3E%3Crect%20width%3D%2264%22%20height%3D%2264%22%20fill%3D%22%23222%22%2F%3E%3Ctext%20x%3D%2232%22%20y%3D%2232%22%20fill%3D%22%23999%22%3E%3F%3C%2Ftext%3E%3C%2Fsvg%3E'"></td>
                <td class="filename" title="${item.filename}">${item.filename}</td>
            </tr>`;
        });
        normalTableBody.innerHTML=html;
        normalEmpty.style.display='none';
    } else {
        normalTableBody.innerHTML='';
        normalEmpty.style.display='block';
    }

    if(defect.length>0){
        let html='';
        defect.forEach(item=>{
            const defectsHtml = item.detections.map(d=>
                `<div class="defect-item ${d.defect_type_en}">
                    <div class="defect-type">${d.defect_type}</div>
                    <div class="defect-desc">${d.description||''}</div>
                    <div class="defect-meta">置信度:${(d.confidence*100).toFixed(1)}% 尺寸:${d.width}x${d.height}</div>
                </div>`
            ).join('');
            html+=`<tr>
                <td><img src="${item.detected_image}" class="thumbnail" onclick="showLargeImage('${item.detected_image}')" onerror="this.src='...'"></td>
                <td class="filename">${item.filename}</td>
                <td><div class="defect-list">${defectsHtml}</div></td>
            </tr>`;
        });
        defectTableBody.innerHTML=html;
        defectEmpty.style.display='none';
    } else {
        defectTableBody.innerHTML='';
        defectEmpty.style.display='block';
    }

    normalCount.textContent=normal.length;
    defectCount.textContent=defect.length;

    if (normal.length === 0 && defect.length > 0) {
        currentResultTab = 'defect';
    } else if (normal.length > 0 && defect.length === 0) {
        currentResultTab = 'normal';
    }

    updateLayout();
}

function updateProgress(percent,completed,total) {
    progressFill.style.width=percent+'%';
    progressText.textContent=Math.round(percent)+'%';
    progressDetail.textContent=completed+'/'+total;
}

function showToast(msg,type='info') {
    const toast=document.getElementById('toast');
    toast.textContent=msg;
    toast.className=`toast show ${type}`;
    setTimeout(()=>toast.classList.remove('show'),3000);
}

//历史记录查询
window.queryHistory = async () => {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const filename = document.getElementById('filenameFilter').value;
    const hasDefect = document.getElementById('defectFilter').value;

    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (filename) params.append('filename', filename);
    if (hasDefect !== '') params.append('has_defect', hasDefect);

    try {
        const resp = await fetch(`/query_history?${params.toString()}`);
        const data = await resp.json();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        updateHistoryTable(data.records);
        historyCount.textContent = data.total;
        showToast(`查询到 ${data.total} 条记录`, 'success');
    } catch (e) {
        showToast('查询失败: ' + e.message, 'error');
    }
};

function updateHistoryTable(records) {
    if (!records || records.length === 0) {
        historyTableBody.innerHTML = '<tr><td colspan="5" class="empty-message">暂无历史记录</td></tr>';
        return;
    }

    let html = '';
    records.forEach(record => {
        const defectClass = record.has_defect ? 'defect-badge' : 'defect-badge normal';
        const defectText = record.has_defect ? `有缺陷 (${record.defect_count}个)` : '正常';

        html += `
            <tr>
                <td>
                    <img src="/get_history_image/${record.id}?type=detected" 
                         class="history-thumbnail" 
                         onclick="showLargeImage('/get_history_image/${record.id}?type=detected')"
                         onerror="this.src='data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2250%22%20height%3D%2250%22%20viewBox%3D%220%200%2050%2050%22%3E%3Crect%20width%3D%2250%22%20height%3D%2250%22%20fill%3D%22%23222%22%2F%3E%3Ctext%20x%3D%2225%22%20y%3D%2225%22%20fill%3D%22%23999%22%3E%3F%3C%2Ftext%3E%3C%2Fsvg%3E'">
                </td>
                <td>${record.filename}</td>
                <td>${record.detection_time}</td>
                <td><span class="${defectClass}">${defectText}</span></td>
                <td>
                    <div class="history-actions">
                        <button class="history-btn" onclick="viewHistoryDetail(${record.id})">详情</button>
                        <button class="history-btn" onclick="downloadHistoryItem(${record.id})">下载</button>
                    </div>
                </td>
            </tr>
        `;
    });
    historyTableBody.innerHTML = html;
}

window.resetFilters = () => {
    const today = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(today.getDate() - 7);

    document.getElementById('startDate').value = formatDate(sevenDaysAgo);
    document.getElementById('endDate').value = formatDate(today);
    document.getElementById('filenameFilter').value = '';
    document.getElementById('defectFilter').value = '';
    queryHistory();
};

window.refreshHistory = () => {
    queryHistory();
};

window.downloadHistory = async () => {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const filename = document.getElementById('filenameFilter').value;
    const hasDefect = document.getElementById('defectFilter').value;

    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (filename) params.append('filename', filename);
    if (hasDefect !== '') params.append('has_defect', hasDefect);

    try {
        showToast('正在打包下载...', 'info');
        const resp = await fetch(`/download_history?${params.toString()}`);
        if (!resp.ok) throw new Error('下载失败');

        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `history_${formatDate(new Date())}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showToast('下载成功', 'success');
    } catch (e) {
        showToast('下载失败: ' + e.message, 'error');
    }
};

window.viewHistoryDetail = (id) => {
    window.open(`/history_detail/${id}`, '_blank');
};

window.downloadHistoryItem = async (id) => {
    try {
        const resp = await fetch(`/download_history_item/${id}`);
        if (!resp.ok) throw new Error('下载失败');

        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `history_${id}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showToast('下载成功', 'success');
    } catch (e) {
        showToast('下载失败: ' + e.message, 'error');
    }
};

// 大图预览
window.showLargeImage = (src) => {
    const modal=document.getElementById('imageModal'), modalImage=document.getElementById('modalImage');
    modalImage.src=src;
    modal.style.display='flex';
    resetImageTransform();
};

document.querySelector('.modal-close').onclick = ()=> document.getElementById('imageModal').style.display='none';
window.onclick = (e)=> {
    if(e.target === document.getElementById('imageModal'))
        document.getElementById('imageModal').style.display='none';
};

let scale=1, translateX=0, translateY=0, isDragging=false, startX, startY;

function resetImageTransform() {
    scale=1;
    translateX=0;
    translateY=0;
    updateImageTransform();
}

function updateImageTransform() {
    document.getElementById('modalImage').style.transform=`translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

document.getElementById('zoomIn').onclick = ()=> {
    scale = Math.min(scale+0.2,3);
    updateImageTransform();
};

document.getElementById('zoomOut').onclick = ()=> {
    scale = Math.max(scale-0.2,0.5);
    updateImageTransform();
};

document.getElementById('resetView').onclick = resetImageTransform;

const modalImage = document.getElementById('modalImage');
modalImage.addEventListener('mousedown',(e)=>{
    if(e.button!==0)return;
    isDragging=true;
    startX=e.clientX-translateX;
    startY=e.clientY-translateY;
    modalImage.style.cursor='grabbing';
    e.preventDefault();
});

document.addEventListener('mousemove',(e)=>{
    if(!isDragging)return;
    translateX=e.clientX-startX;
    translateY=e.clientY-startY;
    updateImageTransform();
});

document.addEventListener('mouseup',()=>{
    isDragging=false;
    modalImage.style.cursor='grab';
});

modalImage.addEventListener('wheel',(e)=>{
    e.preventDefault();
    const delta=e.deltaY>0?-0.1:0.1;
    scale=Math.max(0.5,Math.min(3,scale+delta));
    updateImageTransform();
});

// 导出统计功能

window.exportStats = async () => {
    try {
        showToast('正在导出统计数据...', 'info');

        const response = await fetch('/api/stats/export');
        if (!response.ok) throw new Error('导出失败');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `stats_${formatDate(new Date())}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showToast('导出成功', 'success');
    } catch (error) {
        console.error('导出失败:', error);
        showToast('导出失败: ' + error.message, 'error');
    }
};

//刷新数据
window.refreshAllData = async () => {
    showToast('刷新数据中...', 'info');
    location.reload();
};

if (typeof formatDate !== 'function') {
    window.formatDate = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    };
}

// 清除历史记录相关变量
let clearMode = 'filtered'; // 'filtered' / 'all'
let recordsToDelete = [];

// 显示清除历史记录确认模态框
function showClearHistoryModal(mode) {
    clearMode = mode;

    const modal = document.getElementById('clearHistoryModal');
    const title = document.getElementById('clearModalTitle');
    const message = document.getElementById('clearModalMessage');
    const countInfo = document.getElementById('clearCountInfo');

    if (mode === 'all') {
        title.textContent = '清除全部历史记录';
        message.textContent = '确定要清除所有历史记录吗？此操作不可恢复，所有图片文件也将被删除。';

        // 获取总记录数
        const totalRecords = document.getElementById('historyCount').textContent;
        countInfo.textContent = `将删除全部 ${totalRecords} 条历史记录`;
    } else {
        title.textContent = '清除筛选结果';

        // 获取当前显示的记录数
        const tableBody = document.getElementById('historyTableBody');
        const rows = tableBody.querySelectorAll('tr:not(.empty-message)');
        const recordCount = rows.length;

        // 收集要删除的记录ID
        recordsToDelete = [];
        rows.forEach(row => {
            const deleteBtn = row.querySelector('button[onclick^="deleteHistoryRecord"]');
            if (deleteBtn) {
                const onclickAttr = deleteBtn.getAttribute('onclick');
                const match = onclickAttr.match(/deleteHistoryRecord\((\d+)\)/);
                if (match) {
                    recordsToDelete.push(parseInt(match[1]));
                }
            }
        });

        message.textContent = '确定要清除当前筛选出的历史记录吗？此操作不可恢复，对应的图片文件也将被删除。';
        countInfo.textContent = `将删除 ${recordCount} 条筛选出的记录`;

        if (recordCount === 0) {
            alert('当前没有符合条件的记录可清除');
            return;
        }
    }

    modal.style.display = 'flex';
}

// 关闭清除模态框
function closeClearModal() {
    document.getElementById('clearHistoryModal').style.display = 'none';
    recordsToDelete = [];
}

// 确认清除历史记录
function confirmClearHistory() {
    const confirmBtn = document.getElementById('confirmClearBtn');
    confirmBtn.disabled = true;
    confirmBtn.textContent = '清除中...';

    let url = '/api/clear_history';
    let body = {};

    if (clearMode === 'all') {
        body = { mode: 'all' };
    } else {
        body = {
            mode: 'filtered',
            record_ids: recordsToDelete,
            filters: {
                start_date: document.getElementById('startDate').value,
                end_date: document.getElementById('endDate').value,
                filename: document.getElementById('filenameFilter').value,
                has_defect: document.getElementById('defectFilter').value
            }
        };
    }

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast(`成功清除 ${data.deleted_count} 条记录`, 'success');
            //刷新历史记录列表
            refreshHistory();
            //更新统计信息
            if (typeof updateStats === 'function') {
                updateStats();
            }
        } else {
            showToast('清除失败: ' + (data.error || '未知错误'), 'error');
        }

        closeClearModal();
    })
    .catch(error => {
        console.error('清除失败:', error);
        showToast('清除失败: ' + error.message, 'error');
        closeClearModal();
    })
    .finally(() => {
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认清除';
    });
}

//删除单条历史记录
function deleteHistoryRecord(recordId) {
    if (!confirm('确定要删除这条历史记录吗？此操作不可恢复，对应的图片文件也将被删除。')) {
        return;
    }

    fetch('/api/clear_history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            mode: 'single',
            record_ids: [recordId]
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('删除成功', 'success');
            refreshHistory();
        } else {
            showToast('删除失败: ' + (data.error || '未知错误'), 'error');
        }
    })
    .catch(error => {
        console.error('删除失败:', error);
        showToast('删除失败: ' + error.message, 'error');
    });
}


function renderHistoryTable(records) {
    const tbody = document.getElementById('historyTableBody');
    const historyCount = document.getElementById('historyCount');

    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-message">暂无历史记录</td></tr>';
        historyCount.textContent = '0';
        return;
    }

    let html = '';
    records.forEach(record => {
        const hasDefect = record.has_defect ? '是' : '否';
        const defectClass = record.has_defect ? 'defect-badge' : 'normal-badge';

        let defectSummary = '';
        if (record.defects && record.defects.length > 0) {
            const defectTypes = record.defects.map(d => d.defect_type).join(', ');
            defectSummary = `${record.defect_count}个缺陷 (${defectTypes})`;
        } else {
            defectSummary = '无';
        }

        html += `
            <tr>
                <td>
                    <img src="/get_history_image/${record.id}?type=detected" 
                         class="thumbnail" 
                         onclick="showImageModal('${record.id}', 'detected')"
                         style="cursor: pointer; width: 60px; height: 60px; object-fit: cover;">
                </td>
                <td>${record.filename}</td>
                <td>${record.detection_time}</td>
                <td><span class="${defectClass}">${hasDefect}</span><br><small>${defectSummary}</small></td>
                <td>
                    <div style="display: flex; gap: 5px;">
                        <button class="table-btn" onclick="downloadHistoryItem(${record.id})" title="下载">⬇</button>
                        <button class="table-btn" onclick="viewHistoryDetail(${record.id})" title="查看详情">👁️</button>
                        <button class="table-btn delete-btn" onclick="deleteHistoryRecord(${record.id})" title="删除" style="color: #e74c3c;">✕</button>
                    </div>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    historyCount.textContent = records.length;
}

function queryHistory() {
    //获取筛选条件
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const filename = document.getElementById('filenameFilter').value;
    const hasDefect = document.getElementById('defectFilter').value;

    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (filename) params.append('filename', filename);
    if (hasDefect !== '') params.append('has_defect', hasDefect);

    const tbody = document.getElementById('historyTableBody');
    tbody.innerHTML = '<tr><td colspan="6" class="empty-message">加载中...</td></tr>';

    fetch(`/query_history?${params.toString()}`)
        .then(response => response.json())
        .then(data => {
            if (data.records) {
                renderHistoryTable(data.records);
            } else {
                renderHistoryTable([]);
                if (data.error) {
                    showToast('查询失败: ' + data.error, 'error');
                }
            }
        })
        .catch(error => {
            console.error('查询失败:', error);
            showToast('查询失败: ' + error.message, 'error');
            renderHistoryTable([]);
        });
}