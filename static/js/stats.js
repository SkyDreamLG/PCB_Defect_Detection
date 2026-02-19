// 图表实例
let defectTypeChart, ratioChart, dailyTrendChart, confidenceChart;

// 分页相关变量
let currentPage = 1;
let pageSize = 10;
let totalRecords = 0;
let allDefectRecords = [];

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 设置默认日期范围（最近30天）
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);

    document.getElementById('startDate').value = formatDate(thirtyDaysAgo);
    document.getElementById('endDate').value = formatDate(today);

    // 加载所有数据
    loadAllData();
});

// 格式化日期
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// 显示/隐藏加载动画
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// 加载所有数据
async function loadAllData() {
    showLoading(true);
    try {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        await Promise.all([
            loadOverviewStats(startDate, endDate),  // 传递日期参数
            loadDefectTypeStats(startDate, endDate),
            loadDailyTrend(),
            loadConfidenceDistribution(startDate, endDate),  // 传递日期参数
            loadAllDefectRecords(startDate, endDate)  // 传递日期参数
        ]);
    } catch (error) {
        console.error('加载数据失败:', error);
        showToast('加载数据失败', 'error');
    } finally {
        showLoading(false);
    }
}

// 加载概览统计（支持日期筛选）
async function loadOverviewStats(startDate, endDate) {
    try {
        let url = '/api/stats/overview';
        if (startDate && endDate) {
            url += `?start_date=${startDate}&end_date=${endDate}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        document.getElementById('totalCount').textContent = data.total_count || 0;
        document.getElementById('defectCount').textContent = data.defect_count || 0;
        document.getElementById('normalCount').textContent = data.normal_count || 0;
        document.getElementById('defectRatio').textContent = data.defect_ratio + '%';

        // 更新比例图表
        updateRatioChart(data.normal_count, data.defect_count);
    } catch (error) {
        console.error('加载概览统计失败:', error);
    }
}

// 加载缺陷类型统计
async function loadDefectTypeStats(startDate, endDate) {
    try {
        let url = '/api/stats/defect_types';
        if (startDate && endDate) {
            url += `?start_date=${startDate}&end_date=${endDate}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        updateDefectTypeChart(data);
        updateDefectDetailTable(data);
    } catch (error) {
        console.error('加载缺陷类型统计失败:', error);
    }
}

// 加载每日趋势
async function loadDailyTrend() {
    try {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        const response = await fetch(`/api/stats/daily?start_date=${startDate}&end_date=${endDate}`);
        const data = await response.json();

        updateDailyTrendChart(data);
    } catch (error) {
        console.error('加载每日趋势失败:', error);
    }
}

// 加载置信度分布（支持日期筛选）
async function loadConfidenceDistribution(startDate, endDate) {
    try {
        let url = '/api/stats/confidence_distribution';
        if (startDate && endDate) {
            url += `?start_date=${startDate}&end_date=${endDate}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        updateConfidenceChart(data);
    } catch (error) {
        console.error('加载置信度分布失败:', error);
    }
}

// 加载所有缺陷记录（支持日期筛选）
async function loadAllDefectRecords(startDate, endDate) {
    try {
        let url = '/query_history?has_defect=1';
        if (startDate && endDate) {
            url += `&start_date=${startDate}&end_date=${endDate}`;
        }

        // 获取所有有缺陷的记录
        const response = await fetch(url);
        const data = await response.json();

        allDefectRecords = data.records || [];
        totalRecords = allDefectRecords.length;

        // 重置到第一页
        currentPage = 1;

        // 更新分页控件
        updatePagination();

        // 显示当前页数据
        displayCurrentPage();
    } catch (error) {
        console.error('加载缺陷记录失败:', error);
        showToast('加载缺陷记录失败', 'error');
    }
}

// 显示当前页数据
function displayCurrentPage() {
    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, totalRecords);
    const pageData = allDefectRecords.slice(start, end);

    updateRecentDefectsTable(pageData);
    updatePaginationInfo(start + 1, end, totalRecords);
}

// 更新分页信息
function updatePaginationInfo(start, end, total) {
    const paginationInfo = document.getElementById('paginationInfo');
    if (paginationInfo) {
        if (total > 0) {
            paginationInfo.textContent = `显示 ${start}-${end} 条，共 ${total} 条记录`;
        } else {
            paginationInfo.textContent = '暂无记录';
        }
    }
}

// 更新分页控件
function updatePagination() {
    const totalPages = Math.ceil(totalRecords / pageSize);
    const paginationContainer = document.getElementById('pagination');

    if (!paginationContainer) return;

    if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        return;
    }

    let html = '<div class="pagination-controls">';

    // 上一页按钮
    html += `<button class="pagination-btn" onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>上一页</button>`;

    // 页码按钮
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    if (startPage > 1) {
        html += `<button class="pagination-btn" onclick="changePage(1)">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="pagination-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<span class="pagination-ellipsis">...</span>`;
        }
        html += `<button class="pagination-btn" onclick="changePage(${totalPages})">${totalPages}</button>`;
    }

    // 下一页按钮
    html += `<button class="pagination-btn" onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>下一页</button>`;

    html += '</div>';

    paginationContainer.innerHTML = html;
}

// 切换页码
window.changePage = (page) => {
    const totalPages = Math.ceil(totalRecords / pageSize);
    if (page < 1 || page > totalPages) return;

    currentPage = page;
    displayCurrentPage();
    updatePagination();
};

// 更改每页显示数量
window.changePageSize = () => {
    const select = document.getElementById('pageSizeSelect');
    pageSize = parseInt(select.value);
    currentPage = 1; // 重置到第一页
    displayCurrentPage();
    updatePagination();
};

// 更新比例图表
function updateRatioChart(normal, defect) {
    const ctx = document.getElementById('ratioChart').getContext('2d');

    if (ratioChart) {
        ratioChart.destroy();
    }

    ratioChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['合格PCB', '缺陷PCB'],
            datasets: [{
                data: [normal, defect],
                backgroundColor: ['#4f9e7a', '#f05454'],
                borderColor: ['#2d5940', '#8b3d3d'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e0e2e8',
                        font: { size: 12 }
                    }
                }
            }
        }
    });
}

// 更新缺陷类型图表
function updateDefectTypeChart(data) {
    const ctx = document.getElementById('defectTypeChart').getContext('2d');

    if (defectTypeChart) {
        defectTypeChart.destroy();
    }

    defectTypeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.name),
            datasets: [{
                label: '缺陷数量',
                data: data.map(item => item.count),
                backgroundColor: data.map(item => item.color),
                borderColor: '#fff',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#3e4052' },
                    ticks: { color: '#e0e2e8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e0e2e8' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 更新每日趋势图表
function updateDailyTrendChart(data) {
    const ctx = document.getElementById('dailyTrendChart').getContext('2d');

    if (dailyTrendChart) {
        dailyTrendChart.destroy();
    }

    dailyTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(item => item.date),
            datasets: [
                {
                    label: '总数',
                    data: data.map(item => item.total),
                    borderColor: '#3a6ea5',
                    backgroundColor: 'rgba(58, 110, 165, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '缺陷数',
                    data: data.map(item => item.defect),
                    borderColor: '#f05454',
                    backgroundColor: 'rgba(240, 84, 84, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#3e4052' },
                    ticks: { color: '#e0e2e8' }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#e0e2e8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e0e2e8' }
                }
            }
        }
    });
}

// 更新置信度分布图表
function updateConfidenceChart(data) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');

    if (confidenceChart) {
        confidenceChart.destroy();
    }

    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.range),
            datasets: [{
                label: '缺陷数量',
                data: data.map(item => item.count),
                backgroundColor: '#5f9df3',
                borderColor: '#3a6ea5',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#3e4052' },
                    ticks: { color: '#e0e2e8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e0e2e8' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 更新缺陷详情表格
function updateDefectDetailTable(data) {
    const tbody = document.querySelector('#defectDetailTable tbody');
    const total = data.reduce((sum, item) => sum + item.count, 0);

    if (total === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="empty-message">暂无缺陷数据</td></tr>';
        return;
    }

    let html = '';
    data.forEach(item => {
        const percentage = total > 0 ? ((item.count / total) * 100).toFixed(1) : 0;
        html += `
            <tr>
                <td>${item.name}</td>
                <td>${item.count}</td>
                <td>${percentage}%</td>
                <td>
                    <span class="color-indicator" style="background: ${item.color};"></span>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

// 更新最新缺陷记录表格
function updateRecentDefectsTable(records) {
    const tbody = document.getElementById('recentDefectsBody');

    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-message">暂无缺陷记录</td></tr>';
        return;
    }

    let html = '';
    records.forEach(record => {
        if (record.defects && record.defects.length > 0) {
            record.defects.forEach(defect => {
                html += `
                    <tr>
                        <td>${record.detection_time || '未知'}</td>
                        <td title="${record.filename}">${truncateString(record.filename, 30)}</td>
                        <td>
                            <span class="defect-badge ${defect.defect_type_en}">
                                ${defect.defect_type}
                            </span>
                        </td>
                        <td>${(defect.confidence * 100).toFixed(1)}%</td>
                        <td>
                            <a href="/history_detail/${record.id}" class="view-btn" target="_blank">查看</a>
                        </td>
                    </tr>
                `;
            });
        }
    });

    tbody.innerHTML = html;
}

// 截断长字符串
function truncateString(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength - 3) + '...';
}

// 应用日期筛选
async function applyDateFilter() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    showLoading(true);
    try {
        // 重新加载所有数据，都带上日期参数
        await Promise.all([
            loadOverviewStats(startDate, endDate),
            loadDefectTypeStats(startDate, endDate),
            loadDailyTrend(),
            loadConfidenceDistribution(startDate, endDate),
            loadAllDefectRecords(startDate, endDate)
        ]);
        showToast('数据已更新', 'success');
    } catch (error) {
        console.error('更新数据失败:', error);
        showToast('更新失败', 'error');
    } finally {
        showLoading(false);
    }
}

// 刷新所有数据
// 刷新所有数据
async function refreshAllData() {
    // 获取当前选择的日期范围
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    showLoading(true);
    try {
        await Promise.all([
            loadOverviewStats(startDate, endDate),
            loadDefectTypeStats(startDate, endDate),
            loadDailyTrend(),
            loadConfidenceDistribution(startDate, endDate),
            loadAllDefectRecords(startDate, endDate)
        ]);
        showToast('数据已刷新', 'success');
    } catch (error) {
        console.error('刷新数据失败:', error);
        showToast('刷新失败', 'error');
    } finally {
        showLoading(false);
    }
}

// 导出统计数据
async function exportStats() {
    try {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        showToast('正在导出统计数据...', 'info');

        const response = await fetch(`/api/stats/export?start_date=${startDate}&end_date=${endDate}`);
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
}

// 显示提示消息
function showToast(message, type = 'info') {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.className = 'toast';
        document.body.appendChild(toast);
    }

    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// 重置日期筛选
window.resetDateFilter = () => {
    // 设置默认日期范围（最近30天）
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);

    document.getElementById('startDate').value = formatDate(thirtyDaysAgo);
    document.getElementById('endDate').value = formatDate(today);

    // 重新加载所有数据
    refreshAllData();
    showToast('筛选已重置', 'success');
};