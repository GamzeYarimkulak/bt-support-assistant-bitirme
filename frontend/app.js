const API_BASE_URL = 'http://localhost:8000';

const API_ENDPOINTS = {
    chat: `${API_BASE_URL}/api/v1/chat`,
    anomalyStats: `${API_BASE_URL}/api/v1/anomaly/stats`,
    anomalyDetect: `${API_BASE_URL}/api/v1/anomaly/detect`,
    anomalyQuality: `${API_BASE_URL}/api/v1/anomaly/quality`
};

let chatMessages = [];
let sessionId = null;
let qualityLoaded = false;
const CHAT_HISTORY_STORAGE_KEY = 'bt_support_chat_messages';
const MAX_STORED_CHAT_MESSAGES = 20;
const MAX_CONTEXT_MESSAGES = 10;

function getOrCreateSessionId() {
    if (sessionId) {
        return sessionId;
    }

    sessionId = localStorage.getItem('bt_support_session_id');
    if (!sessionId) {
        sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
        localStorage.setItem('bt_support_session_id', sessionId);
    }

    return sessionId;
}

function loadStoredChatMessages() {
    try {
        const storedMessages = JSON.parse(localStorage.getItem(CHAT_HISTORY_STORAGE_KEY) || '[]');
        if (!Array.isArray(storedMessages)) {
            chatMessages = [];
            return;
        }

        chatMessages = storedMessages
            .filter(message => message && ['user', 'assistant'].includes(message.role) && message.text)
            .slice(-MAX_STORED_CHAT_MESSAGES);
    } catch (error) {
        chatMessages = [];
        localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
    }
}

function saveChatMessages() {
    const messagesToStore = chatMessages
        .filter(message => ['user', 'assistant'].includes(message.role))
        .slice(-MAX_STORED_CHAT_MESSAGES);
    localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(messagesToStore));
}

function buildApiConversationHistory(messages) {
    return messages
        .filter(message => ['user', 'assistant'].includes(message.role) && message.text)
        .slice(-MAX_CONTEXT_MESSAGES)
        .map(message => ({
            role: message.role,
            content: String(message.text)
        }));
}

function renderChatMessages() {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) {
        return;
    }

    if (chatMessages.length === 0) {
        messagesContainer.innerHTML = `
            <div class="chat-empty-state">
                <div class="chat-empty-state-icon"></div>
                <div class="chat-empty-state-text">Oturum hazır</div>
                <div class="chat-empty-state-hint">BT destek soruları için RAG hattı beklemede</div>
            </div>
        `;
        return;
    }

    messagesContainer.innerHTML = '';
    chatMessages.forEach(message => {
        messagesContainer.appendChild(createMessageElement(message));
    });
    scrollToBottom();
}

function createMessageElement(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;

    if (message.role === 'user') {
        messageDiv.innerHTML = `
            <div class="message-bubble">
                <div class="message-content">${escapeHtml(message.text)}</div>
                <div class="message-timestamp">${formatTimestamp(message.timestamp)}</div>
            </div>
        `;
        return messageDiv;
    }

    if (message.role === 'error') {
        messageDiv.className = 'message error';
        messageDiv.innerHTML = `
            <div class="message-bubble">
                <div class="message-content">${escapeHtml(message.text)}</div>
                <div class="message-timestamp">${formatTimestamp(message.timestamp)}</div>
            </div>
        `;
        return messageDiv;
    }

    const confidenceBadgeHtml = message.confidence !== undefined
        ? `<span class="confidence-badge ${getConfidenceClass(message.confidence)}">Güven: ${Math.round(message.confidence * 100)}%</span>`
        : '';

    const languageBadge = message.language
        ? `<span class="message-badge">${escapeHtml(String(message.language).toUpperCase())}</span>`
        : '';

    const hasAnswerBadge = message.has_answer !== undefined
        ? `<span class="message-badge">${message.has_answer ? 'Yanıt var' : 'Yanıt yok'}</span>`
        : '';

    const sourcesHtml = renderSources(message.sources || []);
    const debugHtml = renderDebugInfo(message.debug_info);

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${formatAssistantMessage(message.text)}</div>
            <div class="message-metadata">
                ${confidenceBadgeHtml}
                ${hasAnswerBadge}
                ${languageBadge}
            </div>
            ${sourcesHtml}
            ${debugHtml}
            <div class="message-timestamp">${formatTimestamp(message.timestamp)}</div>
        </div>
    `;

    return messageDiv;
}

function renderSources(sources) {
    if (!sources.length) {
        return '';
    }

    return `
        <div class="message-sources">
            <div class="message-sources-title">Kaynaklar</div>
            ${sources.map((source, idx) => {
                const title = source.title || source.doc_id || `Kaynak ${idx + 1}`;
                const score = Number(source.relevance_score ?? 0);
                return `
                    <div class="message-source-item">
                        <span class="message-source-title">${idx + 1}. ${escapeHtml(title)}</span>
                        <span class="message-source-score">Skor ${score.toFixed(2)}</span>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function renderDebugInfo(debugInfo) {
    if (!debugInfo) {
        return '';
    }

    const alpha = debugInfo.alpha_used;
    const alphaLabel = alpha === null || alpha === undefined ? 'N/A' : Number(alpha).toFixed(2);
    const alphaHint = alpha < 0.4
        ? '(Embedding ağırlıklı)'
        : alpha > 0.6
            ? '(BM25 ağırlıklı)'
            : '(Dengeli)';

    return `
        <div class="message-debug-info">
            <div class="message-debug-title">Arama detayları</div>
            <div class="message-debug-content">
                <div class="debug-item">
                    <span class="debug-label">Dinamik Alpha:</span>
                    <span class="debug-value">${alphaLabel}</span>
                    <span class="debug-hint">${alpha === null || alpha === undefined ? '' : alphaHint}</span>
                </div>
                <div class="debug-item">
                    <span class="debug-label">Sorgu Tipi:</span>
                    <span class="debug-value">${escapeHtml(debugInfo.query_type || 'N/A')}</span>
                </div>
                <div class="debug-item">
                    <span class="debug-label">BM25 sonuçları:</span>
                    <span class="debug-value">${debugInfo.bm25_results_count || 0}</span>
                </div>
                <div class="debug-item">
                    <span class="debug-label">Embedding sonuçları:</span>
                    <span class="debug-value">${debugInfo.embedding_results_count || 0}</span>
                </div>
                <div class="debug-item">
                    <span class="debug-label">Hibrit sonuçlar:</span>
                    <span class="debug-value">${debugInfo.hybrid_results_count || 0}</span>
                </div>
            </div>
        </div>
    `;
}

function formatAssistantMessage(text = '') {
    const lines = String(text || '').split(/\r?\n/);
    const html = [];
    let openList = null;

    const closeList = () => {
        if (openList) {
            html.push(`</${openList}>`);
            openList = null;
        }
    };

    const openListIfNeeded = (type) => {
        if (openList === type) {
            return;
        }
        closeList();
        const className = type === 'ol' ? 'answer-list answer-list-numbered' : 'answer-list';
        html.push(`<${type} class="${className}">`);
        openList = type;
    };

    for (const rawLine of lines) {
        const line = rawLine.trim();

        if (!line) {
            closeList();
            continue;
        }

        if (/^(Sorunuz|Your question):/i.test(line)) {
            closeList();
            html.push(`<div class="answer-question">${formatInline(line)}</div>`);
            continue;
        }

        if (/^\*\*[^*]+:\*\*$/.test(line) || /^\*\*[^*]+\*\*$/.test(line)) {
            closeList();
            const title = line.replace(/^\*\*/, '').replace(/\*\*$/, '').replace(/:$/, '');
            html.push(`<div class="answer-section-title">${formatInline(title)}</div>`);
            continue;
        }

        const numberedMatch = line.match(/^(\d+)\.\s+(.+)$/);
        if (numberedMatch) {
            openListIfNeeded('ol');
            html.push(`<li>${formatInline(numberedMatch[2])}</li>`);
            continue;
        }

        const bulletMatch = line.match(/^[-•✓]\s+(.+)$/);
        if (bulletMatch) {
            openListIfNeeded('ul');
            html.push(`<li>${formatInline(bulletMatch[1])}</li>`);
            continue;
        }

        if (/^\((Toplam|Total)\s+/i.test(line)) {
            closeList();
            html.push(`<div class="answer-note">${formatInline(line)}</div>`);
            continue;
        }

        if (/^(Kaynak|Kaynaklar|Source|Sources):/i.test(line)) {
            closeList();
            html.push(`<div class="answer-source-note">${formatInline(line)}</div>`);
            continue;
        }

        closeList();
        html.push(`<p class="answer-paragraph">${formatInline(line)}</p>`);
    }

    closeList();
    return html.join('');
}

function formatInline(value = '') {
    return escapeHtml(value).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.7) return 'confidence-high';
    if (confidence >= 0.4) return 'confidence-medium';
    return 'confidence-low';
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value === null || value === undefined ? '' : String(value);
    return div.innerHTML;
}

function scrollToBottom() {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function buildContextualQuery(messages, currentInput) {
    const MAX_CONTEXT_MESSAGES = 4;
    const MAX_USER_MESSAGE_LENGTH = 150;
    const MAX_ASSISTANT_BRIEF = 120;
    const MAX_ASSISTANT_FULL = 600;
    const MAX_TOTAL_LENGTH = 1200;

    if (!messages || messages.length === 0) {
        return currentInput;
    }

    const recentMessages = messages
        .filter(msg => msg.role !== 'error')
        .slice(-MAX_CONTEXT_MESSAGES);

    if (recentMessages.length === 0) {
        return currentInput;
    }

    const stepPattern = /(\d+|birinci|ikinci|üçüncü|dördüncü|beşinci)\s*\.?\s*adım/i;
    const isStepFollowUp = stepPattern.test(currentInput);
    const contextLines = [];
    let totalLength = 0;

    for (let i = 0; i < recentMessages.length; i++) {
        const msg = recentMessages[i];
        const roleLabel = msg.role === 'user' ? 'Kullanıcı' : 'Asistan';
        let msgText = msg.text || '';
        let maxLen = msg.role === 'user' ? MAX_USER_MESSAGE_LENGTH : MAX_ASSISTANT_BRIEF;

        if (msg.role === 'assistant') {
            const isLastAssistant = (i === recentMessages.length - 1)
                || (i === recentMessages.length - 2 && recentMessages[recentMessages.length - 1].role === 'user');
            maxLen = isStepFollowUp && isLastAssistant ? MAX_ASSISTANT_FULL : MAX_ASSISTANT_BRIEF;
        }

        if (msgText.length > maxLen) {
            msgText = `${msgText.substring(0, maxLen)}...`;
        }

        msgText = msgText.replace(/\n{3,}/g, '\n\n').trim();
        const contextLine = `${roleLabel}: ${msgText}`;

        if (totalLength + contextLine.length + currentInput.length + 100 > MAX_TOTAL_LENGTH) {
            if (isStepFollowUp && msg.role === 'assistant' && contextLines.length === 0) {
                contextLines.push(contextLine);
            }
            break;
        }

        contextLines.push(contextLine);
        totalLength += contextLine.length;
    }

    if (contextLines.length === 0) {
        return currentInput;
    }

    if (isStepFollowUp) {
        return [
            'Önceki konuşma:',
            contextLines.join('\n'),
            '',
            `Kullanıcı şimdi yukarıdaki adımlardan biri hakkında soru soruyor: ${currentInput}`,
            '',
            'Lütfen ilgili adımı daha detaylı açıkla.'
        ].join('\n');
    }

    return [
        'Önceki konuşma:',
        contextLines.join('\n'),
        '',
        `Yeni sorum: ${currentInput}`
    ].join('\n');
}

async function submitChatQuery() {
    const queryInput = document.getElementById('query-input');
    const languageSelect = document.getElementById('language-select');
    const submitBtn = document.getElementById('chat-submit-btn');
    const sendBtnText = document.getElementById('send-btn-text');

    const query = queryInput.value.trim();
    const language = languageSelect.value;

    if (!query) {
        return;
    }

    submitBtn.disabled = true;
    queryInput.disabled = true;
    sendBtnText.textContent = 'Gönderiliyor';
    showLoading(true);
    queryInput.value = '';

    const historyForRequest = buildApiConversationHistory(chatMessages);
    chatMessages.push({
        role: 'user',
        text: query,
        timestamp: Date.now()
    });
    saveChatMessages();
    renderChatMessages();

    try {
        const response = await fetch(API_ENDPOINTS.chat, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                query,
                language,
                session_id: getOrCreateSessionId(),
                messages: historyForRequest
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        chatMessages.push({
            role: 'assistant',
            text: data.answer,
            timestamp: Date.now(),
            confidence: data.confidence,
            has_answer: data.has_answer,
            language: data.language,
            sources: data.sources || [],
            debug_info: data.debug_info || null
        });
        saveChatMessages();
        renderChatMessages();
    } catch (error) {
        chatMessages.push({
            role: 'error',
            text: `Bir hata oluştu: ${error.message}. Lütfen tekrar deneyin.`,
            timestamp: Date.now()
        });
        saveChatMessages();
        renderChatMessages();
    } finally {
        submitBtn.disabled = false;
        queryInput.disabled = false;
        sendBtnText.textContent = 'Gönder';
        showLoading(false);
        queryInput.focus();
    }
}

function showLoading(show) {
    const loadingDiv = document.getElementById('chat-loading');
    if (loadingDiv) {
        loadingDiv.classList.toggle('hidden', !show);
    }
}

function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            panels.forEach(panel => {
                panel.classList.toggle('active', panel.id === `${targetTab}-panel`);
            });

            if (targetTab === 'anomaly' && !qualityLoaded) {
                loadAnomalyQuality();
            }
        });
    });
}

async function loadAnomalyQuality() {
    const loadBtn = document.getElementById('load-quality-btn');
    showAnomalyLoading('quality', true);
    hideAnomalyError('quality');
    hideAnomalyResult('quality');
    if (loadBtn) {
        loadBtn.disabled = true;
    }

    try {
        const response = await fetch(API_ENDPOINTS.anomalyQuality, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        displayAnomalyQuality(await response.json());
        qualityLoaded = true;
    } catch (error) {
        showAnomalyError('quality', `Hata: ${error.message}`);
    } finally {
        showAnomalyLoading('quality', false);
        if (loadBtn) {
            loadBtn.disabled = false;
        }
    }
}

function displayAnomalyQuality(data) {
    const resultDiv = document.getElementById('quality-result');
    const summaryDiv = document.getElementById('quality-summary');
    const detailDiv = document.getElementById('quality-detail');
    const eventLevel = data.event_level || {};
    const dayLevel = data.day_level || {};
    const severity = data.severity || {};
    const counts = data.counts || {};

    resultDiv.classList.remove('hidden');
    summaryDiv.innerHTML = `
        ${renderQualityCard('Event precision', eventLevel.precision, 'quality-good')}
        ${renderQualityCard('Event recall', eventLevel.recall, 'quality-strong')}
        ${renderQualityCard('Event F1', eventLevel.f1, 'quality-strong')}
        ${renderQualityCard('Day F1', dayLevel.f1, 'quality-strong')}
        ${renderQualityCard('Specificity', dayLevel.specificity, 'quality-good')}
        ${renderQualityCard('Severity match', severity.exact_match_rate, 'quality-watch')}
    `;

    detailDiv.innerHTML = `
        <div class="quality-note">
            <span>Scope</span>
            <strong>${formatScope(data.metric_scope)}</strong>
        </div>
        <div class="quality-kv-grid">
            ${renderQualityKV('GT events', counts.ground_truth_events)}
            ${renderQualityKV('Detected', counts.detected_events)}
            ${renderQualityKV('Matched', counts.matched_events)}
            ${renderQualityKV('FP candidates', counts.false_positive_candidates)}
            ${renderQualityKV('Positive days', counts.positive_days)}
            ${renderQualityKV('Negative days', counts.negative_days)}
            ${renderQualityKV('Score threshold', dayLevel.score_threshold, 2)}
            ${renderQualityKV('Semantic drift', data.semantic_drift_evaluable ? 'evaluable' : 'not evaluable')}
        </div>
    `;
}

function renderQualityCard(label, value, className) {
    return `
        <div class="quality-card ${className}">
            <div class="quality-label">${label}</div>
            <div class="quality-value">${formatPercent(value)}</div>
        </div>
    `;
}

function renderQualityKV(label, value, fractionDigits = 0) {
    const displayValue = typeof value === 'number'
        ? value.toFixed(fractionDigits)
        : escapeHtml(value ?? '-');
    return `
        <div class="quality-kv">
            <span>${label}</span>
            <strong>${displayValue}</strong>
        </div>
    `;
}

function formatPercent(value) {
    const numeric = Number(value ?? 0);
    return `${Math.round(numeric * 100)}%`;
}

function formatScope(scope) {
    if (scope === 'independent_synthetic_validation_with_negative_days') {
        return 'Independent validation + negative days';
    }
    return escapeHtml(scope || 'validation report');
}

async function loadAnomalyStats() {
    const loadBtn = document.getElementById('load-stats-btn');
    showAnomalyLoading('stats', true);
    hideAnomalyError('stats');
    hideAnomalyResult('stats');
    loadBtn.disabled = true;

    try {
        const response = await fetch(`${API_ENDPOINTS.anomalyStats}?days=7`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        displayAnomalyStats(await response.json());
    } catch (error) {
        showAnomalyError('stats', `Hata: ${error.message}`);
    } finally {
        showAnomalyLoading('stats', false);
        loadBtn.disabled = false;
    }
}

function displayAnomalyStats(data) {
    const resultDiv = document.getElementById('stats-result');
    const summaryDiv = document.getElementById('stats-summary');
    const tableContainer = document.getElementById('stats-table-container');
    const summary = data.summary || {};
    const severityDistribution = summary.severity_distribution || {};
    const reviewCount = severityDistribution.info || 0;
    const warningCount = severityDistribution.warning || 0;
    const criticalCount = severityDistribution.critical || 0;
    const alertCount = warningCount + criticalCount;

    resultDiv.classList.remove('hidden');
    summaryDiv.innerHTML = `
        <div class="summary-item">
            <div class="summary-label">Pencere</div>
            <div class="summary-value">${summary.total_windows || 0}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Ticket</div>
            <div class="summary-value">${summary.total_tickets || 0}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Review candidate</div>
            <div class="summary-value review">${reviewCount}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Alert</div>
            <div class="summary-value alert">${alertCount}</div>
        </div>
    `;

    if (!data.windows || data.windows.length === 0) {
        tableContainer.innerHTML = '<div class="event-empty">Window verisi yok</div>';
        return;
    }

    tableContainer.innerHTML = `
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Başlangıç</th>
                        <th>Ticket</th>
                        <th>Volume Z</th>
                        <th>Category Div</th>
                        <th>Semantic Drift</th>
                        <th>Score</th>
                        <th>Durum</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.windows.map((window, idx) => renderWindowRow(window, idx, data.drift_scores)).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderWindowRow(window, idx, driftScores) {
    const drift = (driftScores && driftScores[idx]) || {};
    const volumeZscore = Number(drift.volume_zscore ?? window.volume_zscore ?? 0);
    const categoryDivergence = Number(drift.category_divergence ?? window.category_divergence ?? 0);
    const semanticDrift = Number(drift.embedding_shift ?? drift.semantic_drift ?? window.semantic_drift ?? 0);
    const combinedScore = Number(drift.combined_score ?? window.combined_score ?? 0);
    const severity = window.severity || 'normal';
    const meta = getSeverityMeta(severity);

    return `
        <tr>
            <td>${formatDate(window.window_start)}</td>
            <td>${window.total_tickets}</td>
            <td>${volumeZscore.toFixed(2)}</td>
            <td>${categoryDivergence.toFixed(3)}</td>
            <td>${semanticDrift.toFixed(3)}</td>
            <td><strong>${combinedScore.toFixed(3)}</strong></td>
            <td><span class="status-chip ${meta.statusClass} severity-${severity}">${meta.shortLabel}</span></td>
        </tr>
    `;
}

async function loadAnomalyEvents() {
    const loadBtn = document.getElementById('load-events-btn');
    showAnomalyLoading('events', true);
    hideAnomalyError('events');
    hideAnomalyResult('events');
    loadBtn.disabled = true;

    try {
        const response = await fetch(`${API_ENDPOINTS.anomalyDetect}?min_severity=info`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        displayAnomalyEvents(await response.json());
    } catch (error) {
        showAnomalyError('events', `Hata: ${error.message}`);
    } finally {
        showAnomalyLoading('events', false);
        loadBtn.disabled = false;
    }
}

function displayAnomalyEvents(data) {
    const resultDiv = document.getElementById('events-result');
    const summaryDiv = document.getElementById('events-summary');
    const eventsList = document.getElementById('events-list');
    const events = data.events || [];
    const reviewEvents = events.filter(event => event.severity === 'info');
    const alertEvents = events.filter(event => event.severity === 'warning' || event.severity === 'critical');
    const criticalCount = alertEvents.filter(event => event.severity === 'critical').length;
    const warningCount = alertEvents.filter(event => event.severity === 'warning').length;

    resultDiv.classList.remove('hidden');
    summaryDiv.innerHTML = `
        <div class="summary-item">
            <div class="summary-label">Toplam pencere</div>
            <div class="summary-value">${data.total_windows || 0}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Review candidate</div>
            <div class="summary-value review">${reviewEvents.length}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Warning alert</div>
            <div class="summary-value warning">${warningCount}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Critical alert</div>
            <div class="summary-value critical">${criticalCount}</div>
        </div>
    `;

    if (events.length === 0) {
        eventsList.innerHTML = '<div class="event-empty">Olay bulunmadı</div>';
        return;
    }

    eventsList.innerHTML = `
        <div class="event-columns">
            ${renderEventColumn('Alerts', alertEvents)}
            ${renderEventColumn('Review candidates', reviewEvents)}
        </div>
    `;
}

function renderEventColumn(title, events) {
    return `
        <section class="event-column">
            <div class="event-column-header">
                <span>${title}</span>
                <span class="event-count">${events.length}</span>
            </div>
            <div class="event-stack">
                ${events.length ? events.map(renderEventCard).join('') : '<div class="event-empty">Kayıt yok</div>'}
            </div>
        </section>
    `;
}

function renderEventCard(event) {
    const severity = event.severity || 'info';
    const meta = getSeverityMeta(severity);
    const reasons = event.reasons || [];

    return `
        <article class="event-item ${meta.statusClass} severity-${severity}">
            <div class="event-header">
                <span class="event-time">${formatDateRange(event.window_start, event.window_end)}</span>
                <span class="severity-badge severity-${severity}">${meta.label}</span>
            </div>
            <div class="event-score">score ${Number(event.score || 0).toFixed(3)}</div>
            <ul class="event-reasons">
                ${reasons.map(reason => `<li>${escapeHtml(reason)}</li>`).join('')}
            </ul>
        </article>
    `;
}

function getSeverityMeta(severity) {
    if (severity === 'critical') {
        return {
            label: 'Critical alert',
            shortLabel: 'Alert',
            statusClass: 'status-alert'
        };
    }

    if (severity === 'warning') {
        return {
            label: 'Alert',
            shortLabel: 'Alert',
            statusClass: 'status-alert'
        };
    }

    if (severity === 'info') {
        return {
            label: 'Review candidate',
            shortLabel: 'Review',
            statusClass: 'status-review'
        };
    }

    return {
        label: 'Normal',
        shortLabel: 'Normal',
        statusClass: 'status-normal'
    };
}

function formatDate(value) {
    if (!value) {
        return '-';
    }
    return new Date(value).toLocaleDateString('tr-TR');
}

function formatDateRange(startValue, endValue) {
    if (!startValue) {
        return '-';
    }

    const start = new Date(startValue);
    const end = endValue ? new Date(new Date(endValue).getTime() - 1) : start;
    const startText = start.toLocaleDateString('tr-TR');
    const endText = end.toLocaleDateString('tr-TR');

    return startText === endText ? startText : `${startText} - ${endText}`;
}

function showAnomalyLoading(section, show) {
    const loadingDiv = document.getElementById(`${section}-loading`);
    if (loadingDiv) {
        loadingDiv.classList.toggle('hidden', !show);
    }
}

function showAnomalyError(section, message) {
    const errorDiv = document.getElementById(`${section}-error`);
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
}

function hideAnomalyError(section) {
    const errorDiv = document.getElementById(`${section}-error`);
    if (errorDiv) {
        errorDiv.classList.add('hidden');
    }
}

function hideAnomalyResult(section) {
    const resultDiv = document.getElementById(`${section}-result`);
    if (resultDiv) {
        resultDiv.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    getOrCreateSessionId();
    loadStoredChatMessages();
    renderChatMessages();

    const chatSubmitBtn = document.getElementById('chat-submit-btn');
    const queryInput = document.getElementById('query-input');
    const loadQualityBtn = document.getElementById('load-quality-btn');
    const loadStatsBtn = document.getElementById('load-stats-btn');
    const loadEventsBtn = document.getElementById('load-events-btn');

    chatSubmitBtn.addEventListener('click', submitChatQuery);
    queryInput.addEventListener('keydown', event => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            submitChatQuery();
        }
    });

    loadQualityBtn.addEventListener('click', loadAnomalyQuality);
    loadStatsBtn.addEventListener('click', loadAnomalyStats);
    loadEventsBtn.addEventListener('click', loadAnomalyEvents);
});
