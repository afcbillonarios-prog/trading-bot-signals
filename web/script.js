document.addEventListener('DOMContentLoaded', function() {
    const signalsContainer = document.getElementById('signals-container');
    const lastUpdateEl = document.getElementById('last-update');
    
    async function fetchSignals() {
        try {
            const response = await fetch('signals.json');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            updateLastUpdate(data.last_updated);
            displaySignals(data.signals);
        } catch (error) {
            console.error('Error fetching signals:', error);
            signalsContainer.innerHTML = '<p class="error">Error loading signals. Please try again later.</p>';
        }
    }
    
    function updateLastUpdate(timestamp) {
        const date = new Date(timestamp);
        lastUpdateEl.textContent = `Última actualización: ${date.toLocaleString()}`;
    }
    
    function displaySignals(signals) {
        if (signals.length === 0) {
            signalsContainer.innerHTML = '<p>No hay señales disponibles en este momento.</p>';
            return;
        }
        
        signalsContainer.innerHTML = signals.map(signal => {
            const date = new Date(signal.timestamp);
            return `
                <div class="signal-card ${signal.direction.toLowerCase()}">
                    <div class="signal-header">
                        <h2>${signal.direction} ${signal.symbol.replace('/', '')}</h2>
                        <div class="confidence">${(signal.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div class="signal-body">
                        <div class="price-info">
                            <div class="price-item">
                                <label>Entrada</label>
                                <div class="value">$${signal.entry_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                            <div class="price-item">
                                <label>Stop Loss</label>
                                <div class="value">$${signal.stop_loss.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                            <div class="price-item">
                                <label>Take Profit</label>
                                <div class="value">$${signal.take_profit.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                            <div class="price-item">
                                <label>Tamaño Posición</label>
                                <div class="value">${signal.position_size.toFixed(6)}</div>
                            </div>
                            <div class="price-item">
                                <label>Riesgo</label>
                                <div class="value">$${signal.risk_amount.toFixed(2)}</div>
                            </div>
                        </div>
                        <div class="indicators">
                            <div class="indicator-item">
                                <label>EMA20</label>
                                <div class="value">$${signal.indicators.ema20.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                            <div class="indicator-item">
                                <label>EMA50</label>
                                <div class="value">$${signal.indicators.ema50.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                            <div class="indicator-item">
                                <label>RSI</label>
                                <div class="value">${signal.indicators.rsi.toFixed(1)}</div>
                            </div>
                            <div class="indicator-item">
                                <label>ATR</label>
                                <div class="value">$${signal.indicators.atr.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                            </div>
                        </div>
                        <div class="timestamp">
                            <small>${date.toLocaleString()}</small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    // Initial load
    fetchSignals();
    
    // Refresh every 30 seconds
    setInterval(fetchSignals, 30000);
});