(function () {
    function initLiveProgress(card) {
        const refreshSeconds = Number(card.dataset.refreshSeconds || 5);
        const countdownNodes = card.querySelectorAll("[data-countdown]");
        const fill = card.querySelector("[data-refresh-fill]");
        const totalMs = Math.max(1000, refreshSeconds * 1000);
        const start = performance.now();

        function render(now) {
            const elapsed = Math.min(totalMs, now - start);
            const remainingMs = Math.max(0, totalMs - elapsed);
            const remainingRatio = remainingMs / totalMs;

            countdownNodes.forEach(function (node) {
                node.textContent = (remainingMs / 1000).toFixed(1) + "s";
            });

            if (fill) {
                fill.style.transform = "scaleX(" + remainingRatio + ")";
            }

            if (remainingMs > 0) {
                window.requestAnimationFrame(render);
                return;
            }

            window.location.reload();
        }

        window.requestAnimationFrame(render);
    }

    function boot() {
        const cards = document.querySelectorAll("[data-live-progress]");
        cards.forEach(initLiveProgress);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
