(function () {
    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function formatValue(value) {
        return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    }

    function initManualCrop() {
        const image = document.getElementById("source-image-preview");
        const overlay = document.getElementById("crop-editor-overlay");
        const svg = document.getElementById("crop-editor-svg");
        const selection = document.getElementById("crop-selection");
        const polygonGroup = document.getElementById("polygon-mask-group");
        const activePolygon = document.getElementById("active-polygon-mask");
        const clearCropButton = document.getElementById("clear-manual-crop");
        const cropModeButton = document.getElementById("crop-mode-button");
        const polygonModeButton = document.getElementById("polygon-mode-button");
        const finishPolygonButton = document.getElementById("finish-polygon-mask");
        const clearLastPolygonButton = document.getElementById("clear-last-polygon-mask");
        const clearAllPolygonsButton = document.getElementById("clear-all-polygon-masks");
        const polygonField = document.getElementById("exclusion-polygons-input");
        const inputs = {
            left: document.getElementById("crop-left-input"),
            top: document.getElementById("crop-top-input"),
            right: document.getElementById("crop-right-input"),
            bottom: document.getElementById("crop-bottom-input"),
        };

        if (
            !image || !overlay || !svg || !selection || !polygonGroup || !activePolygon ||
            !clearCropButton || !cropModeButton || !polygonModeButton ||
            !finishPolygonButton || !clearLastPolygonButton || !clearAllPolygonsButton ||
            !polygonField || Object.values(inputs).some((input) => !input)
        ) {
            return;
        }

        let mode = "crop";
        let dragState = null;
        let activePolygonPoints = [];
        let polygons = parsePolygons();

        function parsePolygons() {
            try {
                const parsed = JSON.parse(polygonField.value || "[]");
                return Array.isArray(parsed) ? parsed : [];
            } catch (error) {
                return [];
            }
        }

        function writePolygons() {
            polygonField.value = JSON.stringify(polygons, null, 2);
        }

        function overlayRect() {
            return overlay.getBoundingClientRect();
        }

        function setMode(nextMode) {
            mode = nextMode;
            cropModeButton.dataset.active = String(mode === "crop");
            polygonModeButton.dataset.active = String(mode === "polygon");
            overlay.dataset.mode = mode;
            finishPolygonButton.disabled = mode !== "polygon" || activePolygonPoints.length < 3;
            clearLastPolygonButton.disabled = polygons.length === 0 && activePolygonPoints.length === 0;
            clearAllPolygonsButton.disabled = polygons.length === 0;
        }

        function clearActivePolygon() {
            activePolygonPoints = [];
            activePolygon.setAttribute("points", "");
            finishPolygonButton.disabled = true;
        }

        function pointPosition(event) {
            const rect = overlayRect();
            return {
                x: clamp(event.clientX - rect.left, 0, rect.width),
                y: clamp(event.clientY - rect.top, 0, rect.height),
            };
        }

        function normalizedPoint(point) {
            const rect = overlayRect();
            return {
                x: Number(formatValue(clamp(point.x / rect.width, 0, 1))),
                y: Number(formatValue(clamp(point.y / rect.height, 0, 1))),
            };
        }

        function denormalizedPoint(point) {
            const rect = overlayRect();
            return {
                x: point.x * rect.width,
                y: point.y * rect.height,
            };
        }

        function setSelection(left, top, right, bottom) {
            selection.style.display = "block";
            selection.setAttribute("x", left);
            selection.setAttribute("y", top);
            selection.setAttribute("width", Math.max(1, right - left));
            selection.setAttribute("height", Math.max(1, bottom - top));
        }

        function hideSelection() {
            selection.style.display = "none";
        }

        function syncCropFromInputs() {
            const leftValue = parseFloat(inputs.left.value);
            const topValue = parseFloat(inputs.top.value);
            const rightValue = parseFloat(inputs.right.value);
            const bottomValue = parseFloat(inputs.bottom.value);

            if ([leftValue, topValue, rightValue, bottomValue].some((value) => Number.isNaN(value))) {
                hideSelection();
                return;
            }

            const rect = overlayRect();
            setSelection(
                leftValue * rect.width,
                topValue * rect.height,
                rightValue * rect.width,
                bottomValue * rect.height
            );
        }

        function updateCropInputs(left, top, right, bottom) {
            const rect = overlayRect();
            inputs.left.value = formatValue(clamp(left / rect.width, 0, 1));
            inputs.top.value = formatValue(clamp(top / rect.height, 0, 1));
            inputs.right.value = formatValue(clamp(right / rect.width, 0, 1));
            inputs.bottom.value = formatValue(clamp(bottom / rect.height, 0, 1));
        }

        function renderPolygons() {
            while (polygonGroup.firstChild) {
                polygonGroup.removeChild(polygonGroup.firstChild);
            }

            polygons.forEach(function (polygon) {
                const polygonNode = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                const points = (polygon.points || []).map(function (point) {
                    const rendered = denormalizedPoint(point);
                    return rendered.x + "," + rendered.y;
                }).join(" ");
                polygonNode.setAttribute("points", points);
                polygonNode.setAttribute("class", "saved-polygon-mask");
                polygonGroup.appendChild(polygonNode);
            });

            const activePoints = activePolygonPoints.map(function (point) {
                return point.x + "," + point.y;
            }).join(" ");
            activePolygon.setAttribute("points", activePoints);
            finishPolygonButton.disabled = mode !== "polygon" || activePolygonPoints.length < 3;
            clearLastPolygonButton.disabled = polygons.length === 0 && activePolygonPoints.length === 0;
            clearAllPolygonsButton.disabled = polygons.length === 0;
        }

        function finishActivePolygon() {
            if (activePolygonPoints.length < 3) {
                return;
            }
            polygons.push({
                label: "manual polygon " + (polygons.length + 1),
                points: activePolygonPoints.map(normalizedPoint),
            });
            writePolygons();
            clearActivePolygon();
            renderPolygons();
        }

        svg.addEventListener("pointerdown", function (event) {
            if (mode !== "crop") {
                return;
            }
            const start = pointPosition(event);
            dragState = { startX: start.x, startY: start.y };
            svg.setPointerCapture(event.pointerId);
            setSelection(start.x, start.y, start.x + 1, start.y + 1);
            event.preventDefault();
        });

        svg.addEventListener("pointermove", function (event) {
            if (mode !== "crop" || !dragState) {
                return;
            }
            const current = pointPosition(event);
            const left = Math.min(dragState.startX, current.x);
            const top = Math.min(dragState.startY, current.y);
            const right = Math.max(dragState.startX, current.x);
            const bottom = Math.max(dragState.startY, current.y);
            setSelection(left, top, right, bottom);
        });

        function finishCrop(event) {
            if (mode !== "crop" || !dragState) {
                return;
            }
            const current = pointPosition(event);
            const left = Math.min(dragState.startX, current.x);
            const top = Math.min(dragState.startY, current.y);
            const right = Math.max(dragState.startX, current.x);
            const bottom = Math.max(dragState.startY, current.y);
            dragState = null;

            if ((right - left) < 12 || (bottom - top) < 12) {
                hideSelection();
                return;
            }

            setSelection(left, top, right, bottom);
            updateCropInputs(left, top, right, bottom);
        }

        svg.addEventListener("pointerup", finishCrop);
        svg.addEventListener("pointercancel", function () {
            dragState = null;
        });

        svg.addEventListener("click", function (event) {
            if (mode !== "polygon") {
                return;
            }
            const point = pointPosition(event);
            activePolygonPoints.push(point);
            renderPolygons();
        });

        cropModeButton.addEventListener("click", function () {
            setMode("crop");
        });

        polygonModeButton.addEventListener("click", function () {
            setMode("polygon");
        });

        finishPolygonButton.addEventListener("click", finishActivePolygon);

        clearLastPolygonButton.addEventListener("click", function () {
            if (activePolygonPoints.length > 0) {
                activePolygonPoints.pop();
                renderPolygons();
                return;
            }
            polygons.pop();
            writePolygons();
            renderPolygons();
        });

        clearAllPolygonsButton.addEventListener("click", function () {
            polygons = [];
            writePolygons();
            clearActivePolygon();
            renderPolygons();
        });

        clearCropButton.addEventListener("click", function () {
            inputs.left.value = "";
            inputs.top.value = "";
            inputs.right.value = "";
            inputs.bottom.value = "";
            hideSelection();
        });

        polygonField.addEventListener("input", function () {
            polygons = parsePolygons();
            clearActivePolygon();
            renderPolygons();
        });

        Object.values(inputs).forEach(function (input) {
            input.addEventListener("input", syncCropFromInputs);
        });

        window.addEventListener("resize", function () {
            syncCropFromInputs();
            renderPolygons();
        });
        image.addEventListener("load", function () {
            syncCropFromInputs();
            renderPolygons();
        });

        syncCropFromInputs();
        renderPolygons();
        setMode("crop");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initManualCrop);
    } else {
        initManualCrop();
    }
})();
