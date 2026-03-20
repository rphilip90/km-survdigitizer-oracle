(function () {
    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function formatValue(value) {
        return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    }

    function initManualCrop() {
        const image = document.getElementById("source-image-preview");
        const stage = document.getElementById("crop-editor-stage");
        const viewport = document.getElementById("crop-editor-viewport");
        const overlay = document.getElementById("crop-editor-overlay");
        const svg = document.getElementById("crop-editor-svg");
        const selection = document.getElementById("crop-selection");
        const polygonGroup = document.getElementById("polygon-mask-group");
        const activePolygon = document.getElementById("active-polygon-mask");
        const previewCanvas = document.getElementById("cropped-preview-canvas");
        const clearCropButton = document.getElementById("clear-manual-crop");
        const cropModeButton = document.getElementById("crop-mode-button");
        const polygonModeButton = document.getElementById("polygon-mode-button");
        const clearLastPolygonButton = document.getElementById("clear-last-polygon-mask");
        const clearAllPolygonsButton = document.getElementById("clear-all-polygon-masks");
        const polygonField = document.getElementById("exclusion-polygons-input");
        const zoomOutButton = document.getElementById("zoom-out-button");
        const zoomInButton = document.getElementById("zoom-in-button");
        const zoomFitButton = document.getElementById("zoom-fit-button");
        const zoomSlider = document.getElementById("zoom-slider");
        const zoomReadout = document.getElementById("zoom-readout");
        const inputs = {
            left: document.getElementById("crop-left-input"),
            top: document.getElementById("crop-top-input"),
            right: document.getElementById("crop-right-input"),
            bottom: document.getElementById("crop-bottom-input"),
        };

        if (
            !image || !stage || !viewport || !overlay || !svg || !selection || !polygonGroup ||
            !activePolygon || !previewCanvas || !clearCropButton || !cropModeButton ||
            !polygonModeButton || !clearLastPolygonButton || !clearAllPolygonsButton ||
            !polygonField || !zoomOutButton || !zoomInButton || !zoomFitButton ||
            !zoomSlider || !zoomReadout || Object.values(inputs).some(function (input) { return !input; })
        ) {
            return;
        }

        let mode = "crop";
        let dragState = null;
        let activePolygonPoints = [];
        let drawState = null;
        let zoomPercent = Number(zoomSlider.value) || 100;
        let fitScale = 1;
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

        function computeFitScale() {
            if (!image.naturalWidth || !image.naturalHeight) {
                return 1;
            }
            const widthScale = Math.max(0.1, (viewport.clientWidth - 8) / image.naturalWidth);
            const heightScale = Math.max(0.1, (viewport.clientHeight - 8) / image.naturalHeight);
            return Math.max(0.1, Math.min(widthScale, heightScale));
        }

        function updateZoomReadout() {
            zoomReadout.textContent = Math.round(zoomPercent) + "%";
        }

        function setMode(nextMode) {
            mode = nextMode;
            cropModeButton.dataset.active = String(mode === "crop");
            polygonModeButton.dataset.active = String(mode === "polygon");
            cropModeButton.setAttribute("aria-pressed", String(mode === "crop"));
            polygonModeButton.setAttribute("aria-pressed", String(mode === "polygon"));
            overlay.dataset.mode = mode;
            clearLastPolygonButton.disabled = polygons.length === 0 && activePolygonPoints.length === 0;
            clearAllPolygonsButton.disabled = polygons.length === 0;
        }

        function clearActivePolygon() {
            activePolygonPoints = [];
            activePolygon.setAttribute("points", "");
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

            if ([leftValue, topValue, rightValue, bottomValue].some(function (value) { return Number.isNaN(value); })) {
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
            clearLastPolygonButton.disabled = polygons.length === 0 && activePolygonPoints.length === 0;
            clearAllPolygonsButton.disabled = polygons.length === 0;
            renderPreview();
        }

        function applyZoom(options) {
            if (!image.naturalWidth || !image.naturalHeight) {
                return;
            }

            const oldWidth = stage.clientWidth || image.clientWidth || 1;
            const oldHeight = stage.clientHeight || image.clientHeight || 1;
            let centerRatioX = 0.5;
            let centerRatioY = 0.5;

            if (options && options.preserveCenter) {
                centerRatioX = clamp((viewport.scrollLeft + (viewport.clientWidth / 2)) / oldWidth, 0, 1);
                centerRatioY = clamp((viewport.scrollTop + (viewport.clientHeight / 2)) / oldHeight, 0, 1);
            }

            fitScale = computeFitScale();
            const displayScale = fitScale * (zoomPercent / 100);
            const width = Math.max(240, Math.round(image.naturalWidth * displayScale));
            const height = Math.max(180, Math.round(image.naturalHeight * displayScale));

            stage.style.width = width + "px";
            stage.style.height = height + "px";
            image.style.width = width + "px";
            image.style.height = height + "px";
            overlay.style.width = width + "px";
            overlay.style.height = height + "px";
            svg.setAttribute("viewBox", "0 0 " + width + " " + height);
            svg.setAttribute("width", String(width));
            svg.setAttribute("height", String(height));
            updateZoomReadout();

            window.requestAnimationFrame(function () {
                syncCropFromInputs();
                renderPolygons();

                if (options && options.preserveCenter) {
                    viewport.scrollLeft = clamp((centerRatioX * width) - (viewport.clientWidth / 2), 0, Math.max(0, width - viewport.clientWidth));
                    viewport.scrollTop = clamp((centerRatioY * height) - (viewport.clientHeight / 2), 0, Math.max(0, height - viewport.clientHeight));
                }
            });
        }

        function setZoom(nextZoom, options) {
            zoomPercent = clamp(nextZoom, Number(zoomSlider.min) || 50, Number(zoomSlider.max) || 300);
            zoomSlider.value = String(Math.round(zoomPercent));
            applyZoom(options);
        }

        svg.addEventListener("pointerdown", function (event) {
            if (mode === "crop") {
                const start = pointPosition(event);
                dragState = { startX: start.x, startY: start.y };
                svg.setPointerCapture(event.pointerId);
                setSelection(start.x, start.y, start.x + 1, start.y + 1);
                event.preventDefault();
                return;
            }

            if (mode === "polygon") {
                const start = pointPosition(event);
                drawState = { points: [start] };
                activePolygonPoints = [start];
                svg.setPointerCapture(event.pointerId);
                renderPolygons();
                event.preventDefault();
            }
        });

        svg.addEventListener("pointermove", function (event) {
            if (mode === "crop" && dragState) {
                const current = pointPosition(event);
                const left = Math.min(dragState.startX, current.x);
                const top = Math.min(dragState.startY, current.y);
                const right = Math.max(dragState.startX, current.x);
                const bottom = Math.max(dragState.startY, current.y);
                setSelection(left, top, right, bottom);
                return;
            }

            if (mode === "polygon" && drawState) {
                const current = pointPosition(event);
                const lastPoint = drawState.points[drawState.points.length - 1];
                const distance = Math.hypot(current.x - lastPoint.x, current.y - lastPoint.y);
                if (distance >= 4) {
                    drawState.points.push(current);
                    activePolygonPoints = drawState.points.slice();
                    renderPolygons();
                }
            }
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
            renderPreview();
        }

        svg.addEventListener("pointerup", finishCrop);
        svg.addEventListener("pointercancel", function () {
            dragState = null;
            drawState = null;
        });

        svg.addEventListener("pointerup", function (event) {
            if (mode !== "polygon" || !drawState) {
                return;
            }
            const current = pointPosition(event);
            if (drawState.points.length === 1) {
                drawState.points.push(current);
            }
            if (drawState.points.length >= 3) {
                polygons.push({
                    label: "manual mask " + (polygons.length + 1),
                    points: drawState.points.map(normalizedPoint),
                });
                writePolygons();
            }
            drawState = null;
            clearActivePolygon();
            renderPolygons();
        });

        cropModeButton.addEventListener("click", function () {
            setMode("crop");
        });

        polygonModeButton.addEventListener("click", function () {
            setMode("polygon");
        });

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
            renderPreview();
        });

        zoomOutButton.addEventListener("click", function () {
            setZoom(zoomPercent - 20, { preserveCenter: true });
        });

        zoomInButton.addEventListener("click", function () {
            setZoom(zoomPercent + 20, { preserveCenter: true });
        });

        zoomFitButton.addEventListener("click", function () {
            setZoom(100);
            viewport.scrollLeft = 0;
            viewport.scrollTop = 0;
        });

        zoomSlider.addEventListener("input", function () {
            setZoom(Number(zoomSlider.value) || 100, { preserveCenter: true });
        });

        viewport.addEventListener("wheel", function (event) {
            if (!event.ctrlKey && !event.metaKey) {
                return;
            }
            event.preventDefault();
            setZoom(zoomPercent + (event.deltaY < 0 ? 10 : -10), { preserveCenter: true });
        }, { passive: false });

        polygonField.addEventListener("input", function () {
            polygons = parsePolygons();
            clearActivePolygon();
            renderPolygons();
        });

        Object.values(inputs).forEach(function (input) {
            input.addEventListener("input", function () {
                syncCropFromInputs();
                renderPreview();
            });
        });

        window.addEventListener("resize", function () {
            applyZoom();
        });

        image.addEventListener("load", function () {
            applyZoom();
        });

        syncCropFromInputs();
        applyZoom();
        setMode("crop");

        function renderPreview() {
            const context = previewCanvas.getContext("2d");
            if (!context || !image.naturalWidth || !image.naturalHeight) {
                return;
            }

            const leftValue = parseFloat(inputs.left.value);
            const topValue = parseFloat(inputs.top.value);
            const rightValue = parseFloat(inputs.right.value);
            const bottomValue = parseFloat(inputs.bottom.value);

            const hasCrop = ![leftValue, topValue, rightValue, bottomValue].some(function (value) { return Number.isNaN(value); });
            const sourceWidth = image.naturalWidth;
            const sourceHeight = image.naturalHeight;

            const sx = hasCrop ? Math.round(clamp(leftValue, 0, 1) * sourceWidth) : 0;
            const sy = hasCrop ? Math.round(clamp(topValue, 0, 1) * sourceHeight) : 0;
            const sw = hasCrop ? Math.max(1, Math.round((clamp(rightValue, 0, 1) - clamp(leftValue, 0, 1)) * sourceWidth)) : sourceWidth;
            const sh = hasCrop ? Math.max(1, Math.round((clamp(bottomValue, 0, 1) - clamp(topValue, 0, 1)) * sourceHeight)) : sourceHeight;

            previewCanvas.width = sw;
            previewCanvas.height = sh;
            context.clearRect(0, 0, sw, sh);
            context.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);

            context.save();
            context.fillStyle = "rgba(255, 255, 255, 0.92)";
            polygons.forEach(function (polygon) {
                const points = polygon.points || [];
                if (points.length < 3) {
                    return;
                }
                context.beginPath();
                points.forEach(function (point, index) {
                    const px = (point.x * sourceWidth) - sx;
                    const py = (point.y * sourceHeight) - sy;
                    if (index === 0) {
                        context.moveTo(px, py);
                    } else {
                        context.lineTo(px, py);
                    }
                });
                context.closePath();
                context.fill();
            });
            context.restore();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initManualCrop);
    } else {
        initManualCrop();
    }
})();
