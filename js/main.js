document.addEventListener("DOMContentLoaded", () => {
    console.log("Breast L2–L5 demo (four windows + per-frame overlay + reports) loaded.");

   
    const casesConfig = {
        case1: { grade: "L2", videoId: "Case1", maxFrameIndex: 270 },
        case2: { grade: "L2", videoId: "Case2", maxFrameIndex: 270 },
        case3: { grade: "L3", videoId: "Case1", maxFrameIndex: 270 },
        case4: { grade: "L3", videoId: "Case2", maxFrameIndex: 299 },
        case5: { grade: "L4", videoId: "Case1", maxFrameIndex: 299 },
        case6: { grade: "L4", videoId: "Case2", maxFrameIndex: 300 },
        case7: { grade: "L5", videoId: "Case1", maxFrameIndex: 269 },
        case8: { grade: "L5", videoId: "Case2", maxFrameIndex: 299 }
    };

    function padFrameIndex(idx) {
        return idx.toString().padStart(5, "0");
    }

    function updateCaseFrame(caseKey) {
        const cfg = casesConfig[caseKey];
        if (!cfg) return;

        const slider          = document.getElementById(`${caseKey}-slider`);
        const imageEl         = document.getElementById(`${caseKey}-image`);
        const maskEl          = document.getElementById(`${caseKey}-mask`);
        const overlayFrameEl  = document.getElementById(`${caseKey}-overlay-frame`);
        const labelEl         = document.getElementById(`${caseKey}-frame-label`);
        const tagImage   = document.getElementById(`${caseKey}-frame-tag-image`);
        const tagMask    = document.getElementById(`${caseKey}-frame-tag-mask`);
        const tagOverlay = document.getElementById(`${caseKey}-frame-tag-overlay`);

        if (!slider || !imageEl || !maskEl || !overlayFrameEl || !labelEl) return;

        const frameIndex = parseInt(slider.value, 10) || 0;
        const frameName = padFrameIndex(frameIndex);

        const base = `assets`;
        const imgPath          = `${base}/video_image/${cfg.grade}/${cfg.videoId}/${frameName}.jpg`;
        const maskPath         = `${base}/video_mask/${cfg.grade}/${cfg.videoId}/${frameName}.png`;
        const overlayFramePath = `${base}/overlay_image/${cfg.grade}/${cfg.videoId}/${frameName}.jpg`;

        imageEl.src = imgPath;
        maskEl.src = maskPath;
        overlayFrameEl.src = overlayFramePath;
        labelEl.textContent = frameName;

        const tagText = `Frame: ${frameName}`;
        if (tagImage)   tagImage.textContent = tagText;
        if (tagMask)    tagMask.textContent = tagText;
        if (tagOverlay) tagOverlay.textContent = tagText;
    }

    
    const autoplayInterval = 200;
    const sliderTimers = {};   

    Object.keys(casesConfig).forEach(caseKey => {
        const cfg = casesConfig[caseKey];
        const slider = document.getElementById(`${caseKey}-slider`);
        if (!slider) return;

        slider.min = 0;
        slider.max = cfg.maxFrameIndex;
        slider.value = 0;
        updateCaseFrame(caseKey);

       
        function startSliderAutoPlay() {
            if (sliderTimers[caseKey]) return;
            sliderTimers[caseKey] = setInterval(() => {
                let val = parseInt(slider.value, 10) || 0;
                val = (val + 1) > cfg.maxFrameIndex ? 0 : (val + 1);
                slider.value = val;
                updateCaseFrame(caseKey);
            }, autoplayInterval);
        }
        
        function stopSliderAutoPlay() {
            if (sliderTimers[caseKey]) {
                clearInterval(sliderTimers[caseKey]);
                sliderTimers[caseKey] = null;
                
            }
        }

        
        startSliderAutoPlay();

        
        slider.addEventListener("pointerdown", stopSliderAutoPlay, { once: true });

        
        slider.addEventListener("input", () => {
            updateCaseFrame(caseKey);
        });
    });
});