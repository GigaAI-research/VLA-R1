document.addEventListener("DOMContentLoaded", function () {
    // Math
    renderMathInElement(document.body, {
        delimiters: [
            { left: "$", right: "$", display: false },
            { left: "$$", right: "$$", display: true }
        ]
    });
});

// Add a carousel to the page
function createVideoCarousel(videos, targetId, options = {}) {
    const section = document.getElementById(targetId);
    if (!section) {
        console.error(`Element with id '${targetId}' not found.`);
        return;
    }

    const { controls = true, interval = 10000, ratioTemplate = "a4x3demo" } = options;

    const container = document.createElement('div');
    container.className = 'container';

    const row = document.createElement('div');
    row.className = 'row justify-content-center';

    const col = document.createElement('div');
    col.className = 'col-md-12 text-center';

    const carousel = document.createElement('div');
    carousel.className = 'carousel slide';
    carousel.id = targetId + 'Carousel';
    carousel.setAttribute('data-ride', 'carousel');
    carousel.setAttribute('data-interval', interval);

    const indicators = document.createElement('ol');
    indicators.className = 'carousel-indicators';

    const carouselInner = document.createElement('div');
    carouselInner.className = 'carousel-inner';

    videos.forEach((video, index) => {
        const indicator = document.createElement('li');
        indicator.setAttribute('data-target', `#${targetId}Carousel`);
        indicator.setAttribute('data-slide-to', index.toString());
        if (index === 0) indicator.className = 'active';
        indicator.innerHTML = `<span class="indicator-dot"></span>`;
        indicators.appendChild(indicator);

        const carouselItem = document.createElement('div');
        carouselItem.className = `carousel-item ${index === 0 ? 'active' : ''}`;

        const innerRow = document.createElement('div');
        innerRow.className = 'row sm-gutters';

        const innerCol = document.createElement('div');
        innerCol.className = 'col-md-12';

        const videoElement = document.createElement('video');
        videoElement.className = ratioTemplate;
        videoElement.setAttribute('width', '98%');
        videoElement.setAttribute('autoplay', '');
        // related issue: https://6a00999f-abaa-379e-ae7a-6c126b9833cb.kunlun-6swnhz2o-dotu.xyz/link/rfFXnT7NvfEtXoRp?list=shadowrocket
        // videoElement.setAttribute('muted', '');
        videoElement.muted = true;
        videoElement.setAttribute('playsinline', '');
        videoElement.setAttribute('loop', '');
        if (controls) videoElement.setAttribute('controls', '');
        videoElement.setAttribute('poster', video.poster || 'assets/spinner.svg');

        const source = document.createElement('source');
        source.src = video.src;
        source.type = 'video/mp4';
        videoElement.appendChild(source);

        innerCol.appendChild(videoElement);
        innerRow.appendChild(innerCol);
        carouselItem.appendChild(innerRow);
        carouselInner.appendChild(carouselItem);
    });

    const prevControl = document.createElement('a');
    prevControl.className = 'carousel-control-prev';
    prevControl.setAttribute('href', `#${targetId}Carousel`);
    prevControl.setAttribute('role', 'button');
    prevControl.setAttribute('data-slide', 'prev');
    const prevIcon = document.createElement('span');
    prevIcon.className = 'carousel-control-prev-icon';
    prevIcon.setAttribute('aria-hidden', 'true');
    const prevText = document.createElement('span');
    prevText.className = 'sr-only';
    prevText.textContent = 'Previous';
    prevControl.appendChild(prevIcon);
    prevControl.appendChild(prevText);

    const nextControl = document.createElement('a');
    nextControl.className = 'carousel-control-next';
    nextControl.setAttribute('href', `#${targetId}Carousel`);
    nextControl.setAttribute('role', 'button');
    nextControl.setAttribute('data-slide', 'next');
    const nextIcon = document.createElement('span');
    nextIcon.className = 'carousel-control-next-icon';
    nextIcon.setAttribute('aria-hidden', 'true');
    const nextText = document.createElement('span');
    nextText.className = 'sr-only';
    nextText.textContent = 'Next';
    nextControl.appendChild(nextIcon);
    nextControl.appendChild(nextText);

    carousel.appendChild(indicators);
    carousel.appendChild(carouselInner);
    col.appendChild(carousel);
    col.appendChild(prevControl);
    col.appendChild(nextControl);
    row.appendChild(col);
    container.appendChild(row);
    section.appendChild(container);
}

// Add this to the end of the file
// const results_videos = [
//     { src: "video/1.mp4" },
//     { src: "video/2.mp4" },
//     { src: "video/3.mp4" },
//     { src: "video/4.mp4" },
// ];

// createVideoCarousel(results_videos, 'ResultsVideos', { controls: false, interval: 10000 });

// Insert use <div id="ResultsVideos"></div>.