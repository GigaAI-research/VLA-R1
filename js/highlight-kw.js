function createHSMRNode() {
    const strong = document.createElement("strong");
    strong.classList.add("hsmr");
    const spanH = document.createElement("span");
    spanH.textContent = "H";
    strong.appendChild(spanH);
    const spanS = document.createElement("span");
    spanS.textContent = "S";
    strong.appendChild(spanS);
    const spanM = document.createElement("span");
    spanM.textContent = "M";
    strong.appendChild(spanM);
    const spanR = document.createElement("span");
    spanR.textContent = "R";
    strong.appendChild(spanR);
    return strong;
}

// Filter.
function replaceHSMR(node) {
    if (node.nodeType === Node.TEXT_NODE) {
        const matches = node.nodeValue.match(/\bHSMR\b/g);
        if (matches) {
        const parent = node.parentNode;
        const fragments = node.nodeValue.split(/\bHSMR\b/);
        // Replace.
        fragments.forEach((fragment, index) => {
            parent.insertBefore(document.createTextNode(fragment), node);
            if (index < matches.length) {
            parent.insertBefore(createHSMRNode(), node);
            }
        });
        parent.removeChild(node);
        }
    }
    else if (node.nodeType === Node.ELEMENT_NODE && !["A", "SCRIPT", "STYLE"].includes(node.tagName)) {
        Array.from(node.childNodes).forEach(replaceHSMR);
    }
}


// Apply - add this line to the main html file.
// replaceHSMR(document.body);