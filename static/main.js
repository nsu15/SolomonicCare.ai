// --- Font controls ---
const minFontSize = 14;
const maxFontSize = 26;
const defaultFontSize = 16;
let currentFontSize = defaultFontSize;

const form = document.getElementById("support-form");
const submitBtn = document.getElementById("submit-btn");
const loadingSpinner = document.getElementById("loading-spinner");
const ttsToggle = document.getElementById("tts-toggle");
const stopVoiceBtn = document.getElementById("stop-voice-btn");
const resultContainer = document.getElementById("result-container");
const modeSelect = document.getElementById("mode");
const viewHistoryBtn = document.getElementById("view-history-btn");
const backToChatBtn = document.getElementById("back-to-chat-btn");
const historyRange = document.getElementById("history-range");

let controller = null;
let utter = null;
let chatStarted = false;

// --- Font resize ---
function updateFontSize() {
    const aiResponses = resultContainer.querySelectorAll(".ai-response");
    aiResponses.forEach(el => {
        el.style.fontSize = currentFontSize + "px";
        el.style.textIndent = "1.5em";
    });
}
document.getElementById("font-increase").addEventListener("click", () => {
    if (currentFontSize < maxFontSize) {
        currentFontSize++;
        updateFontSize();
    }
});
document.getElementById("font-decrease").addEventListener("click", () => {
    if (currentFontSize > minFontSize) {
        currentFontSize--;
        updateFontSize();
    }
});
document.getElementById("font-reset").addEventListener("click", () => {
    currentFontSize = defaultFontSize;
    updateFontSize();
});

// --- Text-to-speech setup ---
let voices = [];
window.speechSynthesis.onvoiceschanged = () => {
    voices = window.speechSynthesis.getVoices();
};
function unlockTTS() {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    window.removeEventListener('click', unlockTTS);
}
window.addEventListener('click', unlockTTS);

function speakResponse(text) {
    const synth = window.speechSynthesis;
    if (!synth || !text || !ttsToggle.checked) return;
    if (synth.speaking) synth.cancel();

    // Clean bullets & multiple newlines for smooth speech
    let cleanedText = text
        .replace(/(^|\n)[\*\-]\s*/g, ". ")
        .replace(/\n+/g, " ");

    utter = new SpeechSynthesisUtterance(cleanedText);
    utter.lang = "en-US";
    utter.pitch = 1.0;  // Slightly more natural & soothing pitch
    utter.rate = 0.95;

    if (voices.length === 0) voices = synth.getVoices();
    const maleVoice = voices.find(voice =>
        voice.name.toLowerCase().includes("matthew") ||
        voice.name.toLowerCase().includes("daniel") ||
        voice.name.toLowerCase().includes("microsoft david") ||
        voice.name.toLowerCase().includes("google us english")
    ) || voices.find(voice => voice.lang === "en-US");

    if (maleVoice) utter.voice = maleVoice;
    synth.speak(utter);
}
function stopSpeaking() {
    window.speechSynthesis.cancel();
}
stopVoiceBtn.addEventListener("click", stopSpeaking);
window.addEventListener("beforeunload", stopSpeaking);

// --- Chat history ---
function renderChatHistory() {
    let history = JSON.parse(localStorage.getItem("chatHistory") || "[]");
    if (!resultContainer) return;
    resultContainer.innerHTML = `<p><strong>Solomon:</strong></p>`;
    history.forEach(htmlText => {
        const div = document.createElement("div");
        div.className = "ai-response";
        div.style.marginTop = "1em";
        div.style.textIndent = "1.5em";
        div.innerHTML = htmlText;
        resultContainer.appendChild(div);
    });
    updateFontSize();
}

// keep selected mode after reload
let savedMode = localStorage.getItem("selectedMode");
if (savedMode && modeSelect) {
    modeSelect.value = savedMode;
}
renderChatHistory();

// --- View history & back to chat toggle ---
viewHistoryBtn.addEventListener("click", () => {
    fetchHistory();
    historyRange.style.display = "inline-block";
    backToChatBtn.style.display = "inline-block";
    viewHistoryBtn.style.display = "none";
    form.style.display = "none";
});
backToChatBtn.addEventListener("click", () => {
    renderChatHistory();
    historyRange.style.display = "none";
    backToChatBtn.style.display = "none";
    viewHistoryBtn.style.display = "inline-block";
    form.style.display = "block";
});
historyRange.addEventListener("change", fetchHistory);

async function fetchHistory() {
    try {
        const response = await fetch("/api/history");
        const historyData = await response.json();
        let key = "";
        switch (historyRange.value) {
            case "24h": key = "last_1_day"; break;
            case "1w": key = "last_7_days"; break;
            case "2w": key = "last_14_days"; break;
            case "30d": key = "last_30_days"; break;
            default: key = "last_30_days";
        }
        const convos = historyData[key] || [];
        resultContainer.innerHTML = `<strong>Viewing history (${historyRange.options[historyRange.selectedIndex].text}):</strong>`;
        if (convos.length === 0) {
            resultContainer.innerHTML += "<div class='ai-response'>No conversations found in this range.</div>";
        } else {
            convos.forEach(c => {
                const div = document.createElement("div");
                div.className = "ai-response";
                div.style.cursor = "pointer";
                div.style.borderBottom = "1px solid #ccc";
                div.style.padding = "0.3rem 0";
                div.textContent = `${new Date(c.timestamp).toLocaleString()}: ${c.summary}`;
                div.addEventListener("click", () => loadConversation(c.id));
                resultContainer.appendChild(div);
            });
        }
        updateFontSize();
    } catch (e) {
        resultContainer.innerHTML = "<div class='ai-response'>Failed to load history.</div>";
    }
}

async function loadConversation(id) {
    try {
        const response = await fetch("/api/history/load", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ id })
        });
        const convo = await response.json();
        if (convo.error) {
            alert(convo.error);
            return;
        }
        let history = convo.history || [];
        localStorage.setItem("chatHistory", JSON.stringify(history));
        renderChatHistory();
        historyRange.style.display = "none";
        backToChatBtn.style.display = "none";
        viewHistoryBtn.style.display = "inline-block";
        form.style.display = "block";
    } catch {
        alert("Failed to load conversation.");
    }
}

// --- Streaming fetch on form submit ---
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (submitBtn.disabled) return;
    if (!resultContainer) return;

    if (!chatStarted) {
        chatStarted = true;
    }

    // Save current mode
    localStorage.setItem("selectedMode", modeSelect.value);

    resultContainer.innerHTML = `<p><strong>Solomon:</strong></p><div class="ai-response" id="ai-response-text" style="text-indent: 1.5em;"></div>`;
    const aiResponseText = document.getElementById("ai-response-text");

    submitBtn.disabled = true;
    submitBtn.setAttribute("aria-busy", "true");
    loadingSpinner.style.display = "inline-block";

    if (controller) controller.abort();
    controller = new AbortController();

    const formData = new FormData(form);
    const payload = {
        query: formData.get("query"),
        mode: formData.get("mode")
    };

    let fullText = "";

    try {
        const response = await fetch("/api/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });
        if (!response.ok) throw new Error("Network response was not ok");

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        while (!done) {
            const { value, done: doneReading } = await reader.read();
            done = doneReading;
            const chunk = decoder.decode(value || new Uint8Array(), { stream: true });
            fullText += chunk;
            if (aiResponseText) {
                aiResponseText.innerHTML = fullText;
                updateFontSize();
            }
        }

        // Save and re-render chat history
        let history = JSON.parse(localStorage.getItem("chatHistory") || "[]");
        history.push(fullText.trim());
        if (history.length > 25) history = history.slice(-25);
        localStorage.setItem("chatHistory", JSON.stringify(history));
        renderChatHistory();

        if (ttsToggle.checked && fullText.trim()) {
            speakResponse(fullText.trim());
        }
    } catch (error) {
        if (error.name === "AbortError") {
            console.log("Fetch aborted");
        } else {
            console.error("Fetch error:", error);
            if (aiResponseText) aiResponseText.textContent = "Sorry, something went wrong.";
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.removeAttribute("aria-busy");
        loadingSpinner.style.display = "none";
    }
});
