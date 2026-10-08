/**
 * AuriSearch: Voice Recognition, RAG API Client, and Text-to-Speech Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const queryInput = document.getElementById('queryInput');
  const searchForm = document.getElementById('searchForm');
  const submitBtn = document.getElementById('submitBtn');
  const clearBtn = document.getElementById('clearBtn');
  const micBtn = document.getElementById('micBtn');
  const voiceStatus = document.getElementById('voiceStatus');
  
  const resultsSection = document.getElementById('resultsSection');
  const recapQueryText = document.getElementById('recapQueryText');
  const answerContent = document.getElementById('answerContent');
  const responseEngine = document.getElementById('responseEngine');
  const responseLatency = document.getElementById('responseLatency');
  const sourcesGrid = document.getElementById('sourcesGrid');
  const speakBtn = document.getElementById('speakBtn');
  const copyBtn = document.getElementById('copyBtn');

  const dbStatusText = document.getElementById('dbStatusText');
  const engineBadge = document.getElementById('engineBadge');

  let isListening = false;
  let recognition = null;
  let synth = window.speechSynthesis;
  let currentUtterance = null;
  let isSpeaking = false;

  // 1. Initialize Speech-to-Text (Web Speech API)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isListening = true;
      micBtn.classList.add('listening');
      voiceStatus.textContent = "Listening to your voice... Speak clearly";
      voiceStatus.style.color = "#FFDF73";
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        transcript += event.results[i][0].transcript;
      }
      queryInput.value = transcript;
      toggleClearButton();
    };

    recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      stopListening();
      if (event.error === 'not-allowed') {
        voiceStatus.textContent = "Microphone access denied. Please allow microphone permission.";
      } else {
        voiceStatus.textContent = `Voice recognition error (${event.error}). You can type your query below.`;
      }
      voiceStatus.style.color = "#f87171";
    };

    recognition.onend = () => {
      stopListening();
      if (queryInput.value.trim().length >= 2) {
        voiceStatus.textContent = "Voice captured. Executing RAG search...";
        performSearch(queryInput.value.trim());
      } else {
        voiceStatus.textContent = "Click the golden orb to speak or type your question below";
        voiceStatus.style.color = "var(--text-gold)";
      }
    };
  } else {
    voiceStatus.textContent = "Speech recognition is not supported in this browser. Please use Chrome/Edge or type below.";
  }

  function startListening() {
    if (!recognition) return;
    if (synth && synth.speaking) synth.cancel();
    try {
      recognition.start();
    } catch (e) {
      console.error(e);
    }
  }

  function stopListening() {
    isListening = false;
    micBtn.classList.remove('listening');
  }

  micBtn.addEventListener('click', () => {
    if (!recognition) {
      alert("Web Speech API is not supported in this browser. Please try Chrome, Edge, or Safari.");
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      startListening();
    }
  });

  // 2. Clear Button & Input handling
  function toggleClearButton() {
    clearBtn.style.display = queryInput.value.length > 0 ? 'block' : 'none';
  }

  queryInput.addEventListener('input', toggleClearButton);
  clearBtn.addEventListener('click', () => {
    queryInput.value = '';
    toggleClearButton();
    queryInput.focus();
  });

  // 3. Search Form Submission
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (query) {
      performSearch(query);
    }
  });

  // 4. Quick Chips Click
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      queryInput.value = q;
      toggleClearButton();
      performSearch(q);
    });
  });

  // 5. RAG Search Execution
  async function performSearch(query) {
    if (isListening && recognition) recognition.stop();
    if (synth && synth.speaking) synth.cancel();

    // UI Loading State
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Searching...</span>`;
    voiceStatus.textContent = `Mining vector space for: "${query}"...`;
    voiceStatus.style.color = "var(--gold-bright)";

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Server error occurred');
      }

      renderResults(data);
      voiceStatus.textContent = "Query complete. Click Listen to hear the response aloud.";
      voiceStatus.style.color = "#4ade80";

    } catch (err) {
      console.error(err);
      alert(`Search error: ${err.message}`);
      voiceStatus.textContent = "Search encountered an error. Safe defensive safeguards engaged.";
      voiceStatus.style.color = "#f87171";
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <span>Search RAG</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      `;
    }
  }

  // 6. Render Results & Source Badges
  function renderResults(data) {
    recapQueryText.textContent = data.query;
    responseEngine.textContent = data.engine;
    responseLatency.textContent = `⚡ ${data.latency_ms}ms`;

    // Format simple Markdown into HTML
    answerContent.innerHTML = formatMarkdown(data.answer);

    // Render Sources
    sourcesGrid.innerHTML = '';
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `
          <div class="source-header">
            <span class="source-doc-name" title="${src.source}">📄 ${src.source}</span>
            <span class="source-similarity">${src.similarity}% Match</span>
          </div>
          <p class="source-snippet">${escapeHtml(src.content)}</p>
        `;
        sourcesGrid.appendChild(item);
      });
    } else {
      sourcesGrid.innerHTML = `<p style="font-size: 12px; color: var(--text-muted);">No direct source chunks retrieved.</p>`;
    }

    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // 7. Text-To-Speech (TTS)
  speakBtn.addEventListener('click', () => {
    if (!synth) {
      alert("Speech synthesis is not supported on your browser.");
      return;
    }

    if (synth.speaking) {
      synth.cancel();
      isSpeaking = false;
      speakBtn.querySelector('span').textContent = 'Listen';
      return;
    }

    const textToSpeak = stripMarkdownForSpeech(answerContent.innerText);
    currentUtterance = new SpeechSynthesisUtterance(textToSpeak);
    currentUtterance.rate = 1.0;
    currentUtterance.pitch = 1.0;

    currentUtterance.onstart = () => {
      isSpeaking = true;
      speakBtn.querySelector('span').textContent = 'Stop Audio';
      speakBtn.style.borderColor = 'var(--gold-bright)';
    };

    currentUtterance.onend = () => {
      isSpeaking = false;
      speakBtn.querySelector('span').textContent = 'Listen';
      speakBtn.style.borderColor = 'var(--border-subtle)';
    };

    currentUtterance.onerror = () => {
      isSpeaking = false;
      speakBtn.querySelector('span').textContent = 'Listen';
    };

    synth.speak(currentUtterance);
  });

  // 8. Copy to Clipboard
  copyBtn.addEventListener('click', async () => {
    const text = answerContent.innerText;
    try {
      await navigator.clipboard.writeText(text);
      const span = copyBtn.querySelector('span');
      span.textContent = 'Copied!';
      setTimeout(() => { span.textContent = 'Copy'; }, 2000);
    } catch (e) {
      console.warn("Clipboard write failed", e);
    }
  });

  // 9. Markdown Parser Utility
  function formatMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Bold **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headers ###
    html = html.replace(/^### (.*$)/gim, '<h3 style="color:var(--gold-accent); margin:12px 0 6px;">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 style="color:var(--gold-bright); margin:14px 0 8px;">$1</h2>');
    // Bullet points
    html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gms, '<ul>$1</ul>');
    // Paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    return `<p>${html}</p>`;
  }

  function stripMarkdownForSpeech(text) {
    return text
      .replace(/[#*_`]/g, '')
      .replace(/https?:\/\/\S+/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
  }

  // 10. Fetch System Telemetry & Health on startup
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const h = await res.json();
        dbStatusText.textContent = `Vector DB: ${h.documents_indexed} Chunks`;
        if (h.gemini_configured) {
          engineBadge.textContent = 'Gemini 2.0 (Active)';
        } else {
          engineBadge.textContent = `Ollama (${h.ollama_model})`;
        }
      }
    } catch (e) {
      console.warn("Health check unreachable:", e);
    }
  }

  checkHealth();
});
