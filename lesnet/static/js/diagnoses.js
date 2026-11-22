// "What LesNet recognises": the served model's diagnosis list, grouped by how cautious LesNet is about each.
// Wikipedia summaries are progressive enhancement: if they fail, the names alone still make sense.
(() => {
  "use strict";

  const GROUPS = [
    { key: "malignant", tone: "var(--urgent)", title: "Skin cancers",
      text: "If a photo looks like one of these, LesNet will tell you to get it checked." },
    { key: "not_sure", tone: "var(--refer)", title: "Spots that can go either way",
      text: "Usually harmless, but some can look like early cancers, so LesNet leans towards a check." },
    { key: "benign", tone: "var(--low)", title: "Usually harmless spots",
      text: "Common, non-cancerous spots. Still see a doctor if one changes." },
  ];
  // Wikipedia article titles where the plain label isn't the article name
  const WIKI_TITLE = {
    "nevus": "Melanocytic nevus", "dysplastic nevus": "Dysplastic nevus", "vascular lesion": "Vascular anomaly",
    "pigmented benign keratosis": "Seborrheic keratosis", "solar lentigo": "Liver spot",
  };

  const container = document.getElementById("groups");
  const pretty = (s) => { const t = String(s).replace(/_/g, " ").trim(); return t.charAt(0).toUpperCase() + t.slice(1); };

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
  }

  async function describe(label, target) {
    const plain = String(label).replace(/_/g, " ").trim().toLowerCase();
    const title = WIKI_TITLE[plain] || pretty(plain);
    try {
      const response = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`);
      if (!response.ok) throw new Error("no summary");
      const data = await response.json();
      if (!data.extract) throw new Error("no extract");
      target.replaceChildren(
        el("p", { class: "dx-text", text: data.extract }),
        el("p", { class: "dx-source" },
          document.createTextNode("From Wikipedia. "),
          el("a", { href: data.content_urls ? data.content_urls.desktop.page : "#", rel: "noopener", text: "Read the full article" })));
    } catch (_) {
      target.replaceChildren(el("p", { class: "dx-text", text: "No description is available right now." }));
    }
  }

  function render(labels, buckets) {
    container.replaceChildren();
    for (const group of GROUPS) {
      const members = labels.filter((l) => (buckets[l] || "not_sure") === group.key);
      if (!members.length) continue;
      const list = el("ul", { class: "dx-list" });
      for (const label of members.sort((a, b) => pretty(a).localeCompare(pretty(b)))) {
        const body = el("div", {}, el("p", { class: "dx-text", text: "Loading description…" }));
        const details = el("details", {}, el("summary", { text: pretty(label) }), body);
        details.addEventListener("toggle", () => { if (details.open && !details.dataset.loaded) {
          details.dataset.loaded = "1"; describe(label, body); } });
        list.append(el("li", {}, details));
      }
      const section = el("section", { class: "group" }, el("h2", { text: group.title }), el("p", { text: group.text }), list);
      section.style.setProperty("--tone", group.tone);
      container.append(section);
    }
    if (!container.children.length) showUnavailable();
  }

  function showUnavailable() {
    container.replaceChildren(el("p", { class: "dx-empty",
      text: "The list couldn't be loaded just now. The server may be waking up, so refresh the page in a minute." }));
  }

  async function load(attempt) {
    try {
      const response = await fetch("/labels");
      const data = await response.json();
      if (!data.fine || !data.fine.length) throw new Error("empty");
      render(data.fine, data.buckets || {});
    } catch (_) {
      if (attempt < 4) setTimeout(() => load(attempt + 1), 4000 * (attempt + 1));
      else showUnavailable();
    }
  }
  load(0);
})();
