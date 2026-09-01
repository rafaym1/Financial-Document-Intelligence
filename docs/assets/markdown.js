// Minimal markdown renderer shared by the chatbot and memos pages -- just enough of the
// subset the backend actually produces (headings, bold, bullet lists, tables) to avoid
// pulling in a full markdown library for a handful of constructs.
(function () {
  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderInline(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");
  }

  function isTableSeparator(line) {
    return /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test((line || "").trim());
  }

  function splitRow(line) {
    return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  }

  function renderTable(lines, startIdx) {
    const header = splitRow(lines[startIdx]);
    let html = "<table><thead><tr>" + header.map((h) => `<th>${renderInline(h)}</th>`).join("") + "</tr></thead><tbody>";
    let i = startIdx + 2;
    while (i < lines.length && lines[i].trim().startsWith("|")) {
      html += "<tr>" + splitRow(lines[i]).map((c) => `<td>${renderInline(c)}</td>`).join("") + "</tr>";
      i++;
    }
    html += "</tbody></table>";
    return { html, nextIdx: i };
  }

  function renderMarkdown(text) {
    const lines = (text || "").split("\n");
    let html = "";
    let inList = false;
    const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };

    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();

      if (!trimmed) { closeList(); continue; }

      if (trimmed.startsWith("|") && isTableSeparator(lines[i + 1])) {
        closeList();
        const { html: tableHtml, nextIdx } = renderTable(lines, i);
        html += tableHtml;
        i = nextIdx - 1;
        continue;
      }

      const heading = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (heading) {
        closeList();
        const level = heading[1].length;
        html += `<h${level}>${renderInline(heading[2])}</h${level}>`;
        continue;
      }

      if (trimmed.startsWith("- ")) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${renderInline(trimmed.slice(2))}</li>`;
        continue;
      }

      closeList();
      html += `<p>${renderInline(trimmed)}</p>`;
    }
    closeList();
    return html || "<p></p>";
  }

  window.FDIMarkdown = { renderMarkdown, escapeHtml, renderInline };
})();
