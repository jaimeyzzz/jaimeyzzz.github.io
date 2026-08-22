(function () {
  function headingId(heading, used) {
    if (heading.id) return heading.id;
    var base = heading.textContent.trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, "-")
      .replace(/^-+|-+$/g, "") || "section";
    var id = base;
    var index = 2;
    while (used.has(id) || document.getElementById(id)) {
      id = base + "-" + index;
      index += 1;
    }
    used.add(id);
    heading.id = id;
    return id;
  }

  function buildToc() {
    var toc = document.getElementById("article-toc");
    var backButton = document.getElementById("back-to-toc");
    var body = document.querySelector(".article__body");
    if (!toc || !backButton || !body) return;

    var headings = Array.prototype.slice.call(body.querySelectorAll("h2, h3"));
    if (headings.length < 2) return;

    var root = toc.querySelector(".article-toc__list");
    var count = toc.querySelector(".article-toc__count");
    var used = new Set();
    var currentChildren = null;

    headings.forEach(function (heading) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.href = "#" + encodeURIComponent(headingId(heading, used));
      link.textContent = heading.textContent.trim();
      item.appendChild(link);

      if (heading.tagName === "H2") {
        item.className = "article-toc__item article-toc__item--h2";
        root.appendChild(item);
        currentChildren = document.createElement("ol");
        currentChildren.className = "article-toc__children";
        item.appendChild(currentChildren);
      } else {
        item.className = "article-toc__item article-toc__item--h3";
        (currentChildren || root).appendChild(item);
      }
    });

    root.querySelectorAll(".article-toc__children:empty").forEach(function (list) {
      list.remove();
    });
    count.textContent = headings.length + " 节";
    toc.hidden = false;

    function updateBackButton() {
      var passedToc = toc.getBoundingClientRect().bottom < 16;
      backButton.hidden = !passedToc;
    }
    backButton.addEventListener("click", function () {
      toc.open = true;
      var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      toc.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    });
    window.addEventListener("scroll", updateBackButton, { passive: true });
    window.addEventListener("resize", updateBackButton);
    updateBackButton();

    var desktop = window.matchMedia("(min-width: 641px)");
    function matchLayout(event) {
      toc.open = event.matches;
    }
    matchLayout(desktop);
    desktop.addEventListener ? desktop.addEventListener("change", matchLayout) : desktop.addListener(matchLayout);

    toc.addEventListener("click", function (event) {
      if (!event.target.closest("a") || desktop.matches) return;
      toc.open = false;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildToc);
  } else {
    buildToc();
  }
}());
