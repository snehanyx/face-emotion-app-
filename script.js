document.getElementById("mode-toggle").addEventListener("click", function () {
  const body = document.body;
  body.classList.toggle("dark-mode");
  body.classList.toggle("light-mode");

  this.textContent = body.classList.contains("dark-mode")
    ? "☀️ Switch Theme"
    : "🌙 Switch Theme";
});
