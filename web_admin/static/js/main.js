/* =========================================================
   Face Attendance Admin - Main JavaScript
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Mobile Sidebar Toggle
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebar = document.getElementById("sidebar");

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", function () {
      sidebar.classList.toggle("show");
    });
  }

  // 2. Auto-hide alerts after 5 seconds
  const alerts = document.querySelectorAll(".alert-dismissible");
  alerts.forEach(function (alert) {
    setTimeout(function () {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) {
        bsAlert.close();
      }
    }, 5000);
  });

  // 3. Real-time Clock in Navbar
  const clockElement = document.getElementById("liveClock");
  if (clockElement) {
    function updateClock() {
      const now = new Date();
      const options = {
        weekday: "short",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      };
      clockElement.textContent = now.toLocaleString("vi-VN", options);
    }
    updateClock();
    setInterval(updateClock, 1000);
  }
});
