// Shared behaviour for every page.
// The tag filter is a no-op on pages without #filter-bar.

function toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const floatingBtn = document.getElementById('floating-toggle');
        if (sidebar.style.display === 'none') {
            sidebar.style.display = 'block';
            document.body.classList.remove('sidebar-collapsed');
            floatingBtn.style.display = 'none';
        } else {
            sidebar.style.display = 'none';
            document.body.classList.add('sidebar-collapsed');
            floatingBtn.style.display = 'block';
        }
    }
(function () {
  const items = Array.from(document.querySelectorAll('.post-item'));
  const tags  = Array.from(document.querySelectorAll('.tag'));
  const bar   = document.getElementById('filter-bar');
  if (!bar || !items.length) return;   // post pages have no tag filter
  let active = null;

  function apply(tag) {
    active = (active === tag) ? null : tag;
    items.forEach(item => {
      const own = Array.from(item.querySelectorAll('.tag'))
        .map(t => t.textContent.trim().toLowerCase());
      item.style.display = (!active || own.includes(active)) ? '' : 'none';
    });
    tags.forEach(t => t.classList.toggle(
      'tag-active', active !== null && t.textContent.trim().toLowerCase() === active
    ));
    bar.textContent = active ? 'Filtering by "' + active + '" — show all' : '';
    bar.style.display = active ? 'block' : 'none';
  }

  tags.forEach(t => {
    t.setAttribute('role', 'button');
    t.setAttribute('tabindex', '0');
    const name = t.textContent.trim().toLowerCase();
    t.addEventListener('click', () => apply(name));
    t.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); apply(name); }
    });
  });

  bar.addEventListener('click', () => apply(active));
})();
