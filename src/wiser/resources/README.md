Icons used in the toolbars of the Workbench for Imaging Spectroscopy Exploration
and Research (WISER) were downloaded from [Flaticon](https://www.flaticon.com/).
Per the usage policy on this website, these are the authors of the icons used.

Icons made by [Freepik](https://www.flaticon.com/authors/freepik)
from [Flaticon](https://www.flaticon.com/).

*   overview-pane.svg (was earth.svg)
*   spectrum-pane-2.svg (was line-stats.svg)
*   choose-truecolor.svg (was rgb.svg)
*   dataset-info.svg (was information.svg)
*   stretch-builder.svg (was adjust.svg)
*   stretch-builder-2.svg (was bright.svg)
*   clear-all-plots.svg (was interface.svg)
*   configure.svg (was cogwheels.svg)
*   split-view.svg (was writing.svg)
*   bug.svg

Icons made by [Pixel perfect](https://www.flaticon.com/authors/pixel-perfect)
from [Flaticon](https://www.flaticon.com/).

*   stack.svg (was photo.svg)
*   zoom-pane.svg (was image.svg)
*   open-image.svg (was photo.svg)

Icons made by [Payungkead](https://www.flaticon.com/authors/payungkead)
from [Flaticon](https://www.flaticon.com/).

*   spectrum-pane-1.svg (was graphic.svg)

Icons made by [Pixelmeetup](https://www.flaticon.com/authors/pixelmeetup)
from [Flaticon](https://www.flaticon.com/).

*   select.svg (was area.svg)
*   select-2.svg (was shape.svg)

Icons made by [Those Icons](https://www.flaticon.com/authors/those-icons)
from [Flaticon](https://www.flaticon.com/).

*   collect-spectrum.svg (was interface.svg)
*   load-spectra.svg (was interface.svg)

Icons made by [Becris](https://www.flaticon.com/authors/becris)
from [Flaticon](https://www.flaticon.com/).

*   link-scroll.svg (was link.svg)

Icons made by [bqlqn](https://www.flaticon.com/authors/bqlqn)
from [Flaticon](https://www.flaticon.com/).

*   add-roi.svg (was location.svg)

## Theming: adding or using icons

WISER supports light and dark color schemes (see the "Color scheme" setting in
the WISER configuration dialog). To make this work, the toolbar/UI icons are
**monochrome SVGs that are recolored at runtime**: in dark mode they are tinted
to a light color so they stay visible. This recoloring is done on the fly by
`wiser.gui.theme` (rendering the SVG and compositing a tint over its alpha), so
the original SVG files are left untouched — there is no separate light/dark copy
of each icon.

When adding a new icon or using an existing one, follow these rules:

*   **Load icons through `wiser.gui.theme.get_icon()`, not `QIcon()` directly.**
    For example, `theme.get_icon(":/icons/zoom-in.svg")` instead of
    `QIcon(":/icons/zoom-in.svg")`. Only icons loaded through `get_icon()` adapt
    to the active color scheme. (Most toolbar icons go through
    `util.add_toolbar_action()`, which already calls `get_icon()` for you.)

*   **New toolbar icons should be monochrome SVGs**, authored in black. They may
    declare their color as an explicit `fill`/`stroke`, in a `<style>` block, or
    not at all (SVG's default fill is black) — the recoloring handles all three.

*   **Multi-color icons must opt out of tinting** by passing
    `monochrome=False` (e.g. `theme.get_icon(":/icons/choose-truecolor.svg",
    monochrome=False)`); otherwise they would be flattened to a single color.
    Non-SVG icons (such as `wiser.ico`) are always returned unmodified.

*   **Register new SVGs in `resources.qrc`** and rebuild the compiled resources
    (`make generated`) so they are available under the `:/icons/...` prefix.
